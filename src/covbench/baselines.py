"""Observation-only baselines; all window refitting is charged to predict."""
from __future__ import annotations

import numpy as np

from .api import DELETE, UPSERT, DenseMatrix, FactorMatrix, Prediction


class ZeroEstimator:
    def __init__(self, problem, config):
        self.n = problem.n_assets

    def fit(self, observations):
        pass

    def update(self, events):
        pass

    def predict(self, query):
        return Prediction(np.zeros(self.n), FactorMatrix(np.zeros((self.n, 0)), np.zeros(self.n)))

    def synchronize(self):
        pass


class _PointEstimator:
    def __init__(self, problem, config):
        self.n = problem.n_assets
        self.config = dict(config)
        self.lookback = float(config.get("lookback", 0.25))
        self.floor = float(config.get("floor", 1e-8))
        if not np.isfinite(self.lookback) or self.lookback <= 0:
            raise ValueError("lookback must be finite and positive")
        if not np.isfinite(self.floor) or self.floor < 0:
            raise ValueError("floor must be finite and nonnegative")
        self.points = [{} for _ in range(self.n)]

    def fit(self, observations):
        self.points = [{} for _ in range(self.n)]
        for time, asset, value in zip(observations.time, observations.asset, observations.value, strict=True):
            self.points[int(asset)][float(time)] = float(value)

    def update(self, events):
        for time, asset, value, operation in zip(
            events.time, events.asset, events.value, events.operation, strict=True
        ):
            points = self.points[int(asset)]
            if operation == UPSERT:
                points[float(time)] = float(value)
            elif operation == DELETE:
                points.pop(float(time), None)
            else:
                raise ValueError("unknown point operation")

    def _window(self, query, *, anchor=False):
        duration = float(query.end - query.start)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("query duration must be finite and positive")
        if query.track == "historical":
            lo, hi = float(query.start), min(float(query.end), float(query.asof))
        elif query.track == "forecast":
            lo, hi = float(query.asof) - self.lookback, float(query.asof)
        else:
            raise ValueError("unknown query track")
        series = []
        for points in self.points:
            selected = sorted((t, v) for t, v in points.items() if lo <= t <= hi)
            if anchor:
                prior = [(t, v) for t, v in points.items() if t < lo and t <= hi]
                if prior:
                    selected.insert(0, max(prior))
            array = np.asarray(selected, dtype=float).reshape(-1, 2)
            series.append((array[:, 0], array[:, 1]))
        return lo, hi, duration, series

    def synchronize(self):
        pass


class PreviousTickEstimator(_PointEstimator):
    """Previous-tick grid returns; missing return residuals contribute zero."""

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.bins = int(config.get("bins", 32))
        if self.bins < 3:
            raise ValueError("bins must contain at least three grid points")

    def predict(self, query):
        lo, hi, duration, series = self._window(query, anchor=True)
        mean_rate = np.zeros(self.n)
        loadings = np.zeros((self.n, self.bins - 1))
        if hi > lo:
            grid = np.linspace(lo, hi, self.bins)
            dt = (hi - lo) / (self.bins - 1)
            for asset, (times, values) in enumerate(series):
                if len(times) < 2:
                    continue
                indices = np.searchsorted(times, grid, side="right") - 1
                levels = np.full(self.bins, np.nan)
                known = indices >= 0
                levels[known] = values[indices[known]]
                returns = np.diff(levels)
                valid = np.isfinite(returns)
                count = int(valid.sum())
                if not count:
                    continue
                mean_return = returns[valid].mean()
                mean_rate[asset] = mean_return / dt
                if count > 1:
                    loadings[asset, valid] = (returns[valid] - mean_return) * np.sqrt(
                        duration / ((count - 1) * dt)
                    )
        return Prediction(
            mean_rate * duration,
            FactorMatrix(loadings, np.full(self.n, self.floor * duration)),
        )


class AsyncNativeEstimator(_PointEstimator):
    """Native overlap refit, estimated drift, assumed zero observation noise."""

    def __init__(self, problem, config):
        super().__init__(problem, config)
        from async_covariance import fit

        self.native_fit = fit
        self.threads = int(config.get("threads", 1))
        if self.threads < 1 or self.floor <= 0:
            raise ValueError("native baseline requires positive threads and floor")
        self.edges = None
        if self.n > 256:
            degree = int(config.get("degree", 8))
            if degree < 0 or degree % 2:
                raise ValueError("ring degree must be nonnegative and even")
            self.edges = sorted({
                tuple(sorted((i, (i + step) % self.n)))
                for i in range(self.n)
                for step in range(1, min(degree // 2, self.n - 1) + 1)
                if i != (i + step) % self.n
            })

    def predict(self, query):
        _, _, duration, series = self._window(query)
        drift = np.zeros(self.n)
        records = []
        for asset, (times, values) in enumerate(series):
            if len(times) >= 2:
                drift[asset] = (values[-1] - values[0]) / (times[-1] - times[0])
            records.extend((t, asset, v) for t, v in zip(times, values, strict=True))
        table = np.asarray(records, dtype=float).reshape(-1, 3)
        snapshot = self.native_fit(
            table[:, 0], table[:, 1].astype(np.int64), table[:, 2],
            n_assets=self.n, method="overlap", drift=drift, noise_variance=0.0,
            edges=self.edges, threads=self.threads,
        )
        if self.n <= 256:
            covariance = snapshot.dense().copy()
            covariance[np.isnan(covariance)] = 0.0
            diagonal = np.diag_indices(self.n)
            covariance[diagonal] = np.maximum(covariance[diagonal], self.floor)
            matrix = DenseMatrix(covariance * duration)
        else:
            variance = np.where(
                np.isnan(snapshot.variance), self.floor,
                np.maximum(snapshot.variance, self.floor),
            )
            result = snapshot.factor(
                rank=min(int(self.config.get("rank", 8)), self.n - 1),
                variance=variance,
                specific_floor=float(self.config.get("specific_floor", 0.05)),
                max_iter=int(self.config.get("max_iter", 100)),
                seed=int(self.config.get("seed", 0)),
            )
            matrix = FactorMatrix(
                np.asarray(result.model.loadings).copy() * np.sqrt(duration),
                np.asarray(result.model.specific).copy() * duration,
            )
        return Prediction(drift * duration, matrix)


def create_zero(problem, config):
    return ZeroEstimator(problem, config)


def create_previous_tick(problem, config):
    return PreviousTickEstimator(problem, config)


def create_async_native(problem, config):
    return AsyncNativeEstimator(problem, config)
