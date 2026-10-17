"""Synthetic scenario generator and exact moment integration on a discrete master grid.

Generates latent level paths X(t) with piecewise-constant regime parameters
(drift, factor loadings B, and specific variance rates d) under Gaussian,
elliptical Student-t (df > 8), or compound Poisson jump-diffusion innovations,
together with asynchronous noisy observations, arrival latencies, corrections,
and deletions.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .api import DELETE, UPSERT, Events, FactorMatrix, Observations, Prediction, Query


@dataclass(frozen=True)
class Scenario:
    """Scenario specification pairing a unique string identifier with a config dictionary."""
    id: str
    config: dict

    @property
    def n_assets(self) -> int:
        return int(self.config["n_assets"])

    @classmethod
    def from_dict(cls, value: dict) -> "Scenario":
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError("scenario requires a string id")
        config = value.get("config", {k: v for k, v in value.items() if k != "id"})
        if not isinstance(config, dict):
            raise ValueError("scenario config must be an object")
        json.dumps(config, allow_nan=False)
        return cls(value["id"], copy.deepcopy(config))


@dataclass(frozen=True)
class Dataset:
    """Generated synthetic instance containing arrival-ordered events, queries, and exact truth."""
    id: str
    n_assets: int
    events: Events
    arrival: np.ndarray
    queries: tuple[Query, ...]
    truth: tuple[Prediction, ...]
    scenario: Scenario
    seed: int

    def observations_at(self, asof: float) -> Observations:
        """Materialize the active point table by replaying all events with ``arrival <= asof``."""
        if not np.isfinite(asof):
            raise ValueError("asof must be finite")
        stop = int(np.searchsorted(self.arrival, asof, side="right"))
        live = {}
        e = self.events
        for index in range(stop):
            key = (int(e.asset[index]), float(e.time[index]))
            if e.operation[index] == UPSERT:
                live[key] = float(e.value[index])
            else:
                live.pop(key, None)
        ordered = sorted(live.items(), key=lambda item: (item[0][1], item[0][0]))
        return Observations(
            _readonly(np.array([key[1] for key, _ in ordered], dtype=np.float64)),
            _readonly(np.array([key[0] for key, _ in ordered], dtype=np.int64)),
            _readonly(np.array([value for _, value in ordered], dtype=np.float64)),
        )


@dataclass(frozen=True)
class _Regime:
    first_step: int
    last_step: int
    loadings: np.ndarray
    specific: np.ndarray
    drift: np.ndarray


def _readonly(value):
    value.setflags(write=False)
    return value


def _positive(config, key, default):
    value = float(config.get(key, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _fraction(config, key, default):
    value = float(config.get(key, default))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{key} must be between zero and one")
    return value


def _integer(config, key, default, minimum):
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _draw_regime(rng, n, rank, first, last, condition, drift_strength, horizon):
    """Draw drift rates, factor loadings B, and positive specific variance rates d for one regime."""
    # Draw log-uniform marginal variance rates v_i in [condition^(-1/2), condition^(1/2)].
    variance = np.exp(rng.uniform(-np.log(condition) / 2, np.log(condition) / 2, n))
    specific_share = rng.uniform(.2, .8, n)
    if rank:
        directions = rng.normal(size=(n, rank))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        loadings = directions * np.sqrt(variance * (1 - specific_share))[:, None]
        specific = variance * specific_share
    else:
        loadings = np.empty((n, 0))
        specific = variance
    drift = rng.normal(size=n) * drift_strength * np.sqrt(variance / horizon)
    return _Regime(first, last, loadings, specific, drift)


def _latent(rng, regimes, n, steps, horizon, distribution, df, jump_fraction, jump_rate):
    """Simulate the latent level path X(t_s) on the master time grid for s = 0, ..., steps."""
    dt = horizon / steps
    increments = np.empty((steps, n))
    for regime in regimes:
        count = regime.last_step - regime.first_step
        rank = regime.loadings.shape[1]
        factor = rng.normal(size=(count, rank)) @ regime.loadings.T
        specific = rng.normal(size=(count, n)) * np.sqrt(regime.specific)
        if distribution == "student_t":
            radial = np.sqrt((df - 2) / rng.chisquare(df, count))[:, None]
            stochastic = (factor + specific) * radial * np.sqrt(dt)
        elif distribution == "compound_poisson":
            stochastic = (factor + specific) * np.sqrt((1 - jump_fraction) * dt)
            common_count = rng.poisson(jump_rate * dt, size=(count, rank))
            common = rng.normal(size=(count, rank)) * np.sqrt(common_count)
            stochastic += (common @ regime.loadings.T) * np.sqrt(jump_fraction / jump_rate)
            specific_count = rng.poisson(jump_rate * dt, size=(count, n))
            stochastic += (rng.normal(size=(count, n)) * np.sqrt(specific_count)
                           * np.sqrt(regime.specific * jump_fraction / jump_rate))
        else:
            stochastic = (factor + specific) * np.sqrt(dt)
        increments[regime.first_step:regime.last_step] = stochastic + regime.drift * dt
    return np.vstack([np.zeros((1, n)), np.cumsum(increments, axis=0)])


def _sample_indices(rng, n, steps, sampling, fraction, imbalance):
    """Sample sorted grid step indices in {0, ..., steps} for each asset."""
    if sampling == "synchronous":
        interior = np.flatnonzero(rng.random(steps - 1) < fraction) + 1
        common = np.r_[0, interior, steps]
        return [common.copy() for _ in range(n)]
    if sampling == "imbalanced":
        rates = fraction * np.exp(rng.uniform(-np.log(imbalance), 0, n))
    else:
        rates = np.full(n, fraction)
    result = []
    for asset in range(n):
        probability = np.full(steps - 1, rates[asset])
        if sampling == "bursty":
            centers = rng.choice(np.arange(1, steps), size=2, replace=False)
            distance = np.min(np.abs(np.arange(1, steps)[:, None] - centers), axis=1)
            probability *= np.where(distance < max(2, steps // 12), 3., .1)
        interior = np.flatnonzero(rng.random(steps - 1) < np.minimum(probability, 1)) + 1
        result.append(np.r_[0, interior, steps])
    return result


def _integrated(regimes, grid, query, n):
    """Integrate exact latent increment moments over [query.start, query.end] in factor form."""
    mean = np.zeros(n)
    specific = np.zeros(n)
    blocks = []
    for regime in regimes:
        overlap = max(0., min(query.end, grid[regime.last_step])
                      - max(query.start, grid[regime.first_step]))
        if overlap:
            mean += overlap * regime.drift
            specific += overlap * regime.specific
            blocks.append(np.sqrt(overlap) * regime.loadings)
    loadings = np.concatenate(blocks, axis=1) if blocks else np.empty((n, 0))
    return Prediction(_readonly(mean), FactorMatrix(_readonly(loadings), _readonly(specific)))


def generate(scenario: Scenario, seed: int) -> Dataset:
    """Generate a synthetic Dataset for ``scenario`` from a 64-bit integer ``seed``."""
    config = scenario.config
    n = _integer(config, "n_assets", None, 1)
    steps = _integer(config, "grid_steps", 256, 16)
    rank = _integer(config, "rank", min(4, n), 0)
    horizon = _positive(config, "horizon", 1.)
    condition = _positive(config, "condition", 10.)
    if condition < 1:
        raise ValueError("condition must be at least one")
    drift_strength = float(config.get("drift_strength", .2))
    if not np.isfinite(drift_strength) or drift_strength < 0:
        raise ValueError("drift_strength must be finite and nonnegative")
    regime_count = _integer(config, "regimes", 1, 1)
    if regime_count > steps:
        raise ValueError("regimes cannot exceed grid_steps")
    distribution = config.get("distribution", "gaussian")
    if distribution not in {"gaussian", "student_t", "compound_poisson"}:
        raise ValueError("unknown distribution")
    df = _positive(config, "df", 10.)
    if distribution == "student_t" and df <= 8:
        raise ValueError("student_t df must exceed eight")
    jump_fraction = _fraction(config, "jump_fraction", .35)
    jump_rate = _positive(config, "jump_rate", 8.)
    sampling = config.get("sampling", "asynchronous")
    if sampling not in {"synchronous", "asynchronous", "sparse", "imbalanced", "bursty"}:
        raise ValueError("unknown sampling regime")
    fraction = _fraction(config, "observation_fraction", .2 if sampling != "sparse" else .03)
    imbalance = _positive(config, "imbalance", 32.)
    if imbalance < 1:
        raise ValueError("imbalance must be at least one")
    noise = config.get("noise", "none")
    if noise not in {"none", "iid", "correlated", "ar"}:
        raise ValueError("unknown observation noise")
    noise_scale = float(config.get("noise_scale", .03))
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative")
    noise_correlation = _fraction(config, "noise_correlation", .7)
    ar_rho = float(config.get("ar_rho", .7))
    if not np.isfinite(ar_rho) or abs(ar_rho) >= 1:
        raise ValueError("ar_rho must have magnitude below one")
    max_latency = float(config.get("max_latency", 0.))
    if not np.isfinite(max_latency) or max_latency < 0:
        raise ValueError("max_latency must be finite and nonnegative")
    correction_fraction = _fraction(config, "correction_fraction", 0.)
    delete_fraction = _fraction(config, "delete_fraction", 0.)
    if correction_fraction + delete_fraction > 1:
        raise ValueError("correction_fraction + delete_fraction must not exceed one")

    rng = np.random.default_rng(seed)
    grid = np.linspace(0., horizon, steps + 1)
    boundaries = np.linspace(0, steps, regime_count + 1, dtype=int)
    regimes = tuple(_draw_regime(rng, n, rank, int(first), int(last), condition,
                                 drift_strength, horizon)
                    for first, last in zip(boundaries[:-1], boundaries[1:]))
    latent = _latent(rng, regimes, n, steps, horizon, distribution, df, jump_fraction, jump_rate)
    indices = _sample_indices(rng, n, steps, sampling, fraction, imbalance)
    initial_variance = regimes[0].specific + np.einsum("ij,ij->i", regimes[0].loadings, regimes[0].loadings)
    sigma = noise_scale * np.sqrt(initial_variance * horizon)
    common_noise = rng.normal(size=(steps + 1, 2))
    noise_directions = rng.normal(size=(n, 2))
    noise_directions /= np.linalg.norm(noise_directions, axis=1)[:, None]
    records = []
    for asset, selected in enumerate(indices):
        values = latent[selected, asset].copy()
        if noise == "iid":
            values += rng.normal(size=len(selected)) * sigma[asset]
        elif noise == "correlated":
            shared = common_noise[selected] @ noise_directions[asset]
            values += sigma[asset] * (np.sqrt(noise_correlation) * shared
                                     + np.sqrt(1 - noise_correlation) * rng.normal(size=len(selected)))
        elif noise == "ar":
            innovation = rng.normal(size=len(selected)) * sigma[asset]
            errors = np.empty(len(selected))
            errors[0] = innovation[0]
            for index in range(1, len(selected)):
                errors[index] = ar_rho * errors[index - 1] + np.sqrt(1 - ar_rho ** 2) * innovation[index]
            values += errors
        for index, value in zip(selected, values):
            observed = float(grid[index])
            arrival = observed + rng.uniform(0., max_latency)
            kind = rng.random()
            if kind < correction_fraction:
                wrong = value + rng.normal() * .1 * np.sqrt(initial_variance[asset] * horizon)
                records.append((arrival, observed, asset, float(wrong), UPSERT))
                delay = horizon * rng.uniform(.01, .08)
                records.append((arrival + delay, observed, asset, float(value), UPSERT))
            else:
                records.append((arrival, observed, asset, float(value), UPSERT))
                if kind < correction_fraction + delete_fraction:
                    delay = horizon * rng.uniform(.01, .08)
                    records.append((arrival + delay, observed, asset, 0., DELETE))
    records.sort(key=lambda record: record[0])
    arrival = _readonly(np.array([record[0] for record in records], dtype=np.float64))
    events = Events(
        _readonly(np.array([record[1] for record in records], dtype=np.float64)),
        _readonly(np.array([record[2] for record in records], dtype=np.int64)),
        _readonly(np.array([record[3] for record in records], dtype=np.float64)),
        _readonly(np.array([record[4] for record in records], dtype=np.int8)),
    )
    width = max(2, steps // 6)
    checkpoints = [steps // 3, 3 * steps // 5, 4 * steps // 5]
    queries = tuple(query for index in checkpoints for query in (
        Query("historical", float(grid[index]), float(grid[index - width]), float(grid[index])),
        Query("forecast", float(grid[index]), float(grid[index]), float(grid[index + width])),
    ))
    truth = tuple(_integrated(regimes, grid, query, n) for query in queries)
    return Dataset(f"{scenario.id}:seed={seed}", n, events, arrival, queries, truth, scenario, int(seed))


def load_suite(path) -> list[Scenario]:
    """Load and validate a list of Scenario objects from a suite JSON file."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if data.get("version", 1) != 1:
            raise ValueError("unsupported suite version")
        data = data.get("scenarios")
    if not isinstance(data, list):
        raise ValueError("suite must contain a scenario list")
    scenarios = [Scenario.from_dict(value) for value in data]
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("scenario ids must be unique")
    return scenarios
