"""Candidate API v1. Inputs contain observations and query windows, never truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

API_VERSION = 1
UPSERT = 0
DELETE = 1


@dataclass(frozen=True)
class Problem:
    n_assets: int
    mode: Literal["offline", "online"]
    time_unit: str = "arbitrary"


@dataclass(frozen=True)
class Observations:
    """Current point table. Keys are (asset, time); levels are latent-price proxies."""
    time: np.ndarray
    asset: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class Events:
    """Arrival-ordered point edits. DELETE ignores value; UPSERT replaces a key."""
    time: np.ndarray
    asset: np.ndarray
    value: np.ndarray
    operation: np.ndarray


@dataclass(frozen=True)
class Query:
    """Moments of the latent increment X(end)-X(start), with data available at asof.

    Historical queries estimate marginal moments over an observed window.
    Forecast queries predict moments over the subsequent window. Query windows
    use physical time, not observation indices. Observation noise is excluded.
    """
    track: Literal["historical", "forecast"]
    asof: float
    start: float
    end: float


class Matrix(Protocol):
    n_assets: int
    def diagonal(self) -> np.ndarray: ...
    def entries(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray: ...
    def matvec(self, vectors: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DenseMatrix:
    value: np.ndarray

    @property
    def n_assets(self):
        return self.value.shape[0]

    def diagonal(self):
        return np.diag(self.value).copy()

    def entries(self, rows, cols):
        return self.value[rows, cols]

    def matvec(self, vectors):
        return self.value @ vectors


@dataclass(frozen=True)
class FactorMatrix:
    """Covariance = loadings @ loadings.T + diag(specific)."""
    loadings: np.ndarray
    specific: np.ndarray

    @property
    def n_assets(self):
        return self.specific.size

    def diagonal(self):
        return np.einsum("ij,ij->i", self.loadings, self.loadings) + self.specific

    def entries(self, rows, cols):
        return np.einsum("ij,ij->i", self.loadings[rows], self.loadings[cols]) + (rows == cols) * self.specific[rows]

    def matvec(self, vectors):
        d = self.specific if vectors.ndim == 1 else self.specific[:, None]
        return self.loadings @ (self.loadings.T @ vectors) + d * vectors


@dataclass(frozen=True)
class Prediction:
    """Mean and covariance of the requested increment, in level units.

    Second moment is covariance + outer(mean, mean). Outputs must be complete,
    finite host NumPy arrays in DenseMatrix or FactorMatrix. Dense estimates may
    be indefinite: that is diagnosed, never silently repaired by the evaluator.
    Factor specific variances must be nonnegative.
    """
    mean: np.ndarray
    covariance: DenseMatrix | FactorMatrix


class Estimator(Protocol):
    def fit(self, observations: Observations) -> None: ...
    def update(self, events: Events) -> None: ...
    def predict(self, query: Query) -> Prediction: ...
    def synchronize(self) -> None:
        """Complete outstanding device work; CPU implementations use a no-op."""
        ...
