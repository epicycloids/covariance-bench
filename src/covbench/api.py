"""Public candidate interface (API v1) for covariance-bench estimators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

API_VERSION = 1
UPSERT = 0
DELETE = 1


@dataclass(frozen=True)
class Problem:
    """Static problem metadata passed to a candidate's ``create(problem, config)`` factory.

    Attributes:
        n_assets: Number of assets N in the universe (assets are indexed 0, ..., N - 1).
        mode: Execution mode, either ``"offline"`` (table snapshots via ``fit``) or
            ``"online"`` (arrival-ordered point edits via ``update``).
        time_unit: Label describing the physical time unit (default ``"arbitrary"``).
    """
    n_assets: int
    mode: Literal["offline", "online"]
    time_unit: str = "arbitrary"


@dataclass(frozen=True)
class Observations:
    """Materialized point table visible at a cutoff timestamp ``asof``.

    Each entry represents an active observation keyed by ``(asset[k], time[k])``
    after replaying all events arriving on or before ``asof``. Arrays are 1D,
    have equal length, and are sorted lexicographically by ``(time, asset)``.

    Attributes:
        time: Physical measurement timestamps (float64).
        asset: Asset indices in ``{0, ..., n_assets - 1}`` (int64).
        value: Observed level values in estimation units (float64).
    """
    time: np.ndarray
    asset: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class Events:
    """Arrival-ordered batch of point table edits for online updates.

    Edits apply to a finite map keyed by ``(asset[m], time[m])``:
    - ``UPSERT`` (0) inserts or overwrites the value for key ``(asset[m], time[m])``.
    - ``DELETE`` (1) removes key ``(asset[m], time[m])`` if present and ignores ``value[m]``.

    Attributes:
        time: Physical measurement timestamps of the edited points (float64).
        asset: Asset indices in ``{0, ..., n_assets - 1}`` (int64).
        value: Reported level values for ``UPSERT`` operations (float64).
        operation: Edit operation codes, ``UPSERT`` (0) or ``DELETE`` (1) (int8).
    """
    time: np.ndarray
    asset: np.ndarray
    value: np.ndarray
    operation: np.ndarray


@dataclass(frozen=True)
class Query:
    """Request for the mean and covariance of the latent increment X(end) - X(start).

    Historical queries estimate the marginal moments of the latent increment over
    an observed window (rather than posterior moments of the realized path
    conditional on observations). Forecast queries predict the marginal moments
    over the subsequent window using only data available at ``asof``. Query
    endpoints ``start`` and ``end`` use physical time rather than observation
    indices, and observation noise is excluded from the target.

    Attributes:
        track: ``"historical"`` (completed window with ``end == asof``) or
            ``"forecast"`` (future window with ``start == asof < end``).
        asof: Cutoff timestamp up to which arrived events are visible.
        start: Start timestamp of the target increment interval.
        end: End timestamp of the target increment interval.
    """
    track: Literal["historical", "forecast"]
    asof: float
    start: float
    end: float


class Matrix(Protocol):
    """Protocol for an N x N symmetric covariance matrix representation."""
    n_assets: int
    def diagonal(self) -> np.ndarray: ...
    def entries(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray: ...
    def matvec(self, vectors: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DenseMatrix:
    """Explicit symmetric N x N covariance matrix representation.

    Attributes:
        value: 2D finite real host NumPy array of shape ``(n_assets, n_assets)``.
    """
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
    """Factor-plus-diagonal covariance representation: loadings @ loadings.T + diag(specific).

    Attributes:
        loadings: 2D finite real host NumPy array B of shape ``(n_assets, rank)`` with rank >= 0.
        specific: 1D finite nonnegative real host NumPy array d of shape ``(n_assets,)``.
    """
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
    """Estimated mean vector and covariance matrix for a query interval [start, end].

    The implied uncentered second-moment matrix is ``covariance + np.outer(mean, mean)``.
    Both ``mean`` and the arrays inside ``covariance`` must be finite real host NumPy
    arrays (floating-point or integer; copied to contiguous float64 during timed
    prediction). Dense estimates may be indefinite; factor specific variances must be
    nonnegative.

    Attributes:
        mean: 1D finite real host NumPy array of shape ``(n_assets,)``.
        covariance: ``DenseMatrix`` or ``FactorMatrix`` of dimension ``n_assets``.
    """
    mean: np.ndarray
    covariance: DenseMatrix | FactorMatrix


class Estimator(Protocol):
    """Protocol implemented by candidate estimators returned by ``create(problem, config)``."""
    def fit(self, observations: Observations) -> None:
        """Fit the estimator to a materialized point table in offline mode."""
        ...
    def update(self, events: Events) -> None:
        """Apply an arrival-ordered batch of point edits in online mode."""
        ...
    def predict(self, query: Query) -> Prediction:
        """Predict the mean and covariance of X(query.end) - X(query.start) at query.asof."""
        ...
    def synchronize(self) -> None:
        """Block until any asynchronous device operations have completed (no-op on CPU)."""
        ...
