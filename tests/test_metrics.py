import json

import numpy as np
import pytest

from covbench.api import DenseMatrix, FactorMatrix, Prediction
from covbench.metrics import score


def _dense(matrix):
    if isinstance(matrix, DenseMatrix):
        return matrix.value
    return matrix.loadings @ matrix.loadings.T + np.diag(matrix.specific)


def _fixture(n=9, rank=3):
    rng = np.random.default_rng(472)
    truth = Prediction(rng.normal(size=n) * .2,
                       FactorMatrix(rng.normal(size=(n, rank)) * .3, rng.uniform(.5, 2., n)))
    prediction = Prediction(truth.mean + rng.normal(size=n) * .1,
                            FactorMatrix(truth.covariance.loadings + rng.normal(size=(n, rank)) * .05,
                                         truth.covariance.specific * rng.uniform(.8, 1.2, n)))
    return prediction, truth


@pytest.mark.parametrize("prediction_factor", [False, True])
@pytest.mark.parametrize("truth_factor", [False, True])
def test_exact_dense_factor_parity_and_independent_formulas(prediction_factor, truth_factor):
    prediction, truth = _fixture()
    p, t = _dense(prediction.covariance), _dense(truth.covariance)
    if not prediction_factor:
        prediction = Prediction(prediction.mean, DenseMatrix(p))
    if not truth_factor:
        truth = Prediction(truth.mean, DenseMatrix(t))
    result = score(prediction, truth, seed=8, block_size=len(t), block_count=1)
    assert result["status"] == "ok"
    metrics = result["metrics"]
    second_p = p + np.outer(prediction.mean, prediction.mean)
    second_t = t + np.outer(truth.mean, truth.mean)
    np.testing.assert_allclose(metrics["covariance_nmse"], np.sum((p-t)**2) / np.sum(t*t), rtol=1e-12)
    np.testing.assert_allclose(metrics["second_moment_nmse"], np.sum((second_p-second_t)**2) / np.sum(second_t**2), rtol=1e-12)
    np.testing.assert_allclose(metrics["diagonal_nmse"], np.sum((np.diag(p)-np.diag(t))**2) / np.sum(np.diag(t)**2), rtol=1e-12)
    difference = prediction.mean - truth.mean
    np.testing.assert_allclose(metrics["mean_variance_scaled_mse"], np.mean(difference**2 / np.diag(t)), rtol=1e-12)
    np.testing.assert_allclose(metrics["mean_mahalanobis"], difference @ np.linalg.solve(t, difference) / len(t), rtol=1e-12)
    rows, cols = np.triu_indices(len(t), 1)
    cp = p[rows, cols] / np.sqrt(np.diag(p)[rows] * np.diag(p)[cols])
    ct = t[rows, cols] / np.sqrt(np.diag(t)[rows] * np.diag(t)[cols])
    np.testing.assert_allclose(metrics["correlation_mse"], np.mean((cp-ct)**2), rtol=1e-12)
    covariance_kl = np.trace(np.linalg.solve(p, t)) + np.linalg.slogdet(p)[1] - np.linalg.slogdet(t)[1] - len(t)
    np.testing.assert_allclose(metrics["covariance_qlike"], covariance_kl / len(t), rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(metrics["gaussian_kl"], .5*(covariance_kl + difference @ np.linalg.solve(p, difference)) / len(t), rtol=1e-11)
    # Independently reconstruct the seeded unit-L2 evaluation portfolios.
    rng = np.random.default_rng(np.random.SeedSequence(8).spawn(4)[2])
    weights = rng.normal(size=(len(t), 64)); weights[:, 0] = 1.
    weights /= np.linalg.norm(weights, axis=0)
    qp = np.einsum("ij,ij->j", weights, p @ weights)
    qt = np.einsum("ij,ij->j", weights, t @ weights)
    np.testing.assert_allclose(metrics["portfolio_risk_relative_mse"], np.mean((qp/qt-1)**2), rtol=1e-12)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scale", [1e-120, 1e120])
@pytest.mark.parametrize("factor", [False, True])
def test_unit_scale_invariance(scale, factor):
    prediction, truth = _fixture()
    def rescale(value, scale):
        if factor:
            matrix = FactorMatrix(value.covariance.loadings * np.sqrt(scale), value.covariance.specific * scale)
        else:
            matrix = DenseMatrix(_dense(value.covariance) * scale)
        return Prediction(value.mean * np.sqrt(scale), matrix)
    base = score(rescale(prediction, 1.), rescale(truth, 1.), seed=21)
    transformed = score(rescale(prediction, scale), rescale(truth, scale), seed=21)
    assert transformed["status"] == "ok"
    for name, value in base["metrics"].items():
        np.testing.assert_allclose(transformed["metrics"][name], value, rtol=2e-11, atol=1e-14)


def test_second_moment_and_zero_drift_mean_errors():
    matrix = FactorMatrix(np.empty((3, 0)), np.ones(3))
    truth = Prediction(np.zeros(3), matrix)
    prediction = Prediction(np.array([2., 0., 0.]), matrix)
    metrics = score(prediction, truth)["metrics"]
    assert metrics["covariance_nmse"] == 0.
    assert metrics["second_moment_nmse"] == pytest.approx(16./3)
    assert metrics["mean_variance_scaled_mse"] == pytest.approx(4./3)
    assert metrics["mean_mahalanobis"] == pytest.approx(4./3)
    assert metrics["gaussian_kl"] == pytest.approx(2./3)
    assert metrics["covariance_qlike"] == 0.


def test_indefinite_dense_keeps_losses_without_repair():
    truth = Prediction(np.zeros(3), DenseMatrix(np.eye(3)))
    value = np.array([[-.5, .2, 0.], [.2, 1., 0.], [0., 0., 2.]])
    prediction = Prediction(np.zeros(3), DenseMatrix(value))
    result = score(prediction, truth)
    assert result["status"] == "partial"
    assert result["metrics"]["covariance_nmse"] == pytest.approx(np.sum((value-np.eye(3))**2)/3)
    assert result["metrics"]["correlation_mse"] is None
    assert result["metrics"]["gaussian_kl"] is None
    assert result["metrics"]["covariance_qlike"] is None
    assert result["diagnostics"]["prediction"]["psd"]["status"] == "indefinite"
    assert result["diagnostics"]["prediction"]["negative_diagonal_count"] == 1
    np.testing.assert_array_equal(prediction.covariance.value, value)
    json.dumps(result, allow_nan=False)


def test_failed_block_is_not_dropped_from_gaussian_average():
    value = np.eye(8); value[0, 0] = -1.
    result = score(Prediction(np.zeros(8), DenseMatrix(value)),
                   Prediction(np.zeros(8), DenseMatrix(np.eye(8))),
                   exact_dense_limit=1, block_size=2, block_count=4)
    assert result["metrics"]["gaussian_kl"] is None
    assert result["metrics"]["covariance_qlike"] is None
    assert result["metric_details"]["gaussian_scores"]["failed_blocks"]
    assert result["diagnostics"]["prediction"]["psd"]["certificate"] is False
    assert result["diagnostics"]["prediction"]["psd"]["scope"] == "tested_blocks_only"


def test_zero_truth_scale_and_degenerate_mean_are_explicit():
    zero = FactorMatrix(np.empty((3, 0)), np.zeros(3))
    truth = Prediction(np.zeros(3), zero)
    perfect = score(truth, truth)
    assert perfect["metrics"]["covariance_nmse"] == 0.
    assert perfect["metrics"]["second_moment_nmse"] == 0.
    assert perfect["metrics"]["mean_variance_scaled_mse"] == 0.
    assert perfect["metrics"]["gaussian_kl"] is None
    wrong = score(Prediction(np.ones(3), zero), truth)
    assert wrong["metrics"]["mean_variance_scaled_mse"] is None
    assert wrong["metrics"]["second_moment_nmse"] is None
    assert wrong["errors"]
    json.dumps(wrong, allow_nan=False)


@pytest.mark.parametrize("bad", [
    None,
    Prediction(np.zeros(2), DenseMatrix(np.eye(3))),
    Prediction(np.zeros(3), DenseMatrix(np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]]))),
    Prediction(np.zeros(3), DenseMatrix(np.full((3, 3), np.nan))),
    Prediction(np.array([0., np.inf, 0.]), DenseMatrix(np.eye(3))),
    Prediction(np.zeros(3), FactorMatrix(np.ones((3, 1)), np.array([1., -.1, 1.]))),
    Prediction(np.zeros(3), FactorMatrix(np.ones((2, 1)), np.ones(3))),
    Prediction(np.zeros(3), FactorMatrix(np.full((3, 1), 1e200), np.ones(3))),
    Prediction(np.zeros(3, dtype=complex), DenseMatrix(np.eye(3))),
    Prediction([0., 0., 0.], DenseMatrix(np.eye(3))),
])
def test_invalid_outputs_preserve_failure(bad):
    truth = Prediction(np.zeros(3), DenseMatrix(np.eye(3)))
    result = score(bad, truth)
    assert result["status"] == "invalid_prediction"
    assert all(value is None for value in result["metrics"].values())
    assert result["errors"]
    json.dumps(result, allow_nan=False)


def test_wrong_truth_and_dimension_are_distinct_failures():
    valid = Prediction(np.zeros(3), DenseMatrix(np.eye(3)))
    assert score(valid, None)["status"] == "invalid_truth"
    other = Prediction(np.zeros(4), DenseMatrix(np.eye(4)))
    assert score(other, valid)["status"] == "invalid_prediction"


def test_large_factor_scores_do_not_materialize_dense_rows(monkeypatch):
    from covbench import metrics
    prediction, truth = _fixture(n=3000, rank=3)
    def forbidden(*args):
        raise AssertionError("factor scores must not materialize covariance rows")
    monkeypatch.setattr(metrics._ScaledMatrix, "rows", forbidden)
    result = score(prediction, truth, seed=99, pair_samples=128, portfolio_samples=8)
    assert result["status"] == "ok"
    assert result["metric_details"]["covariance_nmse"]["method"] == "factor_algebra"
    assert result["diagnostics"]["prediction"]["psd"]["certificate"] is True


@pytest.mark.parametrize("transform", ["rotate", "split"])
def test_equivalent_factor_coordinates_have_finite_near_zero_loss(transform):
    # Verify that orthogonal factor rotations and column splits with identical
    # specific variances evaluate to near-zero Frobenius loss without cancellation.
    rng = np.random.default_rng(23)
    factors = rng.normal(size=(9, 3))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    other = factors @ rotation if transform == "rotate" else np.column_stack((factors, factors)) / np.sqrt(2.)
    truth = Prediction(np.zeros(9), FactorMatrix(factors, np.ones(9)))
    prediction = Prediction(np.zeros(9), FactorMatrix(other, np.ones(9)))
    result = score(prediction, truth)
    assert result["status"] == "ok"
    assert result["metrics"]["covariance_nmse"] < 1e-28
    assert result["metrics"]["second_moment_nmse"] < 1e-28


def test_sparse_specific_change_does_not_hide_tiny_offdiagonal_error():
    n = 100
    true_specific = np.ones(n)
    prediction_specific = np.ones(n); prediction_specific[0] = 0.
    factors = np.zeros((n, 1)); factors[0, 0] = 1.; factors[1, 0] = 1e-9
    prediction = Prediction(np.zeros(n), FactorMatrix(factors, prediction_specific))
    truth = Prediction(np.zeros(n), FactorMatrix(np.empty((n, 0)), true_specific))
    result = score(prediction, truth)
    assert result["metrics"]["covariance_nmse"] == pytest.approx(2e-18/n, rel=1e-6)


def test_trace_estimate_records_raw_probe_uncertainty_and_shared_queries():
    prediction, truth = _fixture(n=80, rank=3)
    first = score(prediction, truth, seed=100, exact_dense_limit=16, trace_probes=128, pair_samples=32)
    repeat = score(prediction, truth, seed=100, exact_dense_limit=16, trace_probes=128, pair_samples=32)
    exact = score(prediction, truth, seed=100, exact_dense_limit=16, pair_samples=32)
    assert first == repeat
    detail = first["metric_details"]["covariance_nmse"]
    assert detail["evaluation"] == "estimated"
    assert len(detail["numerator_probe_values"]) == 128
    assert detail["numerator_probe_variance"] > 0.
    assert len(detail["raw_probe_covariance"]) == 2
    assert first["metrics"]["covariance_nmse"] == pytest.approx(exact["metrics"]["covariance_nmse"], rel=.3)
    alternate = score(truth, truth, seed=100, exact_dense_limit=16, pair_samples=32)
    assert first["metric_details"]["correlation_mse"] == alternate["metric_details"]["correlation_mse"]
    assert first["metric_details"]["gaussian_scores"]["blocks"] == alternate["metric_details"]["gaussian_scores"]["blocks"]
    json.dumps(first, allow_nan=False)


def test_options_reject_invalid_counts():
    prediction, truth = _fixture()
    for options in ({"seed": True}, {"trace_probes": 1}, {"block_count": 0}, {"portfolio_samples": -1}):
        with pytest.raises((ValueError, TypeError)):
            score(prediction, truth, **options)
