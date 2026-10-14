"""Losses for complete increment moments, with explicit undefined scores.

All losses are minimized. Matrix losses are exact floating-point evaluations
by default; optional trace probes estimate a ratio and retain their raw values.
Gaussian scores describe the declared marginal blocks, not an unobserved joint
likelihood. Nothing here repairs a candidate's covariance.
"""
from __future__ import annotations

import operator
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .api import DenseMatrix, FactorMatrix, Prediction


METRIC_NAMES = (
    "covariance_nmse", "second_moment_nmse", "diagonal_nmse",
    "mean_variance_scaled_mse", "mean_mahalanobis", "correlation_mse",
    "portfolio_risk_relative_mse", "portfolio_second_moment_relative_mse",
    "gaussian_kl", "covariance_qlike",
)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    value = operator.index(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _array(value, name, ndim):
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real host NumPy array")
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with {ndim} dimensions")
    return np.asarray(value, dtype=np.float64)


def _maxabs(value):
    # Limit the temporary for a dense covariance to 128 rows.
    if value.ndim == 1:
        return float(np.max(np.abs(value), initial=0.0))
    return max((float(np.max(np.abs(value[i:i + 128]), initial=0.0))
                for i in range(0, len(value), 128)), default=0.0)


def _validate(value):
    if not isinstance(value, Prediction):
        raise ValueError("output must be Prediction")
    mean = _array(value.mean, "mean", 1)
    n = len(mean)
    if n == 0:
        raise ValueError("mean must be nonempty")
    matrix = value.covariance
    if isinstance(matrix, DenseMatrix):
        covariance = _array(matrix.value, "dense covariance", 2)
        if covariance.shape != (n, n):
            raise ValueError("dense covariance must have shape (n_assets, n_assets)")
        scale = _maxabs(covariance)
        if scale:
            for i in range(0, n, 128):
                difference = covariance[i:i + 128] / scale - covariance.T[i:i + 128] / scale
                if np.max(np.abs(difference), initial=0.0) > 64.0 * np.finfo(float).eps:
                    raise ValueError("dense covariance must be symmetric")
        matrix = DenseMatrix(covariance)
        diagonal = np.diag(covariance).copy()
    elif isinstance(matrix, FactorMatrix):
        loadings = _array(matrix.loadings, "factor loadings", 2)
        specific = _array(matrix.specific, "factor specific", 1)
        if loadings.shape[0] != n or specific.shape != (n,):
            raise ValueError("factor arrays must match n_assets")
        if np.any(specific < 0.0):
            raise ValueError("factor specific variances must be nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            diagonal = np.einsum("ij,ij->i", loadings, loadings) + specific
        if not np.all(np.isfinite(diagonal)):
            raise ValueError("implied factor covariance must be finite")
        matrix = FactorMatrix(loadings, specific)
        # For PSD factors the largest diagonal bounds every covariance entry.
        scale = float(diagonal.max(initial=0.0))
    else:
        raise ValueError("covariance must be DenseMatrix or FactorMatrix")
    return Prediction(mean, matrix), diagonal, scale


class _ScaledMatrix:
    """Covariance/unit², optionally plus (mean/unit)(mean/unit)' ."""
    def __init__(self, matrix, unit, mean=None):
        self.n = matrix.n_assets
        self.unit = unit
        self.dense = matrix.value if isinstance(matrix, DenseMatrix) else None
        self.mean = None if mean is None else mean / unit
        if self.dense is None:
            self.loadings = matrix.loadings / unit
            if self.mean is not None:
                self.loadings = np.column_stack((self.loadings, self.mean))
            self.specific = (matrix.specific / unit) / unit
        else:
            self.loadings = self.specific = None

    def rows(self, start, stop):
        if self.dense is None:
            result = self.loadings[start:stop] @ self.loadings.T
            local = np.arange(stop - start)
            result[local, np.arange(start, stop)] += self.specific[start:stop]
        else:
            result = (self.dense[start:stop] / self.unit) / self.unit
            if self.mean is not None:
                result += self.mean[start:stop, None] * self.mean[None, :]
        return result

    def block(self, indices):
        if self.dense is None:
            factors = self.loadings[indices]
            result = factors @ factors.T
            result[np.diag_indices(len(indices))] += self.specific[indices]
        else:
            result = (self.dense[np.ix_(indices, indices)] / self.unit) / self.unit
            if self.mean is not None:
                result += np.outer(self.mean[indices], self.mean[indices])
        return result

    def matvec(self, vectors):
        if self.dense is None:
            return self.loadings @ (self.loadings.T @ vectors) + self.specific[:, None] * vectors
        result = np.empty_like(vectors)
        for i in range(0, self.n, 128):
            result[i:i + 128] = self.rows(i, min(self.n, i + 128)) @ vectors
        return result


def _norm_sq(matrix):
    if matrix.dense is None:
        gram = matrix.loadings.T @ matrix.loadings
        factor_diagonal = np.einsum("ij,ij->i", matrix.loadings, matrix.loadings)
        return float(np.einsum("ij,ij->", gram, gram)
                     + 2.0 * np.dot(matrix.specific, factor_diagonal)
                     + np.dot(matrix.specific, matrix.specific))
    return float(sum(np.einsum("ij,ij->", block, block)
                     for i in range(0, matrix.n, 128)
                     for block in [matrix.rows(i, min(matrix.n, i + 128))]))


def _distance_sq(left, right):
    if left.dense is None and right.dense is None:
        a, b = left.loadings, right.loadings
        if np.array_equal(a, b):
            difference = left.specific - right.specific
            return float(np.dot(difference, difference)), "factor_algebra", False
        # QR keeps nearly equal low-rank matrices from subtracting two large
        # Frobenius norms. The work is O(N(Kp+Kt)^2), not O(N^2).
        difference_specific = left.specific - right.specific
        changed_diagonal = np.flatnonzero(difference_specific)
        joint = np.column_stack((a, b))
        signs = np.r_[np.ones(a.shape[1]), -np.ones(b.shape[1])]
        # Incorporate a sparse diagonal change into the signed low-rank QR.
        # This also covers identical specific variances: returning ||C||²
        # directly avoids subtracting inconsistent last-bit diagonal residuals
        # for equivalent rotated or split factor coordinates.
        sparse_diagonal = len(changed_diagonal) <= 32
        if sparse_diagonal and len(changed_diagonal):
            diagonal_factors = np.zeros((left.n, len(changed_diagonal)))
            diagonal_factors[changed_diagonal, np.arange(len(changed_diagonal))] = np.sqrt(np.abs(difference_specific[changed_diagonal]))
            joint = np.column_stack((joint, diagonal_factors))
            signs = np.r_[signs, np.sign(difference_specific[changed_diagonal])]
        if joint.shape[1]:
            _, triangular = np.linalg.qr(joint, mode="reduced")
            coefficient = (triangular * signs) @ triangular.T
            low_rank_sq = float(np.einsum("ij,ij->", coefficient, coefficient))
        else:
            low_rank_sq = 0.0
        if sparse_diagonal:
            return low_rank_sq, "factor_algebra", False
        low_rank_diagonal = np.einsum("ij,ij->i", a, a) - np.einsum("ij,ij->i", b, b)
        diagonal_sq = float(np.dot(low_rank_diagonal, low_rank_diagonal))
        offdiagonal_sq = low_rank_sq - diagonal_sq
        # The diagonal difference is calculated from the original factors,
        # independently of QR. Its absolute roundoff depends on those factors,
        # not solely on the possibly vanishing covariance residual.
        diagonal_energy = float(np.sum((np.einsum("ij,ij->i", a, a)
                                        + np.einsum("ij,ij->i", b, b)) ** 2))
        tolerance = 256.0 * np.finfo(float).eps * max(low_rank_sq, diagonal_sq, diagonal_energy)
        if offdiagonal_sq < -tolerance:
            raise ArithmeticError("factor Frobenius algebra lost numerical consistency")
        clamped = offdiagonal_sq < 0.0
        difference = low_rank_diagonal + left.specific - right.specific
        return float(max(0.0, offdiagonal_sq) + np.dot(difference, difference)), "factor_algebra", clamped
    total = 0.0
    for i in range(0, left.n, 128):
        residual = left.rows(i, min(left.n, i + 128)) - right.rows(i, min(right.n, i + 128))
        total += float(np.einsum("ij,ij->", residual, residual))
    return total, "streamed_dense", False


def _normalized(numerator, denominator, name, errors, true_zero=False):
    if denominator == 0.0:
        if true_zero and numerator == 0.0:
            return 0.0
        errors.append(f"{name}: {'zero truth scale' if true_zero else 'normalization underflow'}")
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = float(numerator / denominator)
    if not np.isfinite(result):
        errors.append(f"{name}: normalized loss exceeds floating-point range")
        return None
    return result


def _trace_ratio(left, right, probes, rng, name, errors):
    vectors = rng.integers(0, 2, size=(left.n, probes)).astype(float) * 2.0 - 1.0
    target = right.matvec(vectors)
    difference = left.matvec(vectors) - target
    numerators = np.einsum("ij,ij->j", difference, difference)
    denominators = np.einsum("ij,ij->j", target, target)
    probe_covariance = np.cov(np.vstack((numerators, denominators)), ddof=1)
    result = _normalized(float(numerators.mean()), float(denominators.mean()), name, errors)
    return result, {
        "evaluation": "estimated", "method": "rademacher_trace_ratio",
        "estimand": "squared Frobenius error / squared truth Frobenius norm",
        "note": "The ratio of unbiased trace estimates is not an unbiased ratio estimate.",
        "probe_count": probes,
        "numerator_probe_values": numerators.tolist(),
        "denominator_probe_values": denominators.tolist(),
        "raw_probe_covariance": probe_covariance.tolist(),
        "numerator_probe_variance": float(probe_covariance[0, 0]),
        "denominator_probe_variance": float(probe_covariance[1, 1]),
    }


def _pairs(n, count, rng):
    total = n * (n - 1) // 2
    if total <= count:
        return np.triu_indices(n, 1)
    pairs = set()
    while len(pairs) < count:
        first = int(rng.integers(n))
        second = int(rng.integers(n - 1))
        second += second >= first
        pairs.add((min(first, second), max(first, second)))
    result = np.asarray(sorted(pairs), dtype=np.int64)
    return result[:, 0], result[:, 1]


def _blocks(n, size, count, rng):
    indices = rng.permutation(n)[:min(n, size * count)]
    return [np.sort(indices[i:i + size]) for i in range(0, len(indices), size)]


def _diagnose(value, diagonal, scale, blocks, exact_limit):
    result = {
        "minimum_diagonal": float(diagonal.min()),
        "negative_diagonal_count": int(np.count_nonzero(diagonal < 0.0)),
        "zero_diagonal_count": int(np.count_nonzero(diagonal == 0.0)),
    }
    if isinstance(value.covariance, FactorMatrix):
        result["psd"] = {"status": "psd", "method": "factor_construction",
                         "scope": "full", "certificate": True,
                         "strict_pd_sufficient_condition": bool(np.all(value.covariance.specific > 0.0))}
    elif len(diagonal) <= exact_limit:
        unit = np.sqrt(scale) if scale else 1.0
        eigenvalues = np.linalg.eigvalsh((value.covariance.value / unit) / unit)
        tolerance = 128.0 * np.finfo(float).eps * len(diagonal) * max(1.0, float(np.max(np.abs(eigenvalues))))
        minimum = float(eigenvalues[0])
        result["psd"] = {"status": "indefinite" if minimum < -tolerance else "psd_within_tolerance",
                         "method": "full_symmetric_eigenvalues", "scope": "full", "certificate": False,
                         "minimum_eigenvalue_scaled": minimum,
                         "eigenvalue_scale": scale if scale else 1.0,
                         "negative_eigenvalue_tolerance_scaled": tolerance}
    else:
        unit = np.sqrt(scale) if scale else 1.0
        matrix = _ScaledMatrix(value.covariance, unit)
        minima = [float(np.linalg.eigvalsh(matrix.block(block))[0]) for block in blocks]
        result["psd"] = {"status": "indefinite_checked_block" if min(minima) < -128 * np.finfo(float).eps else "checked_blocks_nonnegative",
                         "method": "declared_block_eigenvalues", "scope": "tested_blocks_only",
                         "certificate": False, "minimum_block_eigenvalues_scaled": minima,
                         "eigenvalue_scale": scale if scale else 1.0}
    return result


def _correlation_mse(prediction, truth, pred_diagonal, true_diagonal, rows, cols, errors):
    if not len(rows):
        errors.append("correlation_mse: no off-diagonal pairs")
        return None
    if np.any(pred_diagonal <= 0.0) or np.any(true_diagonal <= 0.0):
        errors.append("correlation_mse: correlations require strictly positive marginal variances")
        return None
    values = []
    for value, diagonal in ((prediction, pred_diagonal), (truth, true_diagonal)):
        entry = value.covariance.entries(rows, cols)
        first, second = np.sqrt(diagonal[rows]), np.sqrt(diagonal[cols])
        values.append((entry / np.minimum(first, second)) / np.maximum(first, second))
    return float(np.mean((values[0] - values[1]) ** 2))


def _mean_scaled(prediction, truth, diagonal, errors):
    difference = prediction.mean - truth.mean
    if np.any(diagonal < 0.0):
        errors.append("mean_variance_scaled_mse: truth has a negative marginal variance")
        return None
    zero = diagonal == 0.0
    if np.any(difference[zero] != 0.0):
        errors.append("mean_variance_scaled_mse: mean error on a zero-variance truth coordinate")
        return None
    nonzero = (~zero) & (prediction.mean != truth.mean)
    scaled = np.zeros(len(diagonal))
    unit = np.maximum(np.maximum(np.abs(prediction.mean[nonzero]), np.abs(truth.mean[nonzero])),
                      np.sqrt(diagonal[nonzero]))
    scaled[nonzero] = ((prediction.mean[nonzero] / unit - truth.mean[nonzero] / unit)
                       / (np.sqrt(diagonal[nonzero]) / unit))
    scaled /= np.sqrt(len(scaled))
    return float(np.dot(scaled, scaled))


def _mean_mahalanobis(prediction, truth, blocks, exact_limit, errors):
    n = len(truth.mean)
    unit = np.sqrt(_maxabs(truth.covariance.diagonal())) or 1.0
    difference = prediction.mean / unit - truth.mean / unit
    if isinstance(truth.covariance, FactorMatrix) and np.all(truth.covariance.specific > 0.0):
        if np.array_equal(prediction.mean, truth.mean):
            return 0.0, {"scope": "full", "method": "factor_woodbury_residual"}
        root = np.sqrt(truth.covariance.specific)
        standardized = difference / (root / unit)
        factors = truth.covariance.loadings / root[:, None]
        if not factors.shape[1]:
            return float(np.dot(standardized, standardized) / n), {"scope": "full", "method": "diagonal_standardized_residual"}
        gram = np.eye(factors.shape[1]) + factors.T @ factors
        coefficient = cho_solve(cho_factor(gram, lower=True, check_finite=True), factors.T @ standardized)
        residual = standardized - factors @ coefficient
        return float((np.dot(residual, residual) + np.dot(coefficient, coefficient)) / n), {"scope": "full", "method": "factor_woodbury_residual"}
    matrix = _ScaledMatrix(truth.covariance, unit)
    subsets = [np.arange(n)] if n <= exact_limit else blocks
    loss, count = 0.0, 0
    for subset in subsets:
        factor = cho_factor(matrix.block(subset), lower=True, check_finite=True)
        local = difference[subset]
        loss += float(np.dot(local, cho_solve(factor, local)))
        count += len(subset)
    return loss / count, {"scope": "full" if n <= exact_limit else "declared_block_average", "asset_count": count}


def _relative_quadratics(predicted, true, name, errors):
    if np.any(true < 0.0):
        errors.append(f"{name}: truth has a negative portfolio quadratic form")
        return None
    zero = true == 0.0
    if np.any(predicted[zero] != 0.0):
        errors.append(f"{name}: predicted risk differs on a zero-risk truth portfolio")
        return None
    ratio = np.zeros_like(true)
    ratio[~zero] = predicted[~zero] / true[~zero] - 1.0
    return float(np.mean(ratio * ratio))


def _gaussian_blocks(prediction, truth, covariance_unit, blocks, errors):
    unit = covariance_unit
    predicted, true = _ScaledMatrix(prediction.covariance, unit), _ScaledMatrix(truth.covariance, unit)
    difference = prediction.mean / unit - truth.mean / unit
    kl, qlike, count = 0.0, 0.0, 0
    failed = []
    for number, subset in enumerate(blocks):
        p, t = predicted.block(subset), true.block(subset)
        try:
            p_factor, t_factor = cho_factor(p, lower=True, check_finite=True), cho_factor(t, lower=True, check_finite=True)
        except np.linalg.LinAlgError:
            failed.append(number)
            continue
        trace = float(np.trace(cho_solve(p_factor, t)))
        logdet_difference = float(2.0 * (np.log(np.diag(p_factor[0])).sum() - np.log(np.diag(t_factor[0])).sum()))
        covariance_loss = trace + logdet_difference - len(subset)
        tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(trace), abs(logdet_difference), len(subset))
        if covariance_loss < -tolerance:
            raise ArithmeticError("Gaussian block loss lost numerical consistency")
        covariance_loss = max(0.0, covariance_loss)
        local = difference[subset]
        mean_loss = float(np.dot(local, cho_solve(p_factor, local)))
        kl += 0.5 * (covariance_loss + mean_loss)
        qlike += covariance_loss
        count += len(subset)
    if failed:
        errors.append(f"Gaussian scores: non-SPD blocks {failed}; no failed block was dropped from the score")
        return None, None, failed
    return kl / count, qlike / count, failed


def score(prediction, truth, seed=0, *, exact_dense_limit=512, pair_samples=2048,
          portfolio_samples=64, block_size=32, block_count=8, trace_probes=0):
    """Score moments against evaluator-private truth without repairing outputs.

    ``trace_probes=0`` evaluates Frobenius losses exactly. A positive value
    (at least two) explicitly selects trace estimates above ``exact_dense_limit``
    for every representation. Correlation pairs, portfolios and Gaussian blocks
    depend only on dimension, seed and these settings, never on the candidate.
    KL is KL(truth || prediction), averaged per asset across declared blocks;
    QLIKE is the corresponding mean-free covariance divergence.
    """
    seed = _integer(seed, "seed")
    exact_dense_limit = _integer(exact_dense_limit, "exact_dense_limit", 1)
    pair_samples = _integer(pair_samples, "pair_samples", 1)
    portfolio_samples = _integer(portfolio_samples, "portfolio_samples", 1)
    block_size = _integer(block_size, "block_size", 1)
    block_count = _integer(block_count, "block_count", 1)
    trace_probes = _integer(trace_probes, "trace_probes")
    if trace_probes == 1:
        raise ValueError("trace_probes must be zero or at least two")
    result = {"status": "ok", "metrics": dict.fromkeys(METRIC_NAMES),
              "diagnostics": {}, "metric_details": {}, "errors": []}
    errors = result["errors"]
    for name, value in (("prediction", prediction), ("truth", truth)):
        try:
            validated, diagonal, scale = _validate(value)
        except (ValueError, TypeError, OverflowError) as exc:
            result["status"] = f"invalid_{name}"
            errors.append(f"{name}: {exc}")
            return result
        if name == "prediction":
            prediction, pred_diagonal, pred_scale = validated, diagonal, scale
        else:
            truth, true_diagonal, true_scale = validated, diagonal, scale
    n = len(truth.mean)
    if len(prediction.mean) != n:
        result["status"] = "invalid_prediction"
        errors.append("prediction: asset dimension differs from truth")
        return result
    streams = [np.random.default_rng(sequence) for sequence in np.random.SeedSequence(seed).spawn(4)]
    blocks = _blocks(n, block_size, block_count, streams[0])
    rows, cols = _pairs(n, pair_samples, streams[1])
    covariance_unit = np.sqrt(max(pred_scale, true_scale)) or 1.0
    moment_unit = max(covariance_unit, _maxabs(prediction.mean), _maxabs(truth.mean))
    scaled_prediction = _ScaledMatrix(prediction.covariance, covariance_unit)
    scaled_truth = _ScaledMatrix(truth.covariance, covariance_unit)
    metrics, details = result["metrics"], result["metric_details"]
    for name, value, diagonal, scale in (("prediction", prediction, pred_diagonal, pred_scale), ("truth", truth, true_diagonal, true_scale)):
        result["diagnostics"][name] = _diagnose(value, diagonal, scale, blocks, exact_dense_limit)
    for name, left, right, zero in (
        ("covariance_nmse", scaled_prediction, scaled_truth, true_scale == 0.0),
        ("second_moment_nmse", _ScaledMatrix(prediction.covariance, moment_unit, prediction.mean),
         _ScaledMatrix(truth.covariance, moment_unit, truth.mean), true_scale == 0.0 and not np.any(truth.mean)),
    ):
        try:
            if trace_probes and n > exact_dense_limit:
                metrics[name], details[name] = _trace_ratio(left, right, trace_probes, streams[3], name, errors)
            else:
                numerator, method, clamped = _distance_sq(left, right)
                denominator = _norm_sq(right)
                metrics[name] = _normalized(numerator, denominator, name, errors, zero)
                details[name] = {"evaluation": "exact", "method": method,
                                 "numerator_scaled": numerator, "denominator_scaled": denominator,
                                 "unit": left.unit, "roundoff_clamped": clamped}
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"{name}: {exc}")
    scaled_true_diagonal = (true_diagonal / covariance_unit) / covariance_unit
    diagonal_error = ((pred_diagonal / covariance_unit) / covariance_unit) - scaled_true_diagonal
    metrics["diagonal_nmse"] = _normalized(float(np.dot(diagonal_error, diagonal_error)), float(np.dot(scaled_true_diagonal, scaled_true_diagonal)),
                                          "diagonal_nmse", errors, not np.any(true_diagonal))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        metrics["mean_variance_scaled_mse"] = _mean_scaled(prediction, truth, true_diagonal, errors)
        metrics["correlation_mse"] = _correlation_mse(prediction, truth, pred_diagonal, true_diagonal, rows, cols, errors)
    details["correlation_mse"] = {"evaluation": "all_pairs" if n * (n - 1) // 2 <= pair_samples else "fixed_pair_sample",
                                  "pair_count": len(rows), "rows": rows.tolist(), "cols": cols.tolist()}
    try:
        metrics["mean_mahalanobis"], details["mean_mahalanobis"] = _mean_mahalanobis(prediction, truth, blocks, exact_dense_limit, errors)
    except (np.linalg.LinAlgError, ValueError, ArithmeticError) as exc:
        errors.append(f"mean_mahalanobis: truth covariance solve failed: {exc}")
    portfolios = streams[2].normal(size=(n, portfolio_samples))
    portfolios[:, 0] = 1.0
    portfolios /= np.linalg.norm(portfolios, axis=0)
    for name, left, right in (("portfolio_risk_relative_mse", scaled_prediction, scaled_truth),
                              ("portfolio_second_moment_relative_mse", _ScaledMatrix(prediction.covariance, moment_unit, prediction.mean),
                               _ScaledMatrix(truth.covariance, moment_unit, truth.mean))):
        p = np.einsum("ij,ij->j", portfolios, left.matvec(portfolios))
        t = np.einsum("ij,ij->j", portfolios, right.matvec(portfolios))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            metrics[name] = _relative_quadratics(p, t, name, errors)
        details[name] = {"portfolio_count": portfolio_samples, "weights": "unit-L2; first equal-weight, remaining seeded Gaussian"}
    try:
        full_indefinite = any(result["diagnostics"][name]["psd"]["status"] == "indefinite"
                              for name in ("prediction", "truth"))
        if full_indefinite:
            failed = list(range(len(blocks)))
            errors.append("Gaussian scores: exact full covariance diagnosis is indefinite")
        else:
            metrics["gaussian_kl"], metrics["covariance_qlike"], failed = _gaussian_blocks(prediction, truth, covariance_unit, blocks, errors)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        failed = list(range(len(blocks)))
        errors.append(f"Gaussian scores: {exc}")
    details["gaussian_scores"] = {"scope": "declared_marginal_blocks", "normalization": "per asset over all declared blocks",
                                   "direction": "KL(truth || prediction)", "blocks": [block.tolist() for block in blocks],
                                   "failed_blocks": failed, "joint_gaussian_certificate": False}
    # JSON must never carry NaN/Infinity, including probe diagnostics. A lost
    # numerical score remains an explicit failure, never a filtered trial.
    def finite_json(value, path):
        if isinstance(value, dict):
            return {key: finite_json(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                errors.append(f"{path}: nonfinite numerical result")
                return None
            return float(value)
        return value
    result["metrics"] = finite_json(metrics, "metrics")
    result["metric_details"] = finite_json(details, "metric_details")
    result["diagnostics"] = finite_json(result["diagnostics"], "diagnostics")
    if errors:
        result["status"] = "partial"
    return result
