# Metrics

The target is the latent increment's mean `μ` and covariance `Σ`, excluding
observation noise. Its second moment is `M = Σ + μ μᵀ`. Every reported loss is
minimized; the evaluator does not combine them into a scalar fitness.

| Metric | Definition and domain |
| --- | --- |
| `covariance_nmse` | `||Σ̂ − Σ||²_F / ||Σ||²_F`. |
| `second_moment_nmse` | `||M̂ − M||²_F / ||M||²_F`; includes mean error. |
| `diagonal_nmse` | Squared error of marginal variances divided by their squared truth norm. |
| `mean_variance_scaled_mse` | Mean of `(μ̂ᵢ − μᵢ)² / Σᵢᵢ`; does not divide by the drift magnitude. |
| `mean_mahalanobis` | `(μ̂ − μ)ᵀ Σ⁻¹ (μ̂ − μ) / N` where the declared truth covariance solve exists. |
| `correlation_mse` | Mean squared correlation error on declared off-diagonal pairs; requires positive marginal variances. |
| `portfolio_risk_relative_mse` | Mean of `(wᵀΣ̂w / wᵀΣw − 1)²` on declared portfolios. |
| `portfolio_second_moment_relative_mse` | The same quadratic-form loss using second moments. |
| `gaussian_kl` | `KL(N(μ,Σ) || N(μ̂,Σ̂))`, per asset across declared marginal blocks. |
| `covariance_qlike` | `tr(Σ̂⁻¹Σ) + logdet(Σ̂) − logdet(Σ) − block_size`, per asset across those blocks. |

For Student innovations, jumps, or other non-Gaussian generators, Gaussian KL
compares moment-matched Gaussian surrogates. It is not the KL divergence between
the latent distributions. QLIKE likewise assesses covariance through a Gaussian
quasi-likelihood, rather than testing the full distribution.

Frobenius losses are exact floating-point evaluations by default. Factor/factor
comparisons use signed low-rank QR algebra and diagonal terms. Equivalent
rotated or split factor coordinates therefore remain comparable. Dense matrices
are processed in row chunks; large factor covariance matrices are not formed.
Common unit scaling reduces overflow and makes normalized losses invariant to
a common change of level units. Reported numerical roundoff clamps affect the
loss calculation, never the candidate's matrix.

Setting `trace_probes` to at least two explicitly selects Rademacher trace
estimates above `exact_dense_limit`, for every representation. The estimand is
the same normalized squared Frobenius error. Raw numerator and denominator
probe values and their sample covariance are retained. Each trace estimate is
unbiased, but their ratio need not be. This probe variance describes evaluator
approximation, not an estimator confidence interval.

Correlation pairs, unit-L2 portfolios, and Gaussian blocks depend only on the
asset count, evaluator seed, and scoring settings. They are identical across
candidates. The first portfolio is equal-weight; the rest are seeded Gaussian
vectors. Block scores are size-weighted averages over disjoint declared subsets;
they do not include dependence between blocks. Their indices and scope are
recorded. Dense truth Mahalanobis scores use the full matrix at small dimension
and declared blocks at large dimension; positive-specific factor truth supports
a full Woodbury solve. That scope is recorded separately.

Malformed, incomplete, nonfinite, asymmetric, and negative-specific factor
outputs fail validation. Finite symmetric dense outputs may be indefinite:
their algebraic losses remain visible. Correlations or SPD-only scores that
cannot be evaluated become null with a reason. If any declared Gaussian block
fails, the Gaussian score is null; surviving blocks are not silently averaged.
Zero truth scale gives zero only when both quantities agree at zero; otherwise
the normalized loss is undefined. Failed trials remain in the results.

PSD diagnostics use full eigenvalues for small dense matrices and declared
blocks for large dense matrices. Passing large-block checks is not a global PSD
certificate. Nonnegative-specific factor covariance is structurally PSD, though
it can be singular. Neither block checks nor Gaussian scores certify calibrated
uncertainty outside the checked domains.

Pareto fronts compare finite selected objectives only within explicitly matching
suite, evaluator, hardware, configuration, track, scenario, and mode metadata.
Exact ties remain on the frontier. Missing metadata, timeouts, invalid outputs,
and unavailable selected objectives remain explicit exclusions. Monte Carlo
variation across benchmark draws describes uncertainty in measured benchmark
loss; it is not a confidence interval supplied by the estimator.
