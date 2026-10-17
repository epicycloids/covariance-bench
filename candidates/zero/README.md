# Zero Baseline (`zero`)

`zero` is a constant sanity-check baseline implemented in [`src/covbench/baselines.py`](../../src/covbench/baselines.py) (`ZeroEstimator`).

For any asset count $`N`$ and query interval $`[t_{\mathrm{start}}, t_{\mathrm{end}}]`$, `ZeroEstimator` ignores all observations and point edits (`fit` and `update` are no-ops) and returns zero mean, zero covariance, and zero second moment using an empty factor matrix:

```math
\hat{\mu} = 0 \in \mathbb{R}^N, \qquad \hat{\Sigma} = 0 \in \mathbb{R}^{N \times N} \quad (\texttt{FactorMatrix(np.zeros((N, 0)), np.zeros(N))}).
```

Both `offline` and `online` modes use the same implementation, and `synchronize()` is a no-op.

On any query with positive definite ground truth $`\Sigma \succ 0`$, `zero` achieves unit relative error ($`1.0`$) on `covariance_nmse`, `second_moment_nmse`, `diagonal_nmse`, `portfolio_risk_relative_mse`, and `portfolio_second_moment_relative_mse`, while `correlation_mse`, `gaussian_kl`, and `covariance_qlike` are `null` (`status = "partial"`) because the predicted marginal variances are zero. It serves as a timing overhead floor and a unit-loss reference point.
