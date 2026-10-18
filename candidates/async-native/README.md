# Asynchronous Overlap Refit Baseline (`async-native`)

`async-native` (`async-native-refit` in `candidate.json`) is a window-refit adapter for the optional [`async-covariance`](https://github.com/epicycloids/async-covariance) package, implemented in [`src/covbench/baselines.py`](../../src/covbench/baselines.py) (`AsyncNativeEstimator`). Because worker subprocesses import from the project environment `.venv` and `uv sync` removes undeclared packages, install `async-covariance` into `.venv` after syncing. From the `covariance-bench` repository root (assuming a sibling `../async-covariance` checkout):

```sh
uv sync --extra test
uv pip install --python .venv/bin/python ../async-covariance
uv run covbench run --candidate candidates/async-native --suite suites/smoke.json \
  --root-seed 20261013 --replications 3 --timing-repeats 3 --output runs/native
```

## Method

The estimator maintains the active observation table under `UPSERT` replacement and `DELETE` removal. Each `predict(query)` call extracts observations in the historical window $`[t_{\mathrm{start}}, \min(t_{\mathrm{end}}, t_{\mathrm{asof}})]`$ or the forecast lookback window $`[t_{\mathrm{asof}} - \text{lookback}, t_{\mathrm{asof}}]`$ (default `lookback = 0.25`) and performs a batch `async_covariance.fit(..., method="overlap", noise_variance=0.0, threads=threads)`.

- **Drift rate**: estimated per asset from the first and last observations in the window as $`(Y_{i,\mathrm{last}} - Y_{i,\mathrm{first}}) / (t_{i,\mathrm{last}} - t_{i,\mathrm{first}})`$, or zero if the asset has fewer than two window observations.
- **Small and medium universes ($`N \le 256`$)**: evaluates all asset pairs, replaces missing pair estimates (`NaN`) with `0.0`, floors diagonal variance rates at `floor` (default `1e-8`), and returns `DenseMatrix(covariance * duration)` without eigenvalue projection.
- **Large universes ($`N > 256`$)**: evaluates pairs on a deterministic circulant ring graph of even `degree` (default `8`), floors marginal variance rates at `floor`, fits a factor model via `MomentSnapshot.factor` of rank at most `rank` (default `8`) with `specific_floor` (`0.05`), `max_iter` (`100`), and `seed` (`0`), and scales loadings by $`\sqrt{\Delta\tau}`$ and specific variances by $`\Delta\tau = t_{\mathrm{end}} - t_{\mathrm{start}}`$ to return a `FactorMatrix`.

## Execution Profile and Limitations

- **Refit timing**: online `update` calls only maintain the point table; the native overlap scan and any factor fitting occur inside `predict` and are included in `predict_ns` and `completed_ns`. This baseline does not benchmark `async_covariance`'s incremental `OnlineCovariance` engine.
- **Modeling policies**: observation noise variance is set to zero, and the ring graph, rank cap, and variance floor for $`N > 256`$ are fixed baseline policies. Any native fitting errors propagate directly as candidate failures. CPU `synchronize()` is a no-op.
