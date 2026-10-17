# Previous-Tick Baseline (`previous-tick`)

`previous-tick` is a grid-interpolation sample covariance baseline implemented in [`src/covbench/baselines.py`](../../src/covbench/baselines.py) (`PreviousTickEstimator`).

## Method

The estimator maintains a per-asset observation table supporting `UPSERT` replacement and `DELETE` removal. Each `predict(query)` call scans the table over the historical query window $`[t_{\mathrm{start}}, \min(t_{\mathrm{end}}, t_{\mathrm{asof}})]`$ or, for forecast queries, over the lookback interval $`[t_{\mathrm{asof}} - \text{lookback}, t_{\mathrm{asof}}]`$. If an asset has an observation prior to the window start, its most recent pre-window point is included to anchor the initial grid value; no observations after the window end are used.

The window is partitioned into `bins` ($`B \ge 3`$, default `32`) equally spaced grid points with step $`\Delta g`$, producing $`B - 1`$ grid intervals. Each grid point carries forward the latest observation at or before that timestamp. For each asset $`i`$ with at least two selected observations:

- Grid intervals before asset $`i`$'s first observation are `NaN` and omitted; let $`n_i`$ be the number of valid grid returns $`r_{i,b}`$.
- The sample mean return $`\bar{r}_i`$ defines the estimated drift rate $`\hat{\mu}_{\mathrm{rate}, i} = \bar{r}_i / \Delta g`$.
- When $`n_i > 1`$, centered valid returns are scaled by $`\sqrt{\Delta\tau / ((n_i - 1)\Delta g)}`$ (where $`\Delta\tau = t_{\mathrm{end}} - t_{\mathrm{start}}`$ is the requested query duration) and placed into row $`i`$ of a $`N \times (B - 1)`$ loading matrix $`L`$, with omitted pre-observation columns set to zero.

With diagonal floor rate `floor` ($`\phi \ge 0`$, default `1e-8`), the estimator returns

$$
\hat{\mu} = \Delta\tau \, \hat{\mu}_{\mathrm{rate}}, \qquad \hat{\Sigma} = L L^\top + \mathrm{diag}(\phi \Delta\tau \, \mathbf{1}_N)
$$

as a `FactorMatrix` without allocating a dense $`N \times N`$ matrix.

## Assumptions and Limitations

- Assumes stationary drift and covariance rates over the estimation window and forecast horizon, and does not adjust for microstructure noise.
- Carrying observations forward across asynchronous arrivals introduces zero grid returns on un-updated intervals (the Epps effect), and zero-filling pre-observation columns when $`n_i < B - 1`$ can attenuate cross-covariances under partial overlap.
- Online `update` calls only mutate the point dictionaries; window scanning and grid construction are charged to `predict` (and included in `completed_ns`). CPU `synchronize()` is a no-op.
