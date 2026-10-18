# Implementing a Candidate Estimator

This guide specifies how to implement, package, test, and evaluate a candidate estimator for `covariance-bench`. Read [`src/covbench/api.py`](../src/covbench/api.py), [`docs/benchmark-design.md`](../docs/benchmark-design.md), and [`docs/metrics.md`](../docs/metrics.md) before implementing.

## 1. Deliverable and Estimation Target

Create a candidate directory under `candidates/` (for example, `candidates/my-estimator/`) containing executable source code and a `candidate.json` manifest:

```json
{
  "name": "my-estimator",
  "entrypoint": "candidate:create",
  "config": {
    "lookback": 0.25,
    "floor": 1e-8
  },
  "api_version": 1,
  "modes": ["offline", "online"]
}
```

Keep tunable hyperparameters in `config` so the runner can record and reproduce the frozen artifact, and document any required third-party dependencies and statistical assumptions.

For an $`N`$-asset universe with latent level process $`X(t) \in \mathbb{R}^N`$, a query over the physical time interval $`[t_{\mathrm{start}}, t_{\mathrm{end}}]`$ at cutoff timestamp $`t_{\mathrm{asof}}`$ requests the mean vector $`\mu \in \mathbb{R}^N`$ and covariance matrix $`\Sigma \in \mathbb{R}^{N \times N}`$ of the latent increment $`\Delta X = X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$:

```math
\mu = \mathbb{E}[X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})], \qquad \Sigma = \mathrm{Cov}(X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})).
```

The implied uncentered second-moment matrix is

```math
M = \Sigma + \mu \mu^\top.
```

- **Historical queries** (`track = "historical"`, $`t_{\mathrm{start}} < t_{\mathrm{end}} = t_{\mathrm{asof}}`$) estimate the marginal moments of the latent increment over the observed physical-time window (rather than the posterior moments of the realized path conditional on observations).
- **Forecast queries** (`track = "forecast"`, $`t_{\mathrm{asof}} = t_{\mathrm{start}} < t_{\mathrm{end}}`$) predict the marginal moments of the latent increment over the subsequent interval using only information that has arrived by $`t_{\mathrm{asof}}`$.

In both tracks, the target excludes observation noise $`\varepsilon_{i,k}`$ and includes the increment mean $`\mu`$ in the second moment $`M = \Sigma + \mu\mu^\top`$.

## 2. Required API Behavior

The entrypoint factory `create(problem: Problem, config: dict) -> Estimator` receives `Problem(n_assets, mode, time_unit)` and `config`, and returns an object implementing four methods:

- `fit(observations: Observations) -> None`: fits from the materialized point table at the current checkpoint in `"offline"` mode. The 1D read-only arrays `observations.time`, `observations.asset`, and `observations.value` contain physical measurement timestamps $`t_k \le t_{\mathrm{asof}}`$, asset indices $`i_k \in \{0, \dots, N-1\}`$, and observed level proxies $`y_k`$. Calling `fit` replaces any prior observation state.
- `update(events: Events) -> None`: processes arrival-ordered point edits in `"online"` mode. `UPSERT` (`0`) inserts or replaces a point keyed by `(asset, time)`; `DELETE` (`1`) removes that key and ignores `value`. Handle retroactive corrections and deletions of older measurement timestamps, not only append operations.
- `predict(query: Query) -> Prediction`: returns `Prediction(mean, covariance)` for the full latent increment $`X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$, interpreting `query.start`, `query.end`, and `query.asof` in physical time.
- `synchronize() -> None`: completes any outstanding asynchronous accelerator work (or is a no-op for CPU code).

### Output Representation and Validation

`Prediction(mean, covariance)` must contain finite real host NumPy arrays (floating-point or integer; `dtype.kind in "fiu"`):

- `mean`: 1D array of shape `(n_assets,)`.
- `covariance`: an instance of either:
  - `DenseMatrix(value)` with symmetric 2D array `value` of shape `(n_assets, n_assets)`, or
  - `FactorMatrix(loadings, specific)` representing $`B B^\top + \mathrm{diag}(d)`$, with 2D array `loadings` $`B`$ of shape `(n_assets, rank)` ($`K \ge 0`$), nonnegative 1D array `specific` $`d`$ of shape `(n_assets,)`, and finite implied diagonal $`\mathrm{diag}(B B^\top) + d`$.

Return the `DenseMatrix` or `FactorMatrix` API classes directly with materialized host arrays rather than lazy objects that defer computation. During the timed `predict` call, the worker synchronizes devices and copies the output arrays into contiguous `float64` host memory.

### Minimal Skeleton

```python
from covbench.api import DELETE, UPSERT, FactorMatrix, Prediction
import numpy as np


class MyEstimator:
    def __init__(self, problem, config):
        self.n = problem.n_assets
        self.floor = float(config.get("floor", 1e-8))
        self.points = [{} for _ in range(self.n)]

    def fit(self, observations):
        self.points = [{} for _ in range(self.n)]
        for t, a, v in zip(observations.time, observations.asset, observations.value, strict=True):
            self.points[int(a)][float(t)] = float(v)

    def update(self, events):
        for t, a, v, op in zip(events.time, events.asset, events.value, events.operation, strict=True):
            if op == UPSERT:
                self.points[int(a)][float(t)] = float(v)
            elif op == DELETE:
                self.points[int(a)].pop(float(t), None)

    def predict(self, query):
        duration = float(query.end - query.start)
        mean = np.zeros(self.n, dtype=np.float64)
        loadings = np.zeros((self.n, 0), dtype=np.float64)
        specific = np.full(self.n, self.floor * duration, dtype=np.float64)
        return Prediction(mean=mean, covariance=FactorMatrix(loadings, specific))

    def synchronize(self):
        pass


def create(problem, config):
    return MyEstimator(problem, config)
```

## 3. Operational Restrictions and Input Contract

The only permitted instance information is the public API input: `Problem`, the observed point table or arrival-ordered edits, and `Query`.

- Infer observation noise, drift, factor rank, and other model parameters from the permitted observations, or use fixed assumptions declared in `candidate.json`.
- Do **not** inspect evaluator ground truth, generator state, root seeds, held-out suite configurations, or future observations arriving after `query.asof`.
- Do **not** modify the benchmark harness, scoring functions, or scenario definitions.
- Do **not** cache instance answers across independent trials on disk or in external services.

Because the local subprocess runner does not enforce a filesystem or network sandbox, adherence to these restrictions is a required contract of candidate implementations.

## 4. Statistical and Algorithmic Considerations

1. **Interval vs. rate scaling**: if an estimator computes per-unit-time drift rate $`\hat{\mu}_{\mathrm{rate}}`$ and covariance rate $`\hat{\Sigma}_{\mathrm{rate}}`$, scale them by the query duration $`\Delta\tau = t_{\mathrm{end}} - t_{\mathrm{start}}`$ to obtain interval moments $`\hat{\mu} = \hat{\mu}_{\mathrm{rate}} \Delta\tau`$ and $`\hat{\Sigma} = \hat{\Sigma}_{\mathrm{rate}} \Delta\tau`$ (for a `FactorMatrix`, scale `loadings` by $`\sqrt{\Delta\tau}`$ and `specific` by $`\Delta\tau`$).
2. **Metric domains and regularization**: indefinite `DenseMatrix` outputs are retained and diagnosed by the evaluator without projection; algebraic losses remain defined (and `correlation_mse` remains defined whenever marginal variances are strictly positive), while `gaussian_kl` and `covariance_qlike` require that neither matrix is diagnosed as full-matrix `"indefinite"` (for dense matrices with $`N \le N_{\mathrm{exact}}`$, within eigenvalue tolerance) and that Cholesky factorization succeeds on every declared principal block. If a method requires eigenvalue flooring or shrinkage, implement and account for it inside the candidate.
3. **Output representation and scaling**: both `DenseMatrix` and `FactorMatrix` are valid across all suites (`smoke`, `core`, `scale`, and `holdout`). Returning a `FactorMatrix` of rank $`K_p`$ stores the output in $`O(N(K_p + 1))`$ floats and allows the evaluator's QR/Woodbury scorer to evaluate exact Frobenius losses in $`O(N(K_p + mK + 1)^2)`$ time, whereas a `DenseMatrix` candidate is scored by streaming 128-row blocks in $`O(N)`$ workspace and $`O(N^2(mK + 1))`$ time against rank-$`mK`$ factor truth. The candidate's own runtime is determined by its internal estimation algorithm and must fit within the runner's timeout and RSS limits.
4. **Checkpoint work accounting**: `completed_ns` sums all work at a checkpoint (`create` + `fit` + two `predict` calls in offline mode; incremental `update` batches + two `predict` calls in online mode, plus `create` at the first checkpoint). Every timing repetition starts a fresh worker process with no excluded warmup phase, so any JIT compilation triggered during a stage is charged to that stage.

## 5. Evaluation and Validation Workflow

1. **Smoke verification**: verify offline and online execution, point replacement and deletion, forecast queries, and output validity on `suites/smoke.json`:
   ```sh
   uv run covbench run --candidate candidates/my-estimator --suite suites/smoke.json \
     --root-seed 20261013 --replications 2 --timing-repeats 1 --output runs/smoke-check
   ```
2. **Matched development comparison**: because `comparison_key` hashes the evaluator tree, hardware/software environment, execution policy, suite, and `root_seed`, run the candidate and the reference baselines on the same machine with matching flags before computing Pareto frontiers:
   ```sh
   uv run covbench run --candidate candidates/my-estimator --suite suites/core.json \
     --root-seed 20261013 --replications 3 --timing-repeats 2 --output runs/my-core
   uv run covbench run --candidate candidates/previous-tick --suite suites/core.json \
     --root-seed 20261013 --replications 3 --timing-repeats 2 --output runs/tick-core
   uv run covbench run --candidate candidates/zero --suite suites/core.json \
     --root-seed 20261013 --replications 3 --timing-repeats 2 --output runs/zero-core

   uv run covbench frontier runs/my-core/results.json runs/tick-core/results.json runs/zero-core/results.json \
     --objectives completed_ns covariance_nmse --output runs/frontier-core.json
   ```
   Inspect stage timings (`load`, `create`, `fit`, `update`, `predict`, `completed_ns`), statistical losses, Monte Carlo standard errors, PSD diagnostics, and any failures or `null` metrics across regimes. Report both the regimes that improve and those that regress.
3. **Held-out validation**: once code and `candidate.json` are selected, freeze them and evaluate the candidate and baselines on `suites/holdout.json` using a fresh `--root-seed`. Fresh seeds assess new draws from a fixed regime specification, whereas `suites/holdout.json` assesses generalization to new asset dimensions, three-regime transitions, and strong drift. If held-out results are used to guide further changes, treat them as development feedback and use a new held-out configuration for subsequent validation.
