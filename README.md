# covariance-bench

A benchmark harness for evaluating code that estimates latent price-increment moments from sparse, asynchronous, and noisy observations.

For an $`N`$-asset universe with latent level process $`X(t) \in \mathbb{R}^N`$, a query over the physical time interval $`[t_{\mathrm{start}}, t_{\mathrm{end}}]`$ requests the mean vector $`\mu`$, covariance matrix $`\Sigma`$, and implied second-moment matrix $`M`$ of the increment $`\Delta X = X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$:

```math
\mu = \mathbb{E}[\Delta X], \qquad \Sigma = \mathrm{Cov}(\Delta X), \qquad M = \mathbb{E}[\Delta X \, \Delta X^\top] = \Sigma + \mu \mu^\top.
```

Input observations are real-valued level proxies already in estimation units (such as log-prices). Candidates implement a fixed Python interface ([`src/covbench/api.py`](src/covbench/api.py)); their internals may use any language, compiler, or hardware accelerator.

Historical-window estimation and next-window forecasting are evaluated as separate tracks, each supporting offline and online execution. Reports retain stage timings, peak memory, ten statistical loss functions, structural diagnostics, and failure records, and compare candidates via per-regime Pareto frontiers rather than a single global score.

## Quick Start

Requires Python 3.10+ on Linux:

```sh
python -m pip install '.[test]'

covbench run --candidate candidates/previous-tick --suite suites/smoke.json \
  --root-seed 20261013 --replications 3 --timing-repeats 3 --output runs/tick
covbench run --candidate candidates/zero --suite suites/smoke.json \
  --root-seed 20261013 --replications 3 --timing-repeats 3 --output runs/zero
covbench frontier runs/tick/results.json runs/zero/results.json \
  --objectives completed_ns covariance_nmse --output runs/frontier.json
```

Use a fresh `--output` directory for each run. Matching `--root-seed` values produce identical synthetic realizations across candidates; omit `--root-seed` to draw a random 64-bit evaluator seed. `--batch-size 1` measures single-event online delivery (default `128`). `--threads` sets worker CPU affinity and numerical library thread limits (default `1`). Default resource limits are a 60-second per-command timeout (`--timeout`) and a 1,024 MiB worker peak RSS ceiling (`--memory-mib`).

Each run writes `<output>/results.json` with per-query outcomes and per-regime summaries, along with worker logs and saved prediction files under `<output>/trials/`. Timing summaries report sample medians across trials; loss summaries average timing repeats within each synthetic replication before computing the mean and Monte Carlo standard error across independent draws. Failed or incomplete trials remain visible in the report and prevent a complete-case summary score for that regime.

To recompute statistical losses from saved predictions without re-executing candidate code:

```sh
covbench regrade runs/tick/results.json --output runs/tick/regraded.json
```

`regrade` verifies the SHA-256 digests of the saved prediction arrays and regenerated scenario inputs, preserving the original measurement identity alongside the updated scoring identity. Precomputed core-suite reports for the three included baselines are provided in [`examples/`](examples/README.md).

Run the test suite with:

```sh
python -m pytest -q
```

## Candidate Interface

A candidate is a directory containing a `candidate.json` manifest and an importable Python factory:

```json
{
  "name": "my-method",
  "api_version": 1,
  "entrypoint": "candidate:create",
  "config": {},
  "modes": ["offline", "online"]
}
```

`create(problem, config)` returns an estimator implementing `fit`, `update`, `predict`, and `synchronize`. A `Prediction` contains a finite real host NumPy array `mean` of shape `(n_assets,)` and a covariance estimate in either `DenseMatrix(value)` form ($`\hat{\Sigma} \in \mathbb{R}^{N \times N}`$) or `FactorMatrix(loadings, specific)` form ($`\hat{\Sigma} = B B^\top + \mathrm{diag}(d)`$ with $`d \ge 0`$). Candidate dependencies must be installed in the environment prior to evaluation.

Online `update` calls deliver arrival-ordered point edits (`UPSERT` and `DELETE`) keyed by `(asset, time)`, including delayed arrivals, retroactive value corrections, and deletions. The primary timing metric `completed_ns` sums all work required at a checkpoint: both checkpoint `predict` calls (including device synchronization and copying outputs into contiguous `float64` host arrays), plus the checkpoint's offline `fit` or intervening online `update` batches and applicable estimator construction. Every timing repetition runs in a fresh worker process, so any JIT compilation triggered during an API call is charged to that stage.

## Suites, Losses, and Baselines

| Suite | Purpose |
| --- | --- |
| [`suites/smoke.json`](suites/smoke.json) | Three small scenarios ($`N \in \{4, 8\}`$) for contract and integration checks. |
| [`suites/core.json`](suites/core.json) | Twelve scenarios ($`N \in \{16, 64, 256, 1024\}`$) spanning varied distributions, sampling schemes, noise processes, and factor ranks. |
| [`suites/scale.json`](suites/scale.json) | Three optional 4,096-asset scenarios with compact factor truth evaluated under the runner's time and memory limits. |
| [`suites/holdout.json`](suites/holdout.json) | Six scenarios ($`N \in \{32, 96, 384, 768\}`$, three regimes) for validating a frozen candidate on a fresh seed. |

The synthetic generators cover Gaussian, elliptical Student-$`t`$, and compound Poisson jump-diffusion increments, piecewise-constant regime changes, synchronous/asynchronous/sparse/imbalanced/bursty sampling, i.i.d./correlated/autoregressive microstructure noise, and delayed edits. Candidates receive only problem dimensions, causal observations or events, and query intervals; true parameters and seeds remain private to the evaluator.

Reported losses include normalized Frobenius errors on the covariance, second moment, and diagonal; variance-scaled and Mahalanobis mean errors; pairwise correlation MSE; relative portfolio variance and second-moment errors; and block Gaussian KL and QLIKE divergences. Indefinite dense estimates retain all algebraic losses (and correlation MSE whenever marginal variances are strictly positive), while metrics whose domain conditions fail are recorded as `null`.

Included baselines are [`candidates/zero`](candidates/zero/README.md) (zero moments), [`candidates/previous-tick`](candidates/previous-tick/README.md) (grid-interpolated sample covariance), and [`candidates/async-native`](candidates/async-native/README.md) (a window-refit adapter for the optional [`async-covariance`](https://github.com/epicycloids/async-covariance) package).

## Documentation

- [`docs/benchmark-design.md`](docs/benchmark-design.md): statistical contract, online edit semantics, worker isolation and timing model, aggregation rules, provenance, and design conventions.
- [`docs/scenarios.md`](docs/scenarios.md): mathematical specification of the latent increment laws, regime parameters, sampling schedules, noise models, arrival revisions, and suite configurations.
- [`docs/metrics.md`](docs/metrics.md): definitions of the ten loss functions, low-rank QR and Woodbury linear algebra, Rademacher trace estimation, PSD diagnostics, and Pareto dominance.
- [`tasks/implement-estimator.md`](tasks/implement-estimator.md): specification and workflow for implementing, testing, and validating a candidate estimator.
- [`examples/README.md`](examples/README.md): guide to the reference core-suite evaluation reports.
