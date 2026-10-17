# Benchmark Design

`covariance-bench` evaluates algorithms that estimate the mean and covariance of latent price increments from irregular, noisy, and potentially revised observations. Rather than reducing performance to a single scalar score, the harness records stage-level execution times, memory usage, ten statistical loss functions, and structural diagnostics across distinct scenario regimes, tracks, and execution modes.

The public candidate interface is defined in [`src/covbench/api.py`](../src/covbench/api.py).

## 1. Statistical Contract

### 1.1 Target Increment Moments

Let $`X(t) \in \mathbb{R}^N`$ denote a latent vector process over $`N`$ assets on the physical time horizon $`[0, T]`$, initialized at $`X(0) = 0`$. A query specifies a cutoff timestamp $`t_{\mathrm{asof}} \in [0, T]`$ and a target interval $`[t_{\mathrm{start}}, t_{\mathrm{end}}] \subseteq [0, T]`$ with $`t_{\mathrm{start}} < t_{\mathrm{end}}`$.

The estimands are the first and second moments of the latent increment

$$
\Delta X(t_{\mathrm{start}}, t_{\mathrm{end}}) = X(t_{\mathrm{end}}) - X(t_{\mathrm{start}}) \in \mathbb{R}^N
$$

conditional on the draw's piecewise-constant regime parameters:

$$
\mu = \mathbb{E}[\Delta X(t_{\mathrm{start}}, t_{\mathrm{end}})] \in \mathbb{R}^N, \qquad \Sigma = \mathrm{Cov}(\Delta X(t_{\mathrm{start}}, t_{\mathrm{end}})) \in \mathbb{R}^{N \times N}.
$$

The implied uncentered second-moment matrix is

$$
M = \mathbb{E}\!\left[\Delta X(t_{\mathrm{start}}, t_{\mathrm{end}}) \, \Delta X(t_{\mathrm{start}}, t_{\mathrm{end}})^\top\right] = \Sigma + \mu \mu^\top.
$$

Both $`\mu`$ and $`\Sigma`$ are integrated interval quantities over $`[t_{\mathrm{start}}, t_{\mathrm{end}}]`$, expressed in the level units of $`X`$: under constant per-unit-time drift rate $`\mu_{\mathrm{rate}}`$ and covariance rate $`\Sigma_{\mathrm{rate}}`$, an interval of duration $`\Delta\tau = t_{\mathrm{end}} - t_{\mathrm{start}}`$ has $`\mu = \mu_{\mathrm{rate}} \Delta\tau`$ and $`\Sigma = \Sigma_{\mathrm{rate}} \Delta\tau`$, while the outer-product term $`\mu \mu^\top = (\Delta\tau)^2 \mu_{\mathrm{rate}} \mu_{\mathrm{rate}}^\top`$ scales quadratically with $`\Delta\tau`$.

### 1.2 Historical and Forecast Tracks

Each scenario issues queries at three checkpoints $`t_{\mathrm{asof}} \in (0, T)`$. At every checkpoint, two queries of equal duration $`w \Delta t`$ are evaluated:

1. **Historical track** (`track = "historical"`): $`t_{\mathrm{start}} = t_{\mathrm{asof}} - w \Delta t`$ and $`t_{\mathrm{end}} = t_{\mathrm{asof}}`$. The candidate estimates the marginal moments of the latent increment over the observed physical-time window $`[t_{\mathrm{start}}, t_{\mathrm{asof}}]`$.
2. **Forecast track** (`track = "forecast"`): $`t_{\mathrm{start}} = t_{\mathrm{asof}}`$ and $`t_{\mathrm{end}} = t_{\mathrm{asof}} + w \Delta t`$. The candidate predicts the marginal moments of the latent increment over the upcoming window $`[t_{\mathrm{asof}}, t_{\mathrm{end}}]`$ using only information that has arrived by $`t_{\mathrm{asof}}`$.

In both tracks, the target $`(\mu, \Sigma)`$ is the marginal moment pair of the latent increment $`\Delta X(t_{\mathrm{start}}, t_{\mathrm{end}})`$ excluding observation noise, rather than the posterior distribution of the realized path $`X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$ conditional on noisy observations.

### 1.3 Information Boundary

Candidates receive only the problem specification `Problem(n_assets, mode, time_unit)`, their own manifest `config`, the causal observation set available at $`t_{\mathrm{asof}}`$, and the `Query` instances.

The evaluator retains all ground-truth regime parameters ($`\mu_r, B_r, d_r`$), latent sample paths $`X(t)`$, observation noise parameters, random seeds, and events with arrival timestamps $`a > t_{\mathrm{asof}}`$. Candidate configuration is part of the frozen artifact and must not be used as an oracle channel for test-instance parameters.

## 2. Observation Table and Online Event Semantics

### 2.1 Measurement Time vs. Arrival Time

An observation is identified by the composite key $`(i, t)`$, where $`i \in \{0, \dots, N-1\}`$ is the asset index and $`t \in [0, T]`$ is the physical measurement timestamp. Because feeds may experience transmission latency, retroactive corrections, or retractions, each event also carries an arrival timestamp $`a \ge t`$.

An event is a tuple $`(a_m, t_m, i_m, y_m, \mathrm{op}_m)`$ where:

- $`\mathrm{op}_m = \mathrm{UPSERT}`$ (`0`): inserts or replaces the value associated with key $`(i_m, t_m)`$ by $`y_m \in \mathbb{R}`$.
- $`\mathrm{op}_m = \mathrm{DELETE}`$ (`1`): removes key $`(i_m, t_m)`$ from the active observation table; the field $`y_m`$ is ignored.

Because arrival latencies vary across observations and corrections arrive after the initial publication, the sequence of measurement timestamps $`(t_m)`$ in an arrival-ordered stream is not monotonic. Candidates must treat an edit to an existing key $`(i_m, t_m)`$ as a replacement or deletion of that point rather than as an additional independent observation.

### 2.2 Causal Prefix Equivalence

At any cutoff timestamp $`t_{\mathrm{asof}}`$, let $`\mathcal{E}(t_{\mathrm{asof}}) = \{m : a_m \le t_{\mathrm{asof}}\}`$ be the prefix of events arriving on or before $`t_{\mathrm{asof}}`$. Replaying $`\mathcal{E}(t_{\mathrm{asof}})`$ in arrival order defines a finite map

$$
\mathcal{T}(t_{\mathrm{asof}}) : \mathcal{K}(t_{\mathrm{asof}}) \subset \{0, \dots, N-1\} \times [0, T] \longrightarrow \mathbb{R},
$$

containing every key $`(i, t)`$ whose most recent operation in $`\mathcal{E}(t_{\mathrm{asof}})`$ is `UPSERT`.

- In **offline mode**, at each checkpoint $`t_{\mathrm{asof}}`$, the evaluator constructs `Observations(time, asset, value)` by sorting the entries of $`\mathcal{T}(t_{\mathrm{asof}})`$ lexicographically by $`(t, i)`$ and passes this snapshot to `estimator.fit(observations)`.
- In **online mode**, the evaluator delivers the incremental slices of $`\mathcal{E}(t_{\mathrm{asof}})`$ in batches of size at most `batch_size` to `estimator.update(events)`.

An online estimator that maintains the state of $`\mathcal{T}(t_{\mathrm{asof}})`$ therefore observes the exact same underlying point set at each checkpoint as an offline estimator.

## 3. Candidate Artifact and API Contract

### 3.1 Manifest Specification

A candidate is a directory containing source files and a `candidate.json` manifest:

```json
{
  "name": "example-estimator",
  "entrypoint": "module:create",
  "config": {},
  "api_version": 1,
  "modes": ["offline", "online"]
}
```

- `name`: non-empty string identifying the candidate.
- `entrypoint`: Python import specification `module:factory` (resolved relative to the candidate directory).
- `config`: JSON object passed as a dictionary to `factory(problem, config)`.
- `api_version`: integer version of the benchmark interface (`1`).
- `modes`: non-empty list selecting `"offline"`, `"online"`, or both.

At the start of `covbench run`, the candidate directory is copied into `<output>/candidate` (excluding `.git`, `.venv`, `venv`, `__pycache__`, `.pytest_cache`, and `target`), and its SHA-256 directory tree digest is verified before and after evaluation.

### 3.2 Matrix Representations and Output Validation

`estimator.predict(query)` returns `Prediction(mean, covariance)`, where `mean` and the arrays inside `covariance` must be finite real host NumPy arrays (floating-point, signed integer, or unsigned integer; `dtype.kind in "fiu"`). `mean` has shape `(n_assets,)`, and `covariance` is an instance of one of two API classes:

1. `DenseMatrix(value)`: `value` is a finite 2D real NumPy array $`\hat{\Sigma} \in \mathbb{R}^{N \times N}`$ satisfying the relative symmetry condition $`\max_{0 \le i, j < N} |\hat{\Sigma}_{ij} - \hat{\Sigma}_{ji}| \le 64 \varepsilon_{\mathrm{mach}} \max_{0 \le u, v < N} |\hat{\Sigma}_{uv}|`$, where $`\varepsilon_{\mathrm{mach}} = 2^{-52} \approx 2.22 \times 10^{-16}`$.
2. `FactorMatrix(loadings, specific)`: `loadings` is a finite 2D real NumPy array $`B \in \mathbb{R}^{N \times K}`$ ($`K \ge 0`$) and `specific` is a finite nonnegative 1D real NumPy array $`d \in \mathbb{R}_{\ge 0}^N`$ with finite implied diagonal $`\mathrm{diag}(B B^\top) + d`$, representing $`\hat{\Sigma} = B B^\top + \mathrm{diag}(d)`$.

During the timed `predict` stage, the worker validates these conditions and copies the arrays into contiguous `float64` host NumPy arrays.

API validity and statistical domain requirements are evaluated separately:

- A `FactorMatrix` is positive semidefinite by construction.
- A valid `DenseMatrix` may be indefinite. When a dense estimate is indefinite, algebraic losses (`covariance_nmse`, `second_moment_nmse`, `diagonal_nmse`, `portfolio_risk_relative_mse`, `portfolio_second_moment_relative_mse`, `mean_variance_scaled_mse`, `mean_mahalanobis`) are still evaluated.
- `correlation_mse` requires only strictly positive marginal variances ($`\hat{\Sigma}_{ii} > 0`$ and $`\Sigma_{ii} > 0`$ for all $`i`$), so an indefinite matrix with positive diagonal entries still receives a finite correlation score.
- `gaussian_kl` and `covariance_qlike` require that neither matrix is diagnosed as full-matrix `"indefinite"` (checked via symmetric eigenvalues for dense matrices with $`N \le N_{\mathrm{exact}}`$) and that Cholesky factorization succeeds on every declared principal block $`\mathcal{B}_b`$. A globally singular positive semidefinite matrix can therefore pass when every declared proper block is strictly positive definite; if full-matrix diagnosis is `"indefinite"` or any declared block fails Cholesky factorization, both metrics are recorded as `null` (`None` in Python).

## 4. Execution Architecture and Timing Model

### 4.1 Worker Subprocess Protocol

Each trial runs in a dedicated child process (`python -m covbench.worker`) managed by [`src/covbench/runner.py`](../src/covbench/runner.py):

- **Communication**: the runner sends newline-delimited JSON commands (`load`, `create`, `fit`, `update`, `predict`, `close`) to the worker's `stdin` and reads JSON replies from `stdout`. During candidate execution, Python's `sys.stdout` is redirected to `sys.stderr` (captured in `worker.log`) so candidate output cannot corrupt the protocol stream.
- **Data transfer**: array inputs (`Observations` and `Events`) are written by the runner to a local `.npz` file and loaded into read-only NumPy arrays by the worker before starting the stage timer.
- **Resource controls**:
  - Thread limits (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `RAYON_NUM_THREADS`) are set to `--threads`, and `PYTHONHASHSEED` is fixed to `"0"`.
  - When supported by the operating system (`os.sched_setaffinity`), the worker is pinned to the first `--threads` logical CPUs available to the runner.
  - Each command is subject to a wall-clock deadline (`--timeout`, default 60 s) and a peak resident set size ceiling (`--memory-mib`, default 1,024 MiB) monitored via `VmHWM` in `/proc/<pid>/status`.
- **Isolation scope**: subprocess execution isolates process memory and crashes between trials, but does not provide a container, filesystem, or network sandbox. Candidate processes retain the permissions of the invoking user account.

### 4.2 Stage Timings, Synchronization, and Materialization

To measure asynchronous accelerators (such as CUDA GPUs) accurately, the worker defines a barrier `sync()` that calls `estimator.synchronize()`, and when `torch` (with CUDA initialized) or `cupy` is already imported in `sys.modules`, also calls `torch.cuda.synchronize()` or `cupy.cuda.runtime.deviceSynchronize()`.

For `create`, `fit`, and `update`, the worker calls `sync()` before starting `time.perf_counter_ns()` (flushing any prior work), executes the candidate method, and calls `sync()` **before** stopping the timer, so that completion of all work launched by the stage is charged to `duration_ns`.

For `predict(query)`, the worker calls `sync()`, starts the timer, and executes the following sequence before stopping `predict_ns`:

1. `prediction = estimator.predict(query)`
2. `sync()` to complete any device kernels launched during prediction
3. `materialize(prediction, problem.n_assets)` to validate shapes, symmetry, and finiteness and to copy `mean` and `covariance` into freshly allocated, contiguous `float64` host NumPy arrays
4. `sync()` before recording `duration_ns`

Writing the materialized host arrays to `prediction-XXXX.npz` on disk and computing statistical losses in the runner occur after the timer stops. The exact prediction produced during the timed query is the one saved and scored.

Every `(replication, timing_repeat, mode)` trial spawns a fresh worker process. Installed packages, OS file caches, and external compiler caches are not purged, but in-memory process state is reset each trial, and any JIT compilation triggered during `load`, `create`, `fit`, `update`, or `predict` is charged to that stage. Candidates must not cache instance answers across independent trials.

### 4.3 Checkpoint Work Accounting

Let $`c \in \{0, 1, 2\}`$ index the three checkpoints in ascending order of $`t_{\mathrm{asof}}`$, and let $`t_{\mathrm{predict}, c, \mathrm{hist}}`$ and $`t_{\mathrm{predict}, c, \mathrm{fore}}`$ denote the measured `predict_ns` durations (including prediction, synchronization, and `float64` host materialization) of the historical and forecast queries at checkpoint $`c`$.

Because an estimator may trade off ingestion work in `fit`/`update` against query work in `predict`, the primary timing coordinate `completed_ns` sums all work required at checkpoint $`c`$, including both required predictions and their materialization:

**Offline mode.** At each checkpoint $`c`$, a fresh estimator is constructed and fitted to the materialized snapshot before answering both queries:

$$
\mathrm{completed\_ns}_c = t_{\mathrm{create}, c} + t_{\mathrm{fit}, c} + t_{\mathrm{predict}, c, \mathrm{hist}} + t_{\mathrm{predict}, c, \mathrm{fore}}.
$$

**Online mode.** Let $`\mathcal{U}_c`$ denote the batch `update` calls processed between checkpoint $`c-1`$ and checkpoint $`c`$ (containing $`N_{\mathrm{events}, c}`$ total events), with total update time $`t_{\mathrm{update}, c} = \sum_{b \in \mathcal{U}_c} t_{\mathrm{update}, b}`$. Then:

$$
\mathrm{completed\_ns}_c = \mathbf{1}_{\{c = 0\}} t_{\mathrm{create}} + t_{\mathrm{update}, c} + t_{\mathrm{predict}, c, \mathrm{hist}} + t_{\mathrm{predict}, c, \mathrm{fore}}.
$$

For online trials, each query record also reports:

- `update_total_ns`: $`t_{\mathrm{update}, c}`$
- `n_events`: $`N_{\mathrm{events}, c}`$
- `amortized_event_ns`: $`\mathrm{completed\_ns}_c / N_{\mathrm{events}, c}`$ (or `null` if $`N_{\mathrm{events}, c} = 0`$)
- `update_ns_per_event`: $`t_{\mathrm{update}, c} / N_{\mathrm{events}, c}`$ (or `null` if $`N_{\mathrm{events}, c} = 0`$)
- `update_batch_median_ns` and `update_batch_p95_ns`: the median and 95th percentile of $`\{t_{\mathrm{update}, b}\}_{b \in \mathcal{U}_c}`$

## 5. Statistical Aggregation, Provenance, and Regrading

### 5.1 Two-Level Experimental Design

For a given `root_seed`, each scenario with identifier `scenario_id` is simulated across $`R_{\mathrm{rep}}`$ independent replications ($`r \in \{0, \dots, R_{\mathrm{rep}} - 1\}`$). The instance seed for replication $`r`$ is derived deterministically from the first 64 bits of a SHA-256 digest:

$$
\mathrm{seed}(r) = \mathrm{int}_{16}\!\left(\mathrm{SHA256}(\texttt{[root\_seed, scenario\_id, } r\texttt{]})[:16]\right).
$$

Each synthetic dataset is evaluated across $`R_{\mathrm{time}}`$ cold-process timing repetitions ($`m \in \{0, \dots, R_{\mathrm{time}} - 1\}`$). The random seed used to sample correlation pairs, random portfolios, and Gaussian sub-blocks for query index $`q \in \{0, \dots, 5\}`$ is $`(\mathrm{seed}(r) \oplus \texttt{0xD7AD5C32}) + q \pmod{2^{64}}`$, which depends only on the dataset seed and query index.

### 5.2 Summary Aggregation

The runner groups query outcomes by `(scenario_id, mode, track, checkpoint)`. Each group contains $`R_{\mathrm{rep}} \times R_{\mathrm{time}}`$ records.

**Status.** A summary group has `status = "failed"` if any trial in the group failed (timeout, memory limit, crash, or invalid output), `"partial"` if all trials succeeded but at least one trial had an undefined domain-restricted metric, and `"ok"` if every metric succeeded in every trial.

**Timings and memory.** Each timing field (`completed_ns`, `fit_ns`, `predict_ns`, `update_total_ns`, `amortized_event_ns`, `update_ns_per_event`, `update_batch_median_ns`, `update_batch_p95_ns`) is summarized by the sample median across all $`R_{\mathrm{rep}} \times R_{\mathrm{time}}`$ trials (or `null` if any trial is missing that timing). `peak_rss_bytes` is the maximum across all trials in the group.

**Loss means and Monte Carlo standard errors.** For each metric $`\ell`$, timing repetitions within the same replication $`r`$ are averaged first: $`\bar{\ell}_r = \frac{1}{R_{\mathrm{time}}} \sum_{m=0}^{R_{\mathrm{time}}-1} \ell_{r,m}`$. If every trial in the group has a finite value for $`\ell`$ and `failures == 0`, the summary reports the replication mean and Monte Carlo standard error across the $`R_{\mathrm{rep}}`$ independent synthetic draws:

$$
\hat{\ell} = \frac{1}{R_{\mathrm{rep}}} \sum_{r=0}^{R_{\mathrm{rep}}-1} \bar{\ell}_r, \qquad \mathrm{SE}(\hat{\ell}) = \sqrt{\frac{1}{R_{\mathrm{rep}}(R_{\mathrm{rep}} - 1)} \sum_{r=0}^{R_{\mathrm{rep}}-1} (\bar{\ell}_r - \hat{\ell})^2} \quad (R_{\mathrm{rep}} > 1).
$$

If any trial in the group fails or produces `null` for $`\ell`$, the summary sets `record[name] = null` and records `n_complete` (the number of replications in which all timing repeats produced a non-null value).

### 5.3 Provenance and Regrading

Every report records cryptographic digests of the experimental inputs:

- `candidate_sha256`: SHA-256 tree hash of the frozen candidate directory.
- `evaluator_sha256`: SHA-256 tree hash of `src/covbench/`.
- `suite_sha256`: SHA-256 file hash of the suite JSON specification.
- `inputs_sha256`: SHA-256 digest of the generated event arrays, arrival timestamps, and queries for each trial.
- `output_sha256`: SHA-256 file hash of each saved `prediction-XXXX.npz` file.
- `comparison_key`: SHA-256 digest combining `suite_sha256`, `evaluator_sha256`, observed hardware/software environment, execution policy (`threads`, `timeout`, `memory_mib`, `batch_size`, `timing_repeats`, `replications`), and `root_seed`.

When `covbench regrade` is run on an existing `results.json`, it regenerates each trial's dataset from `(scenario_config, instance_seed)`, checks that the regenerated `inputs_sha256` and each saved file's `output_sha256` match the recorded hashes, recomputes `score(...)`, and writes a new report recording both `measurement_evaluator_sha256` and `scoring_evaluator_sha256`.

## 6. Development and Holdout Protocol

1. **Development phase**: use `suites/smoke.json` for fast verification and `suites/core.json` (or `suites/scale.json`) with a fixed `--root-seed` to compare candidate revisions and hyperparameter configurations against the baseline estimators on identical synthetic realizations.
2. **Selection and freezing**: select candidate configurations on the per-regime Pareto frontier (`covbench frontier`), then freeze the candidate directory and `candidate.json`.
3. **Holdout validation**: evaluate the frozen candidate alongside baselines using a fresh `--root-seed` on `suites/holdout.json`. Fresh seeds test new random draws from a given scenario specification, whereas `suites/holdout.json` tests generalization across distinct asset counts, regime counts, innovation parameters, and drift scales. Because `suites/holdout.json` is public in the repository, reusing held-out feedback to guide further tuning turns those cases into development data; subsequent validation requires a fresh seed and held-out configuration.

## 7. Related Benchmark Literature

Several structural conventions in `covariance-bench` adapt ideas from established software and systems benchmarks (without claiming compatibility or shared dependencies):

- **SWE-bench** ([evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)): separates submitted artifacts from per-instance harness execution and structured reporting.
- **Harbor / Terminal-Bench** ([task overview](https://docs.harborframework.com/tasks/overview); [separate verifier](https://docs.harborframework.com/tasks/separate-verifier)): separates task instructions, candidate solution, and evaluator-owned verification (adopted here without Harbor's container isolation).
- **Airspeed Velocity (ASV)** ([writing benchmarks](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)): separates setup, import, and timed execution phases and makes process-level state resets explicit.
- **MLPerf Inference LoadGen** ([LoadGen scope](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/README.md#scope-of-the-loadgens-responsibilities); [inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#51-loadgen-operation)) and **PyTorch CUDA semantics** ([asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda#asynchronous-execution)): require device synchronization before stopping timers and score the exact outputs produced during timed execution.
- **AlgoTune** ([paper](https://algotune.io/paper.pdf#page=4); [timed-output check](https://github.com/oripress/AlgoTune/blob/dff9914c10800c7a031c9e8c3d4d1c8cd1b38906/AlgoTuner/utils/evaluator/solution_checks.py#L245)): pairs seeded input generation, reference baselines, iterative development feedback, and held-out evaluation of candidate code, while scoring the timed output directly.
