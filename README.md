# covariance-bench

A benchmark for code that estimates covariance and second moments from sparse,
asynchronous price observations. Candidates implement a fixed Python API;
their internals can use any language or accelerator.
Inputs are real-valued levels already in estimation units, not raw prices that
the candidate should logarithmize.

Historical-window estimation and next-window prediction are separate tracks.
Each has offline and online execution. Results retain timing, memory, statistical
loss, diagnostics, and failures. Pareto views compare identical workloads and
environments; there is no global scalar score.

## Run

Python 3.10+, Linux:

```sh
python -m pip install '.[test]'
covbench run --candidate candidates/previous-tick --suite suites/smoke.json \
  --root-seed 20261013 --replications 3 --timing-repeats 3 --output runs/tick
covbench run --candidate candidates/zero --suite suites/smoke.json \
  --root-seed 20261013 --replications 3 --timing-repeats 3 --output runs/zero
covbench frontier runs/tick/results.json runs/zero/results.json \
  --objectives completed_ns covariance_nmse --output runs/frontier.json
```

Use an unused output directory for every run. Matching root seeds give paired
draws across candidates. Omit the seed for a fresh evaluator draw. `--batch-size 1`
measures scalar online delivery; the default is 128. `--threads` sets worker CPU
affinity and numerical thread limits. Different policies get different comparison
keys. A 60-second per-command timeout and 1 GiB worker RSS ceiling are defaults.

`results.json` contains raw per-query outcomes and per-regime summaries. Each
trial also saves logs and the exact measured predictions for regrading. Timing
summaries use medians; loss means and Monte Carlo errors use independent simulated
draws, averaging repeated timings within each draw. Missing or failed trials
remain visible and prevent a complete-case headline score.

## Submit code

Create a directory containing `candidate.json` and an importable factory:

```json
{"name":"my-method","api_version":1,"entrypoint":"candidate:create",
 "config":{},"modes":["offline","online"]}
```

`create(problem, config)` returns `fit`, `update`, `predict`, and `synchronize`
methods. A prediction contains a mean vector and a complete dense covariance or
factor-plus-diagonal representation of **`X(end)-X(start)`**. The second moment
is `covariance + outer(mean, mean)`. Use [API v1](src/covbench/api.py) and the
[code-generation task](tasks/implement-estimator.md) as the outer loop's contract.
Install declared dependencies before evaluating; the runner does not install
candidate dependencies or call an LLM.

Online updates include replacement, deletion, and delayed observations. Query
work is timed: `completed_ns` includes all required predictions at a checkpoint,
plus its fit or intervening updates and applicable construction. An inexpensive
update that defers work to `predict` still pays that cost. Import/startup are
recorded separately. Every timing repetition starts a fresh process; compilation
triggered by an API call is included. Host output materialization and device
completion are included; evaluator transport, output-file I/O, and scoring are
excluded.

## Workloads and losses

| Suite | Purpose |
| --- | --- |
| `smoke` | Three tiny cases for contract and integration checks |
| `core` | Twelve cases, 16–1,024 assets, varied covariance ranks |
| `scale` | Three optional 4,096-asset cases with compact truth |
| `holdout` | Different sizes/configurations for a frozen candidate |

The [generators](docs/scenarios.md) cover Gaussian, Student-t and jump increments,
changing covariance/drift, sparse and imbalanced observation schedules, multiple
noise models, and revisions. Covariance parameters are randomized per draw.
Candidates receive no true rank, drift, noise parameters, future observations, or
generator seed.

Losses include normalized covariance and second-moment squared error, diagonal,
mean, correlation and portfolio-risk error, and Gaussian KL/QLIKE on stated
blocks. Exact and sampled metrics identify their scopes. Dense indefinite
estimates keep algebraic losses; SPD-only losses are explicitly unavailable.
The evaluator never silently repairs a candidate's matrix. See
[loss definitions](docs/metrics.md) and
[measurement and design](docs/benchmark-design.md) for semantics and the
conventions adopted from AlgoTune, SWE-bench, Harbor, ASV and MLPerf.

Baselines are a zero control, previous-tick sample covariance, and an optional
[async-covariance](https://github.com/epicycloids/async-covariance) refit adapter.
The native adapter needs that package installed; its window refitting is charged
to query time. None defines the benchmark's ground truth.

## Improve and validate

Use core runs for feedback, retain candidate configurations on the frontier,
then freeze source/configuration before comparing them on a fresh shared root
seed and the holdout suite. Held-out configurations are public, so they become
development cases once used for tuning. Frontier membership describes point
estimates; retain loss uncertainty and paired draws when assessing improvement.

This version uses synthetic truth. Real-data evaluation will need an independent
observable target or likelihood and its sampling uncertainty; applying the same
candidate to the future data would not make its output ground truth.

```sh
python -m pytest -q
```
