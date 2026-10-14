# Benchmark design

This private benchmark compares candidate estimators of latent price increment
moments on synthetic observations. Its output is a speed–loss comparison within
each regime, rather than one global fitness score. The candidate API is defined
in [`src/covbench/api.py`](../src/covbench/api.py).

## Statistical contract

A query requests the mean and covariance of the full latent increment
`X(end) - X(start)`, in level units. Historical queries estimate marginal moments
over an observed physical-time window. Forecast queries predict moments over a
future interval using only information available at `asof`. Neither track targets
the covariance of noisy observations, the instantaneous covariance rate, nor a
single realized increment. The second moment is derived as
`covariance + outer(mean, mean)`; drift therefore matters even when covariance is
well estimated.

Candidates receive asset count, mode, time units, observations or arrival-ordered
edits, and query windows. The evaluator retains latent truth, generator seeds,
noise parameters, drift parameters, and structural rank parameters. Future
observations are not supplied before their arrival. Candidate configuration is
part of the submitted artifact, not an oracle channel for a particular test
instance.

An offline fit receives the current point table. Online edits are keyed by
`(asset, time)`: an upsert replaces that key and a delete removes it. Arrival order
and observation time are different. A candidate must handle corrections and
deletions, including an edit to an older point, without treating them as new
independent observations.

## Submission and execution

Each candidate directory contains source and a `candidate.json` manifest:

```json
{
  "name": "example-estimator",
  "entrypoint": "module:create",
  "config": {},
  "api_version": 1,
  "modes": ["offline", "online"]
}
```

The factory `create(problem, config)` returns an estimator with `fit`, `update`,
`predict`, and `synchronize` methods. Predictions contain a mean vector and the
API's `DenseMatrix` or `FactorMatrix`, with complete finite host NumPy arrays.
Factor form represents `loadings @ loadings.T + diag(specific)`, with
nonnegative specific variances. A dense estimate may be indefinite: the evaluator
records this diagnosis and does not silently project or regularize it.

The runner evaluates frozen artifacts in local subprocesses and repeats trials
with fresh workers. Candidate processes receive only the declared API inputs;
scoring uses their returned predictions in the evaluator. This is process and
state separation, not a filesystem or network sandbox. Local candidates remain
able to access resources available to their user account. There is no Docker
backend or claim of hostile-code containment in this implementation.

## Measurement and scoring

Import, factory creation, offline fit, online updates, and prediction are timed
separately. Cold repetitions use new processes and new estimators. There is
currently no excluded JIT or warmup phase: lazy compilation is charged to the
stage that triggers it. Retain raw measurements and report the aggregation used
for comparison; a timing summary is not the wall time of the whole command.
Fresh workers reset process state; installed binaries, operating-system file
caches and external compiler caches are not purged. Candidates must not cache
instance answers across independent trials. These runs measure process-cold
execution, not installation from an empty machine.

An estimator's `synchronize()` completes outstanding device work; CPU estimators
use a no-op. Query timing includes completion and materialization of the returned
host arrays. A candidate must not return a lazy object that moves computation
into scoring. The exact prediction produced by the measured query is scored;
there is no separate untimed rerun to obtain a better prediction. CUDA work is
asynchronous, so synchronization is necessary to measure completed work, as
described in the [PyTorch CUDA documentation](https://docs.pytorch.org/docs/stable/notes/cuda#asynchronous-execution).

API validity and statistical quality are separate. Check output dimensions,
finiteness, canonical representation, and the declared interface before scoring.
Report mean, covariance, and second-moment losses and structural diagnostics
without requiring exact reproduction of a reference estimate. Algebraic losses
remain meaningful for an indefinite dense estimate. Metrics requiring positive
definiteness are flagged undefined when their domain conditions fail, never
given a silently repaired matrix or a zero loss.

Compare candidates on the same synthetic instances. Preserve failures and
timeouts, rather than dropping them to improve an average. A Pareto comparison
uses a declared timing coordinate and loss coordinate for the same regime,
workload, and environment. Separate frontiers preserve differences between asset
counts, observation patterns, distributional regimes, historical and forecast
tracks, and offline and online operation. They do not establish dominance over
unmeasured regimes or combine all losses into a global scalar reward.

## Development and held-out use

Development runs provide tuning feedback. Freeze a candidate's source and config
before a held-out run, and use a fresh held-out root seed outside candidate
inputs. Record the suite configuration, generated workload identity, and seed
provenance in evaluator-owned results. Reusing the same held-out feedback for
further tuning makes it development evidence; it is not fresh validation. New
seeds assess new draws from a regime, while new regime configurations are needed
to assess broader generalization.

The outer loop is: implement or revise a candidate, freeze the artifact, run the
development suite, inspect per-regime speed–loss comparisons, freeze the selected
version, then run a fresh held-out seed configuration. A useful baseline suite
includes simple estimators and stronger statistical baselines. Their timing and
loss are empirical reference points, not definitions of the correct output.

Results retain candidate/config hashes, suite identity and hash, environment
details, raw stage timings, per-query losses and diagnostics, and explicit failure
records. Preserve enough source, configuration, and runtime information to
reproduce a claimed improvement. Comparisons across different environments or
timing policies should be labeled as such.

## Conventions reused from other benchmarks

These are design adaptations, not compatibility claims or dependencies:

- [SWE-bench evaluation](https://www.swebench.com/SWE-bench/guides/evaluation/)
  separates submitted artifacts from per-instance execution and reports. Here the
  artifact is an estimator package, with an explicit identity for each run.
- [Terminal-Bench's Harbor task format](https://docs.harborframework.com/tasks/overview)
  separates instruction, environment, solution, and tests;
  [Harbor's separate verifier](https://docs.harborframework.com/tasks/separate-verifier)
  reinforces evaluator ownership of scoring. Here those responsibilities are
  separated without importing Harbor's container guarantees.
- [ASV's benchmark guidance](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)
  makes setup and repeated-state behavior explicit. Here independent cold workers
  and separate stage timings make the amount of reusable state visible.
- [MLPerf LoadGen](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/README.md#scope-of-the-loadgens-responsibilities)
  keeps measurement separate from task-specific accuracy;
  [its rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#51-loadgen-operation)
  require consistent code and outputs between timing and accuracy. Here the
  measured prediction itself supplies the statistical score.
- [AlgoTune](https://algotune.io/paper.pdf#page=4) pairs input generation,
  reference implementations, development feedback, and held-out evaluation of
  generated code. Its [timed-output check](https://github.com/oripress/AlgoTune/blob/dff9914c10800c7a031c9e8c3d4d1c8cd1b38906/AlgoTuner/utils/evaluator/solution_checks.py#L245)
  reinforces scoring the measured output. This benchmark uses statistical loss
  and per-regime Pareto comparisons instead of AlgoTune's scalar speedup score.
