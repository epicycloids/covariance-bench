# Implement a candidate estimator

Create a candidate package that estimates moments of latent price increments
from irregular observations. The deliverable is executable candidate source plus
a manifest, suitable for the benchmark runner. Improve statistical loss, runtime,
or their tradeoff relative to the available baselines; no single estimator must
win every regime. Read [`src/covbench/api.py`](../src/covbench/api.py) and
[`docs/benchmark-design.md`](../docs/benchmark-design.md) before implementing.

## Deliverable

Put the implementation in a new directory under `candidates/`. Include a
`candidate.json` such as:

```json
{
  "name": "my-estimator",
  "entrypoint": "module:create",
  "config": {},
  "api_version": 1,
  "modes": ["offline", "online"]
}
```

Use an importable module within that directory. Its `create(problem, config)`
factory returns an estimator implementing API v1. Keep tunable choices in the
manifest configuration, so the runner can identify and reproduce the frozen
candidate. Document any required dependencies and the statistical assumptions
behind the method. Use the repository's README for the current runner commands.

## Required behavior

- `fit(Observations)` fits from the current point table. The input arrays contain
  observation times, asset indices, and observed price proxy levels.
- `update(Events)` processes arrival-ordered edits. `UPSERT` replaces a point
  keyed by `(asset, time)`; `DELETE` removes that key and ignores its value.
  Handle edits to previously observed times, not only append operations.
- `predict(Query)` returns `Prediction(mean, covariance)` for the full latent
  increment `X(end) - X(start)`. Historical queries estimate marginal moments
  over the observed window; forecast queries use information available at
  `asof` to predict the later interval. Interpret windows in physical time.
- `synchronize()` completes outstanding device work. Implement a no-op for CPU
  code, or the appropriate completion operation for accelerator code.

Return a finite host NumPy mean array of shape `(n_assets,)` and either
`DenseMatrix` with shape `(n_assets, n_assets)`, or `FactorMatrix` with loadings
of shape `(n_assets, rank)` and nonnegative specific variances of shape
`(n_assets,)`. Factor form represents
`loadings @ loadings.T + diag(specific)`. Return the API classes themselves,
rather than an arbitrary matrix object with deferred computation.

The target excludes observation noise. It includes the latent increment mean,
so an estimated second moment is `covariance + outer(mean, mean)`. Do not
confuse second moments with centered covariance, interval covariance with a
per-unit-time covariance rate, or irregular price observations with equally
spaced returns. Dense indefinite outputs are retained and diagnosed by the
evaluator; if the method needs regularization, implement and account for it in
the candidate itself.

The only instance information is the public API input: `Problem`, the observed
table or edits, and queries. Infer noise, drift, rank, and other model choices
from permitted observations or use fixed documented assumptions. Do not inspect
evaluator truth, generator state, root seeds, held-out configuration, or future
observations; do not alter the harness, scores, or workloads to improve a result.
The local subprocess runner does not enforce a filesystem sandbox, so adherence
to this input contract is required of candidate implementations.

## Evaluate and report

First check both offline and online paths, including replacement and deletion of
old points, forecast queries, and the dense or factor output contract. Then run
the development workloads against the supplied baselines on the same instances.
Inspect import/create/fit/update/predict timings separately, statistical losses,
structural diagnostics, and failures. Query timing includes synchronization and
host output materialization, and the measured prediction is the one scored.
There is no excluded JIT warmup phase; compilation triggered during a stage is
charged there. Repetitions start fresh workers, so do not depend on caches from
earlier independent trials.

Use the per-regime frontier to identify where the method buys accuracy with
extra time or saves time at an acceptable loss. Report the regimes that improve,
the regimes that regress, and relevant failures or undefined metrics. Keep raw
results and identify the candidate source/config hashes, suite, and environment.
Do not turn a development result into a claim of general superiority.

When the implementation and config are selected, freeze them and request a
fresh held-out seed configuration from the evaluator-owned workflow. Evaluate
that unchanged artifact and report the resulting comparison. If held-out
feedback motivates further changes, treat it as development feedback and use a
new held-out configuration for the next validation claim.
