# Native asynchronous covariance refit baseline

Requires the optional `async_covariance` library in the benchmark environment;
this candidate does not install it. Maintains the current observation table,
including replacements and deletions. Every prediction performs a native batch
`fit(method="overlap", threads=1)` on points inside the historical query window
or the forecast `lookback` ending at `asof`. No future observations are used.
This is a **refit baseline**: online updates maintain the table, and the native
scan and any factor fitting are charged to prediction. It does not benchmark the
library's local incremental `OnlineCovariance` update algorithm.

Drift is estimated independently per asset from its first and last retained
points. Fewer than two points give zero drift. Observation noise variance is
assumed zero, a deliberate statistical limitation rather than supplied truth.
The library estimates covariance rates; output covariance is multiplied by the
query duration, and output mean is estimated drift times duration.

For at most 256 assets, the baseline returns raw dense pair estimates, explicitly
filling missing pairs with zero and flooring the diagonal at `floor`. It does not
project an indefinite dense estimate. For larger universes, a deterministic ring
graph with default degree eight selects pairs; `MomentSnapshot.factor` fits a
model of rank at most eight with a positive marginal variance floor and the
configured specific-variance fraction. Missing marginal variances use `floor`.
Those graph, rank, completion, and flooring choices are baseline policies, not
claims that the latent covariance has this structure. Factor fitting uses the
fixed candidate seed, which is unrelated to generator seeds.

Native fit or factor errors propagate as candidate failures. No blanket exception
handler replaces a failed fit with a different estimator. Returned arrays are
materialized host NumPy arrays; CPU synchronization is a no-op.
