# Synthetic scenarios

Each seed draws new drift rates, factor directions, and positive specific
variance rates. These parameters are fixed within each piecewise regime.
Conditional on them, latent increments are independent across master-grid
steps. Simulation storage grows with assets times grid steps, rather than
assets squared. All observations and query endpoints lie on that grid.

For regime coefficients `mu`, `B`, and `d`, an increment of duration `dt` has
exact mean `mu*dt` and covariance `(B B.T + diag(d))*dt`. Gaussian cases use
factor and specific normal shocks. Student-t cases multiply both by the same
independent radial shock `sqrt((df-2)/chi2(df))` at each step. This preserves
covariance while introducing tail dependence, including when `rank=0` gives
diagonal covariance. The core uses df10; holdout cases also use df12. Requiring
`df>8` gives finite eighth moments. This is a process specified on a fixed grid,
not a claim about a continuous-time Student-t process under grid refinement.

Compound-Poisson cases allocate `jump_fraction` of covariance to zero-mean
normal jumps and the remainder to Gaussian diffusion. Independent Poisson
counts with intensity `jump_rate` govern each common factor and each specific
component. A conditional sum of jumps is normal with variance equal to its
count. Jump amplitudes compensate for intensity, so the exact covariance rate
remains `B B.T + diag(d)`.

Queries request moments of the full latent increment `X(end)-X(start)`.
Integrating regime drift over the physical window gives the mean. Integrating
regime covariance rates gives covariance; factor blocks scaled by square roots
of overlap durations represent the result without allocating a dense matrix.
Second moment is covariance plus the outer product of the integrated mean.
Historical queries target marginal integrated moments, rather than a posterior
latent path conditional on observations. Forecast queries target the following
window. Three checkpoints each issue one historical and one forecast query.
Observation noise is excluded from every target.

Sampling selects grid subsets independently of latent returns. Synchronous
cases share a subset; asynchronous cases sample independently by asset.
Sparse cases use low inclusion rates. Imbalanced cases randomize asset rates
over a multiplicative range. Bursty cases concentrate sampling near two random
centers per asset. Initial and terminal points are selected, but arrival delays
and deletions can make them unavailable at a query cutoff.

Point-noise standard deviation is
`noise_scale*sqrt(initial_variance_rate*horizon)`. Iid noise is independent
normal noise. Correlated noise combines randomized two-factor common shocks
at each grid time with independent noise; `noise_correlation` is the common
variance fraction, rather than every pair's correlation. AR noise is stationary
and evolves by selected observation index, not elapsed physical time.

An initial point arrives at its measurement time plus uniform latency.
Corrections first publish an incorrect value, then restore its actual noisy
observation after a positive delay. Deletions retract selected points after
their initial arrival. These selections are exogenous. Replay uses only events
arriving by the cutoff, including all arrival ties. Future corrections and
future measurements never appear in candidate inputs.

`condition` controls randomized diagonal variance spread; it is not a certified
spectral condition number. Positive specific rates make truth positive definite.
Suite loading ranks vary, including diagonal and broad factor cases. Candidate
inputs contain observations/events, query windows, and problem dimensions;
regime coefficients and truth remain evaluator data.

Tests check exact integration across regime boundaries, replay causality,
correction identity, seed reproducibility, and invariance of truth to noise and
edits. Small Monte Carlo checks standardize latent increments using each draw's
own truth and compare their empirical mean/covariance with zero/identity.

`holdout.json` is a public held-out configuration set, not secret data. Freeze
candidate code and selection choices before evaluating it. Fresh seeds
randomize its truth; the public file alone does not enforce an untouched holdout.
