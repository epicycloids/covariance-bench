# Synthetic Scenarios and Data-Generating Process

This document specifies the probability model implemented in [`src/covbench/scenarios.py`](../src/covbench/scenarios.py) and the scenario suites in [`suites/`](../suites/).

Each scenario instance is generated from a parameter dictionary `Scenario.config` and a 64-bit integer seed. Conditional on the drawn regime parameters, latent increments are independent across steps of a master time grid. Simulation storage scales as $`O(NS)`$, and exact ground-truth increment moments for any query interval are represented in $`O(N(mK + 1))`$ space ($`m \le R`$) without allocating $`N \times N`$ matrices.

## 1. Master Time Grid and Piecewise-Constant Regimes

### 1.1 Time Grid and Regime Partition

Let $`N \ge 1`$ be the number of assets (`n_assets`), $`T > 0`$ be the time horizon (`horizon`, default $`1.0`$), and $`S \ge 16`$ be the number of grid steps (`grid_steps`, default $`256`$). The master time grid consists of $`S + 1`$ equally spaced timestamps:

```math
t_s = s \Delta t, \qquad \Delta t = \frac{T}{S}, \qquad s \in \{0, 1, \dots, S\}.
```

All observation timestamps and query window endpoints lie on $`\{t_0, \dots, t_S\}`$.

Given a regime count $`R \in \{1, \dots, S\}`$ (`regimes`, default $`1`$), the step index set $`\{0, \dots, S-1\}`$ is partitioned into $`R`$ contiguous intervals $`[s_{r-1}, s_r)`$ for $`r \in \{1, \dots, R\}`$ with integer boundary indices

```math
s_r = \left\lfloor \frac{r S}{R} \right\rfloor, \qquad 0 = s_0 < s_1 < \dots < s_R = S.
```

Regime $`r`$ governs the physical time interval $`[t_{s_{r-1}}, t_{s_r}]`$ of duration $`(s_r - s_{r-1})\Delta t`$.

### 1.2 Regime Parameter Distribution

For each regime $`r \in \{1, \dots, R\}`$, `_draw_regime` draws a per-unit-time drift vector $`\mu_r \in \mathbb{R}^N`$, a factor loading matrix $`B_r \in \mathbb{R}^{N \times K}`$ of factor rank $`K \ge 0`$ (`rank`), and a positive specific variance vector $`d_r \in \mathbb{R}_{>0}^N`$ independently across regimes.

**Marginal variance rates.** Given the diagonal spread parameter $`\kappa \ge 1`$ (`condition`, default $`10.0`$), draw independent log-uniform marginal variance rates for each asset $`i \in \{0, \dots, N-1\}`$:

```math
v_{r,i} = \exp(U_{r,i}), \qquad U_{r,i} \stackrel{\mathrm{i.i.d.}}{\sim} \mathrm{Uniform}\!\left(-\frac{1}{2}\ln\kappa, \, \frac{1}{2}\ln\kappa\right).
```

Thus $`v_{r,i} \in [\kappa^{-1/2}, \kappa^{1/2}]`$, and the ratio between the upper and lower support bounds of $`v_{r,i}`$ is $`\kappa`$.

**Idiosyncratic variance shares.** Draw independent specific shares for each asset $`i \in \{0, \dots, N-1\}`$:

```math
w_{r,i} \stackrel{\mathrm{i.i.d.}}{\sim} \mathrm{Uniform}(0.2, 0.8).
```

**Factor loadings and specific variances.** When $`K > 0`$, draw a Gaussian matrix $`G_r \in \mathbb{R}^{N \times K}`$ with i.i.d. $`\mathcal{N}(0, 1)`$ entries, normalize each row $`u_{r,i} = G_{r,i\cdot} / \|G_{r,i\cdot}\|_2`$, and set

```math
B_{r,i\cdot} = \sqrt{v_{r,i}(1 - w_{r,i})}\, u_{r,i}, \qquad d_{r,i} = v_{r,i} w_{r,i}.
```

When $`K = 0`$, set $`B_r \in \mathbb{R}^{N \times 0}`$ (empty loading matrix) and $`d_{r,i} = v_{r,i}`$. In both cases, the per-unit-time covariance rate matrix is

```math
\Sigma_r = B_r B_r^\top + \mathrm{diag}(d_r).
```

Because $`\|u_{r,i}\|_2 = 1`$, the $`i`$-th diagonal entry of $`\Sigma_r`$ is $`(\Sigma_r)_{ii} = v_{r,i}(1 - w_{r,i}) + v_{r,i} w_{r,i} = v_{r,i}`$, and because $`d_{r,i} \ge 0.2 \kappa^{-1/2} > 0`$, $`\Sigma_r`$ is strictly positive definite.

**Drift rates.** Given the drift scale $`\alpha_\mu \ge 0`$ (`drift_strength`, default $`0.2`$), draw

```math
\mu_{r,i} = \alpha_\mu \sqrt{\frac{v_{r,i}}{T}}\, Z_{r,i}, \qquad Z_{r,i} \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1).
```

Over the full horizon $`T`$, the drift displacement $`\mu_{r,i} T`$ has root-mean-square magnitude $`\alpha_\mu \sqrt{v_{r,i} T}`$, proportional to the asset's horizon standard deviation $`\sqrt{v_{r,i} T}`$.

## 2. Latent Increment Laws

The latent level process $`X(t_s) \in \mathbb{R}^N`$ is initialized at $`X(0) = 0`$ and evolves across grid steps $`s \in \{0, \dots, S-1\}`$ according to

```math
X(t_{s+1}) - X(t_s) = \mu_r \Delta t + \xi_s, \qquad s \in [s_{r-1}, s_r),
```

where the zero-mean stochastic innovations $`\xi_s \in \mathbb{R}^N`$ are conditionally independent across $`s`$ given $`\{(\mu_r, B_r, d_r)\}_{r=1}^R`$.

At step $`s \in [s_{r-1}, s_r)`$, draw independent standard normal vectors $`z_s \sim \mathcal{N}(0, I_K)`$ and $`\varepsilon_s \sim \mathcal{N}(0, I_N)`$, and define the unit-rate Gaussian shock

```math
\eta_s = B_r z_s + \mathrm{diag}(d_r)^{1/2} \varepsilon_s \sim \mathcal{N}(0, \Sigma_r).
```

Three innovation distributions (`distribution`) are supported, all sharing the exact conditional first and second moments $`\mathbb{E}[\xi_s] = 0`$ and $`\mathrm{Cov}(\xi_s) = \Sigma_r \Delta t`$.

### 2.1 Gaussian Increments (`gaussian`)

```math
\xi_s = \sqrt{\Delta t}\, \eta_s \sim \mathcal{N}(0, \Sigma_r \Delta t).
```

### 2.2 Multivariate Student-$`t`$ Increments (`student_t`)

Given degrees of freedom $`\nu > 8`$ (`df`, default $`10.0`$), draw an independent scalar chi-squared random variable $`W_s \sim \chi^2(\nu)`$ at step $`s`$ and set

```math
\xi_s = \sqrt{\Delta t} \sqrt{\frac{\nu - 2}{W_s}}\, \eta_s.
```

**Moment verification.** Because $`W_s`$ is independent of $`\eta_s`$ and $`\mathbb{E}[W_s^{-1}] = \frac{1}{\nu - 2}`$ for $`\nu > 2`$, we have $`\mathbb{E}[\xi_s] = 0`$ and

```math
\mathrm{Cov}(\xi_s) = \Delta t \, \mathbb{E}\!\left[\frac{\nu - 2}{W_s}\right] \mathrm{Cov}(\eta_s) = \Sigma_r \Delta t.
```

Because the radial multiplier $`\sqrt{(\nu - 2)/W_s}`$ is shared across all $`N`$ coordinates at step $`s`$, $`\xi_s`$ follows an elliptical multivariate Student-$`t`$ distribution with tail dependence and common volatility bursts even when $`K = 0`$. The requirement $`\nu > 8`$ guarantees finite moments up to order $`8`$, ensuring that fourth-moment sample covariance variances are finite.

### 2.3 Compound Poisson Jump-Diffusion Increments (`compound_poisson`)

Given jump covariance share $`\theta_J \in [0, 1]`$ (`jump_fraction`, default $`0.35`$) and Poisson jump intensity $`\lambda_J > 0`$ (`jump_rate`, default $`8.0`$), draw independent Poisson jump counts and standard normal jump amplitudes:

```math
N_{s,\mathrm{fac}} \sim \mathrm{Poisson}(\lambda_J \Delta t)^{\otimes K}, \qquad N_{s,\mathrm{spec}} \sim \mathrm{Poisson}(\lambda_J \Delta t)^{\otimes N}, \qquad \zeta_s \sim \mathcal{N}(0, I_K), \qquad \omega_s \sim \mathcal{N}(0, I_N),
```

and set

```math
\xi_s = \sqrt{(1 - \theta_J)\Delta t}\, \eta_s + \sqrt{\frac{\theta_J}{\lambda_J}} B_r \!\left(\sqrt{N_{s,\mathrm{fac}}} \odot \zeta_s\right) + \sqrt{\frac{\theta_J}{\lambda_J}} \mathrm{diag}(d_r)^{1/2} \!\left(\sqrt{N_{s,\mathrm{spec}}} \odot \omega_s\right),
```

where $`\odot`$ denotes elementwise multiplication.

**Moment verification.** Conditional on a count $`M \sim \mathrm{Poisson}(\lambda_J \Delta t)`$, the sum of $`M`$ independent $`\mathcal{N}(0, \theta_J / \lambda_J)`$ jumps has distribution $`\mathcal{N}(0, M \theta_J / \lambda_J) \stackrel{d}{=} \sqrt{\theta_J / \lambda_J}\sqrt{M}\, Z`$ with $`Z \sim \mathcal{N}(0, 1)`$. By the law of total covariance,

```math
\mathrm{Cov}\!\left(\sqrt{N_{s,\mathrm{fac}}} \odot \zeta_s\right) = \mathbb{E}[N_{s,\mathrm{fac},1}] I_K = (\lambda_J \Delta t) I_K, \qquad \mathrm{Cov}\!\left(\sqrt{N_{s,\mathrm{spec}}} \odot \omega_s\right) = (\lambda_J \Delta t) I_N.
```

Therefore, the diffusion term contributes $`(1 - \theta_J)\Sigma_r \Delta t`$ and the factor and idiosyncratic jump terms contribute $`\frac{\theta_J}{\lambda_J} (\lambda_J \Delta t) (B_r B_r^\top + \mathrm{diag}(d_r)) = \theta_J \Sigma_r \Delta t`$, yielding $`\mathrm{Cov}(\xi_s) = \Sigma_r \Delta t`$.

## 3. Exact Ground-Truth Moment Integration

Consider any query interval $`[t_{\mathrm{start}}, t_{\mathrm{end}}]`$ with endpoints on the master time grid ($`0 \le t_{\mathrm{start}} < t_{\mathrm{end}} \le T`$). For each regime $`r \in \{1, \dots, R\}`$, let

```math
\tau_r = \max\!\left(0, \, \min(t_{\mathrm{end}}, t_{s_r}) - \max(t_{\mathrm{start}}, t_{s_{r-1}})\right)
```

denote the duration of overlap between $`[t_{\mathrm{start}}, t_{\mathrm{end}}]`$ and regime $`r`$. Because step increments are conditionally independent across $`s`$, the exact mean and covariance of $`\Delta X(t_{\mathrm{start}}, t_{\mathrm{end}}) = X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$ are the sums of the step moments over the interval:

```math
\mu([t_{\mathrm{start}}, t_{\mathrm{end}}]) = \sum_{r=1}^R \tau_r \mu_r, \qquad \Sigma([t_{\mathrm{start}}, t_{\mathrm{end}}]) = \sum_{r=1}^R \tau_r \Sigma_r = \sum_{r=1}^R \tau_r B_r B_r^\top + \mathrm{diag}\!\left(\sum_{r=1}^R \tau_r d_r\right).
```

Let $`r_1 < r_2 < \dots < r_m`$ index the $`m \le R`$ regimes with $`\tau_{r_j} > 0`$. Defining the concatenated loading matrix and integrated specific variance vector

```math
B_{\mathrm{query}} = \begin{bmatrix} \sqrt{\tau_{r_1}} B_{r_1} & \sqrt{\tau_{r_2}} B_{r_2} & \cdots & \sqrt{\tau_{r_m}} B_{r_m} \end{bmatrix} \in \mathbb{R}^{N \times (mK)}, \qquad d_{\mathrm{query}} = \sum_{j=1}^m \tau_{r_j} d_{r_j} \in \mathbb{R}_{>0}^N,
```

we obtain the exact representation

```math
\Sigma([t_{\mathrm{start}}, t_{\mathrm{end}}]) = B_{\mathrm{query}} B_{\mathrm{query}}^\top + \mathrm{diag}(d_{\mathrm{query}}).
```

`_integrated` stores ground truth as `Prediction(mean, FactorMatrix(B_query, d_query))`. This representation uses $`O(N(mK + 1))`$ storage (including the length-$`N`$ `mean` and `d_query` vectors when $`K = 0`$). When a candidate also returns a `FactorMatrix` of rank $`K_p`$, exact Frobenius scoring takes $`O(N(K_p + mK + 1)^2)`$ time; when a candidate returns a `DenseMatrix`, the scorer streams over 128-row blocks in $`O(N)`$ temporary memory, reconstructing the 128-row truth slices from the rank-$`mK`$ factor truth in $`O(N^2(mK + 1))`$ time.

## 4. Observation Sampling, Noise, and Revisions

### 4.1 Observation Sampling Schedules

For each asset $`i \in \{0, \dots, N-1\}`$, `_sample_indices` selects a strictly increasing sequence of grid indices $`\mathcal{S}_i = (s_{i,0}, s_{i,1}, \dots, s_{i,m_i - 1}) \subseteq \{0, \dots, S\}`$ independently of the latent path $`X`$. The boundary indices $`0`$ and $`S`$ are always included in $`\mathcal{S}_i`$ (though arrival latency or deletion can make them unavailable at a given cutoff $`t_{\mathrm{asof}}`$). Interior indices $`s \in \{1, \dots, S-1\}`$ are sampled according to `sampling` and `observation_fraction` $`p_{\mathrm{obs}} \in [0, 1]`$ (default $`0.03`$ if `sampling == "sparse"`, else $`0.2`$):

- **`synchronous`**: each interior index $`s \in \{1, \dots, S-1\}`$ is included with probability $`p_{\mathrm{obs}}`$ via a single Bernoulli draw shared across all $`N`$ assets, so $`\mathcal{S}_0 = \mathcal{S}_1 = \dots = \mathcal{S}_{N-1}`$.
- **`asynchronous`** and **`sparse`**: for each asset $`i`$ independently, each interior index $`s`$ is included with probability $`p_{\mathrm{obs}}`$.
- **`imbalanced`**: given `imbalance` $`\gamma \ge 1`$ (default $`32.0`$), each asset $`i`$ draws a baseline inclusion rate $`p_i = p_{\mathrm{obs}} \exp(U_i)`$ with $`U_i \stackrel{\mathrm{i.i.d.}}{\sim} \mathrm{Uniform}(-\ln\gamma, 0)`$, so $`p_i \in [p_{\mathrm{obs}}/\gamma, p_{\mathrm{obs}}]`$, and includes each interior index $`s`$ independently with probability $`\min(p_i, 1)`$.
- **`bursty`**: for each asset $`i`$, two distinct burst centers $`c_{i,1}, c_{i,2} \in \{1, \dots, S-1\}`$ are drawn uniformly without replacement. With half-width $`h = \max(2, \lfloor S/12 \rfloor)`$, interior step $`s`$ is included with probability $`\min(1, 3 p_{\mathrm{obs}})`$ when $`\min(|s - c_{i,1}|, |s - c_{i,2}|) < h`$, and with probability $`\min(1, 0.1 p_{\mathrm{obs}})`$ otherwise.

### 4.2 Microstructure Noise Models

At each sampled grid index $`s = s_{i,k} \in \mathcal{S}_i`$, the noisy level observation is

```math
Y_{i,k} = X_i(t_{s_{i,k}}) + \varepsilon_{i,k}, \qquad k \in \{0, \dots, m_i - 1\}.
```

Given `noise_scale` $`\sigma_{\mathrm{noise}} \ge 0`$ (default $`0.03`$), asset $`i`$'s noise standard deviation is calibrated to its initial regime variance rate $`v_{1,i} = (\Sigma_1)_{ii}`$:

```math
\sigma_{\mathrm{obs}, i} = \sigma_{\mathrm{noise}} \sqrt{v_{1,i} T}.
```

Four noise specifications (`noise`) are supported:

- **`none`**: $`\varepsilon_{i,k} = 0`$.
- **`iid`**: $`\varepsilon_{i,k} \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_{\mathrm{obs}, i}^2)`$ independently across assets and observation times.
- **`correlated`**: given common noise variance share $`\rho_{\mathrm{noise}} \in [0, 1]`$ (`noise_correlation`, default $`0.7`$), draw a 2-factor common shock $`\psi_s \sim \mathcal{N}(0, I_2)`$ at every grid index $`s \in \{0, \dots, S\}`$ and a random unit direction $`a_i \in \mathbb{R}^2`$ ($`\|a_i\|_2 = 1`$) for each asset $`i`$. At $`s = s_{i,k}`$, $`\varepsilon_{i,k} = \sigma_{\mathrm{obs}, i} (\sqrt{\rho_{\mathrm{noise}}}\, (a_i^\top \psi_s) + \sqrt{1 - \rho_{\mathrm{noise}}}\, \nu_{i,k})`$ with $`\nu_{i,k} \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1)`$. Each $`\varepsilon_{i,k}`$ has marginal variance $`\sigma_{\mathrm{obs}, i}^2`$, and two assets $`i, j`$ observed at the same grid step $`s`$ have noise correlation $`\rho_{\mathrm{noise}} (a_i^\top a_j)`$.
- **`ar`**: given autoregressive coefficient $`\rho_{\mathrm{AR}} \in (-1, 1)`$ (`ar_rho`, default $`0.7`$), $`\varepsilon_{i,k}`$ follows a stationary $`\mathrm{AR}(1)`$ process in observation-event time $`k \in \{0, \dots, m_i - 1\}`$ (one step per sampled observation of asset $`i`$): $`\varepsilon_{i,0} = \zeta_{i,0}`$ and $`\varepsilon_{i,k} = \rho_{\mathrm{AR}} \varepsilon_{i,k-1} + \sqrt{1 - \rho_{\mathrm{AR}}^2}\, \zeta_{i,k}`$ for $`k \ge 1`$, where $`\zeta_{i,k} \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_{\mathrm{obs}, i}^2)`$.

### 4.3 Arrival Latency, Corrections, and Deletions

Each sampled point $`(i, t_{s_{i,k}}, Y_{i,k})`$ is converted into one or two arrival-ordered table events governed by `max_latency` $`L_{\max} \ge 0`$ (default $`0.0`$), `correction_fraction` $`p_{\mathrm{corr}} \in [0, 1]`$ (default $`0.0`$), and `delete_fraction` $`p_{\mathrm{del}} \in [0, 1]`$ (default $`0.0`$), where $`p_{\mathrm{corr}} + p_{\mathrm{del}} \le 1`$:

1. **Initial arrival time**: draw $`\ell_{i,k} \sim \mathrm{Uniform}(0, L_{\max})`$ and set $`a_{i,k} = t_{s_{i,k}} + \ell_{i,k}`$.
2. **Event generation**: draw $`u_{i,k} \sim \mathrm{Uniform}(0, 1)`$:
   - **Correction** ($`u_{i,k} < p_{\mathrm{corr}}`$): an initial corrupted observation $`\tilde{Y}_{i,k} = Y_{i,k} + 0.1 \sqrt{v_{1,i} T}\, Z_{i,k}`$ with $`Z_{i,k} \sim \mathcal{N}(0, 1)`$ is emitted as `(a_{i,k}, t_{s_{i,k}}, i, \tilde{Y}_{i,k}, UPSERT)`, followed after delay $`\delta_{i,k} \sim \mathrm{Uniform}(0.01 T, 0.08 T)`$ by the correcting event `(a_{i,k} + \delta_{i,k}, t_{s_{i,k}}, i, Y_{i,k}, UPSERT)`.
   - **Deletion** ($`p_{\mathrm{corr}} \le u_{i,k} < p_{\mathrm{corr}} + p_{\mathrm{del}}`$): the initial event `(a_{i,k}, t_{s_{i,k}}, i, Y_{i,k}, UPSERT)` is followed after delay $`\delta_{i,k} \sim \mathrm{Uniform}(0.01 T, 0.08 T)`$ by the retraction `(a_{i,k} + \delta_{i,k}, t_{s_{i,k}}, i, 0.0, DELETE)`.
   - **Unrevised point** ($`u_{i,k} \ge p_{\mathrm{corr}} + p_{\mathrm{del}}`$): a single event `(a_{i,k}, t_{s_{i,k}}, i, Y_{i,k}, UPSERT)` is emitted.

All emitted events across all assets are sorted in non-decreasing order of arrival timestamp $`a_m`$. Replaying events at cutoff $`t_{\mathrm{asof}}`$ includes all events with $`a_m \le t_{\mathrm{asof}}`$ (including arrival ties) and excludes all future measurements and future corrections with $`a_m > t_{\mathrm{asof}}`$.

## 5. Query Checkpoints and Scenario Parameters

### 5.1 Checkpoint Placement

With window width $`w = \max(2, \lfloor S/6 \rfloor)`$ grid steps, each dataset evaluates three checkpoint step indices:

```math
c_0 = \left\lfloor \frac{S}{3} \right\rfloor, \qquad c_1 = \left\lfloor \frac{3S}{5} \right\rfloor, \qquad c_2 = \left\lfloor \frac{4S}{5} \right\rfloor.
```

At each checkpoint $`c_q`$ ($`q \in \{0, 1, 2\}`$), with cutoff time $`t_{\mathrm{asof}} = t_{c_q}`$, two queries of physical duration $`w \Delta t`$ are generated:

- Historical query: $`[t_{\mathrm{start}}, t_{\mathrm{end}}] = [t_{c_q - w}, t_{c_q}]`$ with $`t_{\mathrm{asof}} = t_{c_q}`$.
- Forecast query: $`[t_{\mathrm{start}}, t_{\mathrm{end}}] = [t_{c_q}, t_{c_q + w}]`$ with $`t_{\mathrm{asof}} = t_{c_q}`$.

### 5.2 Scenario Configuration Reference

| Parameter | Type | Default | Constraint | Meaning |
| --- | --- | --- | --- | --- |
| `n_assets` | `int` | *(required)* | $`N \ge 1`$ | Number of assets $`N`$. |
| `grid_steps` | `int` | `256` | $`S \ge 16`$ | Number of master time grid steps $`S`$. |
| `rank` | `int` | `min(4, n_assets)` | $`K \ge 0`$ | Per-regime factor loading rank $`K`$. |
| `horizon` | `float` | `1.0` | $`T > 0`$ | Total physical time horizon $`T`$. |
| `condition` | `float` | `10.0` | $`\kappa \ge 1`$ | Support ratio $`\kappa`$ of regime marginal variance rates $`v_{r,i}`$. |
| `drift_strength` | `float` | `0.2` | $`\alpha_\mu \ge 0`$ | Dimensionless scale $`\alpha_\mu`$ of per-regime drift rates. |
| `regimes` | `int` | `1` | $`1 \le R \le S`$ | Number of piecewise-constant regimes $`R`$. |
| `distribution` | `str` | `"gaussian"` | `"gaussian"`, `"student_t"`, `"compound_poisson"` | Latent step innovation law. |
| `df` | `float` | `10.0` | $`\nu > 8`$ (for `student_t`) | Degrees of freedom $`\nu`$ for radial Student-$`t`$ shocks. |
| `jump_fraction` | `float` | `0.35` | $`\theta_J \in [0, 1]`$ | Fraction $`\theta_J`$ of covariance rate due to compound Poisson jumps. |
| `jump_rate` | `float` | `8.0` | $`\lambda_J > 0`$ | Jump arrival intensity $`\lambda_J`$ per unit time. |
| `sampling` | `str` | `"asynchronous"` | `"synchronous"`, `"asynchronous"`, `"sparse"`, `"imbalanced"`, `"bursty"` | Observation timestamp sampling scheme. |
| `observation_fraction` | `float` | `0.03` (`sparse`) or `0.2` | $`p_{\mathrm{obs}} \in [0, 1]`$ | Baseline grid-point inclusion probability $`p_{\mathrm{obs}}`$. |
| `imbalance` | `float` | `32.0` | $`\gamma \ge 1`$ | Maximum cross-asset sampling rate ratio $`\gamma`$ in `imbalanced` sampling. |
| `noise` | `str` | `"none"` | `"none"`, `"iid"`, `"correlated"`, `"ar"` | Observation microstructure noise process. |
| `noise_scale` | `float` | `0.03` | $`\sigma_{\mathrm{noise}} \ge 0`$ | Noise standard deviation relative to $`\sqrt{v_{1,i} T}`$. |
| `noise_correlation` | `float` | `0.7` | $`\rho_{\mathrm{noise}} \in [0, 1]`$ | Common 2-factor variance share in `correlated` noise. |
| `ar_rho` | `float` | `0.7` | $`\vert\rho_{\mathrm{AR}}\vert < 1`$ | Serial correlation coefficient $`\rho_{\mathrm{AR}}`$ in `ar` noise. |
| `max_latency` | `float` | `0.0` | $`L_{\max} \ge 0`$ | Maximum uniform arrival delay $`L_{\max}`$. |
| `correction_fraction` | `float` | `0.0` | $`p_{\mathrm{corr}} \in [0, 1]`$ | Probability $`p_{\mathrm{corr}}`$ of an initially corrupted then corrected point. |
| `delete_fraction` | `float` | `0.0` | $`p_{\mathrm{del}} \in [0, 1]`$, $`p_{\mathrm{corr}} + p_{\mathrm{del}} \le 1`$ | Probability $`p_{\mathrm{del}}`$ of a delayed point deletion. |

## 6. Benchmark Suites

### 6.1 Smoke Suite (`suites/smoke.json`)

| Scenario ID | $`N`$ | $`S`$ | $`K`$ | Distribution | Sampling ($`p_{\mathrm{obs}}`$) | Noise ($`\sigma_{\mathrm{noise}}`$) | $`R`$ | $`\kappa`$ | Edits ($`L_{\max}, p_{\mathrm{corr}}, p_{\mathrm{del}}`$) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gaussian-async-4` | 4 | 32 | 2 | `gaussian` | `asynchronous` (0.6) | `none` | 1 | 10 | $`0, 0, 0`$ |
| `student-edits-8` | 8 | 32 | 2 | `student_t` ($`\nu=10`$) | `bursty` (0.5) | `ar` (0.04) | 2 | 10 | $`0.06, 0.15, 0.05`$ |
| `jumps-correlated-8` | 8 | 32 | 2 | `compound_poisson` | `imbalanced` (0.6, $`\gamma=8`$) | `correlated` (0.03) | 2 | 100 | $`0, 0, 0`$ |

### 6.2 Core Suite (`suites/core.json`)

| Scenario ID | $`N`$ | $`S`$ | $`K`$ | Distribution | Sampling ($`p_{\mathrm{obs}}`$) | Noise ($`\sigma_{\mathrm{noise}}`$) | $`R`$ | $`\kappa`$ | $`\alpha_\mu`$ | Edits ($`L_{\max}, p_{\mathrm{corr}}, p_{\mathrm{del}}`$) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gaussian-clean-16` | 16 | 256 | 8 | `gaussian` | `synchronous` (0.7) | `none` | 1 | $`10`$ | 0.3 | $`0, 0, 0`$ |
| `student-noise-edits-16` | 16 | 256 | 0 | `student_t` ($`\nu=10`$) | `bursty` (0.3) | `iid` (0.04) | 2 | $`30`$ | 0.5 | $`0.025, 0.06, 0.025`$ |
| `jumps-change-condition-16` | 16 | 256 | 16 | `compound_poisson` | `asynchronous` (0.4) | `correlated` (0.03) | 2 | $`10^3`$ | 0.4 | $`0.03, 0.08, 0.03`$ |
| `gaussian-clean-64` | 64 | 384 | 16 | `gaussian` | `asynchronous` (0.35) | `none` | 1 | $`10`$ | 0.3 | $`0, 0, 0`$ |
| `student-noise-edits-64` | 64 | 384 | 8 | `student_t` ($`\nu=10`$) | `imbalanced` (0.4) | `ar` (0.04) | 2 | $`30`$ | 0.5 | $`0.025, 0.06, 0.025`$ |
| `jumps-change-condition-64` | 64 | 384 | 32 | `compound_poisson` | `bursty` (0.25) | `iid` (0.03) | 2 | $`10^4`$ | 1.0 | $`0.03, 0.08, 0.03`$ |
| `gaussian-clean-256` | 256 | 512 | 32 | `gaussian` | `sparse` (0.08) | `none` | 1 | $`10`$ | 0.3 | $`0, 0, 0`$ |
| `student-noise-edits-256` | 256 | 512 | 4 | `student_t` ($`\nu=10`$) | `asynchronous` (0.2) | `correlated` (0.04) | 2 | $`30`$ | 0.5 | $`0.025, 0.06, 0.025`$ |
| `jumps-change-condition-256` | 256 | 512 | 16 | `compound_poisson` | `imbalanced` (0.35) | `ar` (0.03) | 2 | $`10^5`$ | 2.0 | $`0.03, 0.08, 0.03`$ |
| `gaussian-clean-1024` | 1024 | 512 | 16 | `gaussian` | `imbalanced` (0.25) | `none` | 1 | $`10`$ | 0.3 | $`0, 0, 0`$ |
| `student-noise-edits-1024` | 1024 | 512 | 8 | `student_t` ($`\nu=10`$) | `sparse` (0.06) | `iid` (0.04) | 2 | $`30`$ | 0.5 | $`0.025, 0.06, 0.025`$ |
| `jumps-change-condition-1024` | 1024 | 512 | 32 | `compound_poisson` | `bursty` (0.12) | `correlated` (0.03) | 2 | $`10^6`$ | 3.0 | $`0.03, 0.08, 0.03`$ |

### 6.3 Scale Suite (`suites/scale.json`)

| Scenario ID | $`N`$ | $`S`$ | $`K`$ | Distribution | Sampling ($`p_{\mathrm{obs}}`$) | Noise ($`\sigma_{\mathrm{noise}}`$) | $`R`$ | $`\kappa`$ | Edits ($`L_{\max}, p_{\mathrm{corr}}, p_{\mathrm{del}}`$) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gaussian-async-4096` | 4096 | 256 | 8 | `gaussian` | `asynchronous` (0.15) | `iid` (0.02) | 1 | $`10`$ | $`0, 0, 0`$ |
| `student-imbalanced-4096` | 4096 | 384 | 32 | `student_t` ($`\nu=10`$) | `imbalanced` (0.25, $`\gamma=32`$) | `ar` (0.04) | 2 | $`10`$ | $`0.02, 0.04, 0.01`$ |
| `jumps-sparse-change-4096` | 4096 | 256 | 64 | `compound_poisson` | `bursty` (0.06) | `correlated` (0.03) | 2 | $`10^6`$ | $`0.03, 0.06, 0.02`$ |

### 6.4 Holdout Suite (`suites/holdout.json`)

| Scenario ID | $`N`$ | $`S`$ | $`K`$ | Distribution | Sampling ($`p_{\mathrm{obs}}`$) | Noise ($`\sigma_{\mathrm{noise}}`$) | $`R`$ | $`\kappa`$ | $`\alpha_\mu`$ | Edits ($`L_{\max}, p_{\mathrm{corr}}, p_{\mathrm{del}}`$) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `holdout-gaussian-change-32` | 32 | 288 | 6 | `gaussian` | `bursty` (0.19) | `ar` (0.045, $`\rho=0.45`$) | 3 | $`50`$ | 0.7 | $`0.04, 0.11, 0.04`$ |
| `holdout-student-96` | 96 | 360 | 12 | `student_t` ($`\nu=12`$) | `imbalanced` (0.32, $`\gamma=64`$) | `iid` (0.055) | 3 | $`80`$ | 1.2 | $`0.045, 0.09, 0.04`$ |
| `holdout-jumps-384` | 384 | 480 | 24 | `compound_poisson` ($`\theta_J=0.5, \lambda_J=12`$) | `sparse` (0.045) | `correlated` (0.05, $`\rho=0.4`$) | 3 | $`500`$ | 0.35 | $`0.06, 0.15, 0.05`$ |
| `holdout-gaussian-high-drift-768` | 768 | 384 | 12 | `gaussian` | `synchronous` (0.35) | `correlated` (0.02) | 3 | $`100`$ | 32.0 | $`0.02, 0.04, 0.015`$ |
| `holdout-student-bursts-384` | 384 | 320 | 0 | `student_t` ($`\nu=12`$) | `bursty` (0.08) | `ar` (0.06, $`\rho=-0.4`$) | 3 | $`300`$ | 16.0 | $`0.07, 0.18, 0.07`$ |
| `holdout-rare-jumps-32` | 32 | 192 | 20 | `compound_poisson` ($`\theta_J=0.7, \lambda_J=3`$) | `asynchronous` (0.27) | `iid` (0.015) | 3 | $`1000`$ | 0.1 | $`0.015, 0.20, 0.08`$ |

## 7. Modeling Assumptions and Limitations

1. **Discrete master grid for Student-$`t`$ increments**: in the `student_t` model, each grid step $`s`$ draws an independent radial multiplier $`\sqrt{(\nu - 2)/W_s}`$. First and second moments sum additively across steps, whereas the sum of independent Student-$`t`$ step increments on the fixed grid $`\{t_0, \dots, t_S\}`$ is not closed under convolution (unlike the Gaussian and compound Poisson families) and converges toward a Gaussian law as $`S \to \infty`$ for a fixed physical window.
2. **Regime-conditional increment moments**: ground truth $`(\mu([t_{\mathrm{start}}, t_{\mathrm{end}}]), \Sigma([t_{\mathrm{start}}, t_{\mathrm{end}}]))`$ is the moment pair of the latent increment $`X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$ conditional on the drawn regime parameters $`\{(\mu_r, B_r, d_r)\}_{r=1}^R`$; it is determined by integrating the regime rates over the query interval and is invariant to the realized sample path $`X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$ and noisy observations $`\{Y_{i,k}\}`$.
3. **Diagonal variance spread vs. eigenvalue condition number**: the parameter $`\kappa`$ (`condition`) sets the support ratio of the marginal variance rates $`v_{r,i} \in [\kappa^{-1/2}, \kappa^{1/2}]`$. Because $`B_r B_r^\top`$ contributes positive semidefinite off-diagonal structure and $`d_{r,i} = w_{r,i} v_{r,i}`$ with $`w_{r,i} \in [0.2, 0.8]`$, the spectral condition number $`\lambda_{\max}(\Sigma_r) / \lambda_{\min}(\Sigma_r)`$ generally differs from $`\kappa`$.
4. **Independent regime draws across forecast boundaries**: each regime $`r \in \{1, \dots, R\}`$ draws $`(\mu_r, B_r, d_r)`$ independently inside `_draw_regime`.
   - When $`R = 2`$ (as in `smoke`, `core`, and `scale`), the single regime boundary lies at $`s_1 = \lfloor S/2 \rfloor`$, whereas checkpoints lie at $`\lfloor S/3 \rfloor`$, $`\lfloor 3S/5 \rfloor`$, and $`\lfloor 4S/5 \rfloor`$ with width $`w = \lfloor S/6 \rfloor`$. Because $`\lfloor S/3 \rfloor + \lfloor S/6 \rfloor \le \lfloor S/2 \rfloor < \lfloor 3S/5 \rfloor`$, forecast windows do not cross into an unentered regime; only the checkpoint-1 historical window $`[t_{\lfloor 3S/5 \rfloor - w}, t_{\lfloor 3S/5 \rfloor}]`$ straddles $`s_1`$.
   - When $`R = 3`$ (as in `holdout.json`), regime boundaries lie at $`s_1 = \lfloor S/3 \rfloor`$ and $`s_2 = \lfloor 2S/3 \rfloor`$. At checkpoint $`0`$ ($`c_0 = \lfloor S/3 \rfloor = s_1`$), the forecast window $`[t_{s_1}, t_{s_1 + w}]`$ lies entirely in Regime 2 before any increments of Regime 2 have occurred; at checkpoint $`1`$ ($`c_1 = \lfloor 3S/5 \rfloor`$), the forecast window $`[t_{c_1}, t_{c_1 + w}]`$ crosses $`s_2 = \lfloor 2S/3 \rfloor`$ into Regime 3. Because the new regime's factor directions, variance rates, and drift vector are drawn independently of prior regimes, only the hyperparameter scale is shared across those forecast regime boundaries.
