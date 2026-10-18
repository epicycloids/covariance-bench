# Loss Metrics, Numerical Linear Algebra, and Pareto Frontiers

This document specifies the loss functions, numerical linear algebra routines, structural diagnostics, and Pareto dominance relations implemented in [`src/covbench/metrics.py`](../src/covbench/metrics.py) and [`src/covbench/pareto.py`](../src/covbench/pareto.py).

## 1. Target Quantities and Input Validation

For a query over an $`N`$-asset universe, let $`(\mu, \Sigma)`$ denote the ground-truth mean vector $`\mu \in \mathbb{R}^N`$ and covariance matrix $`\Sigma \in \mathbb{R}^{N \times N}`$ of the latent increment $`\Delta X = X(t_{\mathrm{end}}) - X(t_{\mathrm{start}})`$, and let $`(\hat{\mu}, \hat{\Sigma})`$ denote the candidate estimator's predicted mean and covariance. The true and predicted second-moment matrices are

```math
M = \Sigma + \mu \mu^\top, \qquad \hat{M} = \hat{\Sigma} + \hat{\mu} \hat{\mu}^\top.
```

All reported metrics are losses to be minimized.

### 1.1 Validation Rules

Before evaluating any metric, `_validate` verifies that both `prediction` and `truth` are instances of `Prediction` with matching dimension $`N \ge 1`$ and finite real host NumPy arrays (floating-point, signed integer, or unsigned integer; `dtype.kind in "fiu"`), which are converted to `float64`:

**`DenseMatrix(value)`**: `value` must have shape $`(N, N)`$, contain only finite entries, and satisfy the relative symmetry bound

```math
\max_{0 \le i, j < N} |\hat{\Sigma}_{ij} - \hat{\Sigma}_{ji}| \le 64 \varepsilon_{\mathrm{mach}} \, s_{\max}, \qquad s_{\max} = \max_{0 \le u, v < N} |\hat{\Sigma}_{uv}|,
```

evaluated in 128-row chunks (`_maxabs`).

**`FactorMatrix(loadings, specific)`**: `loadings` $`B \in \mathbb{R}^{N \times K}`$ ($`K \ge 0`$) and `specific` $`d \in \mathbb{R}^N`$ must have shapes $`(N, K)`$ and $`(N,)`$, contain only finite entries, satisfy $`d_i \ge 0`$ for all $`i`$, and have a finite implied diagonal $`\hat{\Sigma}_{ii} = \|B_{i\cdot}\|_2^2 + d_i`$. Because $`\hat{\Sigma} = B B^\top + \mathrm{diag}(d) \succeq 0`$, the maximum entry magnitude is bounded by $`s_{\max} = \max_{0 \le i < N} \hat{\Sigma}_{ii}`$ via the Cauchy–Schwarz inequality $`|\hat{\Sigma}_{ij}| \le \sqrt{\hat{\Sigma}_{ii}\hat{\Sigma}_{jj}}`$.

If `prediction` fails validation (or has dimension different from `truth`), `score` returns `status = "invalid_prediction"` with all metrics set to `None`.

## 2. Loss Function Definitions

| Metric Name | Mathematical Definition | Domain Requirements |
| --- | --- | --- |
| `covariance_nmse` | $`\dfrac{\Vert\hat{\Sigma} - \Sigma\Vert_F^2}{\Vert\Sigma\Vert_F^2}`$ | $`\Vert\Sigma\Vert_F > 0`$ (or $`0`$ if $`\hat{\Sigma} = \Sigma = 0`$). |
| `second_moment_nmse` | $`\dfrac{\Vert\hat{M} - M\Vert_F^2}{\Vert M\Vert_F^2}`$ | $`\Vert M\Vert_F > 0`$ (or $`0`$ if $`\hat{M} = M = 0`$). |
| `diagonal_nmse` | $`\dfrac{\sum_{i=0}^{N-1} (\hat{\Sigma}_{ii} - \Sigma_{ii})^2}{\sum_{i=0}^{N-1} \Sigma_{ii}^2}`$ | $`\sum_i \Sigma_{ii}^2 > 0`$ (or $`0`$ if $`\mathrm{diag}(\hat{\Sigma}) = \mathrm{diag}(\Sigma) = 0`$). |
| `mean_variance_scaled_mse` | $`\dfrac{1}{N} \sum_{i=0}^{N-1} \dfrac{(\hat{\mu}_i - \mu_i)^2}{\Sigma_{ii}}`$ | $`\Sigma_{ii} \ge 0`$, with $`\hat{\mu}_i = \mu_i`$ whenever $`\Sigma_{ii} = 0`$. |
| `mean_mahalanobis` | $`\dfrac{1}{N} (\hat{\mu} - \mu)^\top \Sigma^{-1} (\hat{\mu} - \mu)`$ (full or block average) | $`\Sigma \succ 0`$ on the evaluated scope. |
| `correlation_mse` | $`\dfrac{1}{\vert\mathcal{P}\vert} \sum_{(i,j) \in \mathcal{P}} (\hat{C}_{ij} - C_{ij})^2`$ | $`N \ge 2`$, $`\hat{\Sigma}_{ii} > 0`$, and $`\Sigma_{ii} > 0`$ for all $`i`$. |
| `portfolio_risk_relative_mse` | $`\dfrac{1}{Q} \sum_{q=1}^Q \left(\dfrac{w_q^\top \hat{\Sigma} w_q}{w_q^\top \Sigma w_q} - 1\right)^2`$ | $`w_q^\top \Sigma w_q \ge 0`$, with $`w_q^\top \hat{\Sigma} w_q = 0`$ if $`w_q^\top \Sigma w_q = 0`$. |
| `portfolio_second_moment_relative_mse` | $`\dfrac{1}{Q} \sum_{q=1}^Q \left(\dfrac{w_q^\top \hat{M} w_q}{w_q^\top M w_q} - 1\right)^2`$ | $`w_q^\top M w_q \ge 0`$, with $`w_q^\top \hat{M} w_q = 0`$ if $`w_q^\top M w_q = 0`$. |
| `gaussian_kl` | $`\dfrac{1}{\sum_{b=1}^m \vert\mathcal{B}_b\vert} \sum_{b=1}^m D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu_b, \Sigma_b) \parallel \mathcal{N}(\hat{\mu}_b, \hat{\Sigma}_b)\right)`$ | Neither matrix diagnosed as full-matrix `"indefinite"`, and $`\hat{\Sigma}_b \succ 0, \Sigma_b \succ 0`$ on all declared blocks $`\mathcal{B}_b`$. |
| `covariance_qlike` | $`\dfrac{1}{\sum_{b=1}^m \vert\mathcal{B}_b\vert} \sum_{b=1}^m \left(\mathrm{tr}(\hat{\Sigma}_b^{-1} \Sigma_b) + \ln\det\hat{\Sigma}_b - \ln\det\Sigma_b - \vert\mathcal{B}_b\vert\right)`$ | Neither matrix diagnosed as full-matrix `"indefinite"`, and $`\hat{\Sigma}_b \succ 0, \Sigma_b \succ 0`$ on all declared blocks $`\mathcal{B}_b`$. |

### 2.1 Normalized Frobenius Losses

`covariance_nmse`, `second_moment_nmse`, and `diagonal_nmse` measure relative squared Euclidean error on the covariance matrix, the uncentered second-moment matrix $`M = \Sigma + \mu\mu^\top`$, and the marginal variance vector $`\mathrm{diag}(\Sigma)`$, respectively.

When the true matrix norm in the denominator is zero, `_normalized` returns $`0.0`$ if the predicted matrix is also identically zero, and `None` (with an error message) otherwise.

### 2.2 Mean Estimation Losses

**`mean_variance_scaled_mse`** weights each coordinate's squared mean error $`(\hat{\mu}_i - \mu_i)^2`$ by the inverse true marginal variance $`1 / \Sigma_{ii}`$, making the loss invariant to coordinate-wise rescaling and well-defined when the true drift $`\mu`$ is zero.

**`mean_mahalanobis`** weights the mean error vector $`\delta_\mu = \hat{\mu} - \mu`$ by the true precision matrix $`\Sigma^{-1}`$. When `truth.covariance` is a `FactorMatrix` with strictly positive specific variances $`d > 0`$ (which holds for all benchmark-generated truth), `_mean_mahalanobis` evaluates the exact full-dimension quadratic form $`\frac{1}{N} \delta_\mu^\top \Sigma^{-1} \delta_\mu`$ using the Woodbury identity (Section 3.3). When `truth.covariance` is a `DenseMatrix` (or has zero specific variances), it evaluates the full Cholesky solve if $`N \le N_{\mathrm{exact}}`$ (`exact_dense_limit`, default $`512`$), and otherwise averages the quadratic form over the declared sub-blocks $`\mathcal{B}_1, \dots, \mathcal{B}_m`$:

```math
\frac{1}{\sum_{b=1}^m |\mathcal{B}_b|} \sum_{b=1}^m (\hat{\mu}_{\mathcal{B}_b} - \mu_{\mathcal{B}_b})^\top \Sigma_{\mathcal{B}_b, \mathcal{B}_b}^{-1} (\hat{\mu}_{\mathcal{B}_b} - \mu_{\mathcal{B}_b}).
```

The evaluation scope (`"full"` or `"declared_block_average"`) is recorded in `metric_details["mean_mahalanobis"]`.

### 2.3 Correlation Loss

Let $`\mathcal{P} = \{(i_1, j_1), \dots, (i_P, j_P)\}`$ ($`0 \le i_p < j_p < N`$) be the evaluated set of off-diagonal asset pairs. If $`\binom{N}{2} \le P_{\max}`$ (`pair_samples`, default $`2048`$), $`\mathcal{P}`$ contains all $`\binom{N}{2}`$ upper-triangular pairs; otherwise $`\mathcal{P}`$ is a seeded uniform sample of $`P_{\max}`$ distinct pairs drawn without replacement (`_pairs`).

`correlation_mse` is defined whenever $`N \ge 2`$ and all predicted and true marginal variances are strictly positive ($`\hat{\Sigma}_{ii} > 0`$ and $`\Sigma_{ii} > 0`$ for all $`i \in \{0, \dots, N-1\}`$), including for indefinite dense matrices with positive diagonals. The predicted and true pairwise correlations

```math
\hat{C}_{ij} = \frac{\hat{\Sigma}_{ij}}{\sqrt{\hat{\Sigma}_{ii}} \sqrt{\hat{\Sigma}_{jj}}}, \qquad C_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}} \sqrt{\Sigma_{jj}}}
```

are computed by dividing sequentially by $`\min(\sqrt{\Sigma_{ii}}, \sqrt{\Sigma_{jj}})`$ and $`\max(\sqrt{\Sigma_{ii}}, \sqrt{\Sigma_{jj}})`$ to avoid intermediate underflow or overflow, and `correlation_mse` is their mean squared difference over $`\mathcal{P}`$.

### 2.4 Portfolio Quadratic-Form Losses

Given `portfolio_samples` $`Q \ge 1`$ (default $`64`$), the evaluator constructs $`Q`$ unit-$`\ell_2`$-norm portfolio weight vectors $`w_1, \dots, w_Q \in \mathbb{R}^N`$ ($`\|w_q\|_2 = 1`$):

- $`w_1 = \frac{1}{\sqrt{N}} \mathbf{1}_N`$ is the equal-weight portfolio.
- $`w_2, \dots, w_Q`$ are independent standard normal draws $`g_q \sim \mathcal{N}(0, I_N)`$ normalized as $`w_q = g_q / \|g_q\|_2`$.

`portfolio_risk_relative_mse` and `portfolio_second_moment_relative_mse` compute the mean squared relative error of the portfolio variance $`w_q^\top \hat{\Sigma} w_q`$ and portfolio second moment $`w_q^\top \hat{M} w_q`$ across $`q \in \{1, \dots, Q\}`$.

### 2.5 Block Gaussian KL and QLIKE Losses

Given `block_size` $`S_{\mathrm{block}} \ge 1`$ (default $`32`$) and `block_count` $`m_{\mathrm{block}} \ge 1`$ (default $`8`$), `_blocks` takes the first $`\min(N, S_{\mathrm{block}} m_{\mathrm{block}})`$ indices of a seeded random permutation of $`\{0, \dots, N-1\}`$ and partitions them into $`m`$ disjoint sorted subsets $`\mathcal{B}_1, \dots, \mathcal{B}_m`$ of size $`k_b = |\mathcal{B}_b| \le S_{\mathrm{block}}`$.

For each block $`\mathcal{B}_b`$, let $`\hat{\mu}_b, \mu_b \in \mathbb{R}^{k_b}`$ and $`\hat{\Sigma}_b, \Sigma_b \in \mathbb{R}^{k_b \times k_b}`$ denote the subvectors and principal submatrices on $`\mathcal{B}_b`$. When $`\hat{\Sigma}_b \succ 0`$ and $`\Sigma_b \succ 0`$, define the block QLIKE covariance divergence

```math
\ell_{\mathrm{QLIKE}}(\hat{\Sigma}_b, \Sigma_b) = \mathrm{tr}(\hat{\Sigma}_b^{-1} \Sigma_b) + \ln\det(\hat{\Sigma}_b) - \ln\det(\Sigma_b) - k_b \ge 0
```

and the block Gaussian Kullback–Leibler divergence from $`\mathcal{N}(\mu_b, \Sigma_b)`$ to $`\mathcal{N}(\hat{\mu}_b, \hat{\Sigma}_b)`$:

```math
D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu_b, \Sigma_b) \parallel \mathcal{N}(\hat{\mu}_b, \hat{\Sigma}_b)\right) = \frac{1}{2}\left(\ell_{\mathrm{QLIKE}}(\hat{\Sigma}_b, \Sigma_b) + (\hat{\mu}_b - \mu_b)^\top \hat{\Sigma}_b^{-1} (\hat{\mu}_b - \mu_b)\right) \ge 0.
```

Both metrics are normalized per asset across the union of the evaluated blocks ($`K_{\mathrm{total}} = \sum_{b=1}^m k_b`$):

```math
\mathrm{covariance\_qlike} = \frac{1}{K_{\mathrm{total}}} \sum_{b=1}^m \ell_{\mathrm{QLIKE}}(\hat{\Sigma}_b, \Sigma_b), \qquad \mathrm{gaussian\_kl} = \frac{1}{K_{\mathrm{total}}} \sum_{b=1}^m D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu_b, \Sigma_b) \parallel \mathcal{N}(\hat{\mu}_b, \hat{\Sigma}_b)\right).
```

**Domain requirements.** `gaussian_kl` and `covariance_qlike` are evaluated when full-matrix diagnosis (`_diagnose`, run on `DenseMatrix` inputs with $`N \le N_{\mathrm{exact}}`$) does not classify either $`\hat{\Sigma}`$ or $`\Sigma`$ as `"indefinite"` (minimum eigenvalue below $`-\tau`$) and Cholesky factorization (`cho_factor`) succeeds on every declared principal block $`\mathcal{B}_1, \dots, \mathcal{B}_m`$. A globally rank-deficient positive semidefinite matrix is therefore eligible when all declared proper blocks are strictly positive definite. If full-matrix diagnosis is `"indefinite"` or Cholesky factorization fails on any declared block $`\mathcal{B}_b`$, both metrics are set to `None` and the failed block indices are recorded in `metric_details["gaussian_scores"]["failed_blocks"]`.

**Quasi-likelihood interpretation.** Under `student_t` or `compound_poisson` scenarios, `gaussian_kl` and `covariance_qlike` evaluate the Gaussian divergence between moment-matched normal surrogates $`\mathcal{N}(\mu_b, \Sigma_b)`$ and $`\mathcal{N}(\hat{\mu}_b, \hat{\Sigma}_b)`$ on the marginal blocks $`\mathcal{B}_b`$.

## 3. Numerical Linear Algebra and High-Dimensional Scalability

### 3.1 Unit Rescaling

To prevent floating-point overflow or underflow when squaring covariance entries (for example, at extreme scales such as $`10^{\pm 120}`$), `score` defines two positive scalar units:

```math
u_\Sigma = \sqrt{\max(\|\hat{\Sigma}\|_{\max}, \, \|\Sigma\|_{\max})} \quad (\text{or } 1 \text{ if } 0), \qquad u_M = \max(u_\Sigma, \, \|\hat{\mu}\|_\infty, \, \|\mu\|_\infty).
```

`_ScaledMatrix(matrix, unit, mean)` represents $`\tilde{\Sigma} = u_\Sigma^{-2} \Sigma`$ for covariance calculations and $`\tilde{M} = u_M^{-2} (\Sigma + \mu\mu^\top)`$ for second-moment calculations. When `matrix` is a `FactorMatrix(B, d)` and `mean` $`\mu`$ is provided, `_ScaledMatrix` appends $`u_M^{-1} \mu`$ as an extra column of the scaled loadings $`u_M^{-1} B`$:

```math
\tilde{M} = \begin{bmatrix} u_M^{-1} B & u_M^{-1} \mu \end{bmatrix} \begin{bmatrix} u_M^{-1} B & u_M^{-1} \mu \end{bmatrix}^\top + \mathrm{diag}(u_M^{-2} d),
```

reducing second-moment factor calculations to the same low-rank algebra as covariance calculations with rank $`K + 1`$.

### 3.2 Exact $`O(N(K_p + K_t + 1)^2)`$ Factor Frobenius Algebra

When both `left` and `right` are in factor form, $`\hat{\Sigma} = A A^\top + \mathrm{diag}(d_A)`$ ($`A \in \mathbb{R}^{N \times K_p}`$) and $`\Sigma = B B^\top + \mathrm{diag}(d_B)`$ ($`B \in \mathbb{R}^{N \times K_t}`$), `_norm_sq` and `_distance_sq` evaluate the exact squared Frobenius norms in $`O(N(K_p + K_t + 1)^2)`$ time (including the $`O(N)`$ vector work on `specific` and `mean` when $`K_p = K_t = 0`$), avoiding $`O(N^2)`$ materialization of $`N \times N`$ matrices.

**Single factor matrix norm (`_norm_sq`).** For $`\Sigma = B B^\top + \mathrm{diag}(d_B)`$, expanding the trace gives

```math
\|\Sigma\|_F^2 = \mathrm{tr}\!\left((B B^\top + \mathrm{diag}(d_B))^2\right) = \|B^\top B\|_F^2 + 2 d_B^\top \mathrm{diag}(B B^\top) + \|d_B\|_2^2,
```

where $`B^\top B \in \mathbb{R}^{K_t \times K_t}`$ and $`\mathrm{diag}(B B^\top)_i = \|B_{i\cdot}\|_2^2`$.

**Signed low-rank QR distance (`_distance_sq`).** To compute $`\|\Delta\|_F^2`$ for $`\Delta = (A A^\top - B B^\top) + \mathrm{diag}(d_A - d_B)`$, `_distance_sq` avoids the catastrophic cancellation of the Gram expansion $`\|A^\top A\|_F^2 + \|B^\top B\|_F^2 - 2\|A^\top B\|_F^2`$ when $`A A^\top \approx B B^\top`$ by setting $`\delta_d = d_A - d_B`$ and factoring the signed low-rank difference $`\Delta_{\mathrm{low}} = A A^\top - B B^\top = J S J^\top`$, where

```math
J = \begin{bmatrix} A & B \end{bmatrix} \in \mathbb{R}^{N \times (K_p + K_t)}, \qquad S = \mathrm{diag}(I_{K_p}, \, -I_{K_t}).
```

Compute the thin QR factorization $`J = Q R`$, where $`Q \in \mathbb{R}^{N \times r}`$ has orthonormal columns ($`Q^\top Q = I_r`$, $`r \le K_p + K_t`$) and $`R \in \mathbb{R}^{r \times (K_p + K_t)}`$ is upper trapezoidal. By orthogonal invariance of the Frobenius norm,

```math
\|\Delta_{\mathrm{low}}\|_F^2 = \|Q (R S R^\top) Q^\top\|_F^2 = \|R S R^\top\|_F^2,
```

where $`C = R S R^\top \in \mathbb{R}^{r \times r}`$ cancels matched subspaces prior to squaring.

- **Sparse diagonal difference** ($`|\{i : \delta_{d,i} \ne 0\}| \le 32`$, which includes $`\delta_d = 0`$): the nonzero entries of $`\delta_d`$ are appended directly to $`J`$ as scaled canonical unit vectors $`\sqrt{|\delta_{d,i}|}\, e_i`$ with corresponding signs $`\mathrm{sgn}(\delta_{d,i})`$ in $`S`$. Then $`\Delta = J S J^\top`$ and $`\|\Delta\|_F^2 = \|R S R^\top\|_F^2`$ directly, avoiding separate diagonal subtraction when $`A`$ and $`B`$ are orthogonal rotations or column splittings of the same covariance.
- **General diagonal difference** ($`|\{i : \delta_{d,i} \ne 0\}| > 32`$): let $`\delta_{\mathrm{low}, i} = \|A_{i\cdot}\|_2^2 - \|B_{i\cdot}\|_2^2`$ be the diagonal of $`\Delta_{\mathrm{low}}`$. Separating off-diagonal and diagonal entries gives $`\|\Delta\|_F^2 = (\|R S R^\top\|_F^2 - \|\delta_{\mathrm{low}}\|_2^2) + \|\delta_{\mathrm{low}} + \delta_d\|_2^2`$. If floating-point roundoff makes the off-diagonal difference slightly negative within $`-256 \varepsilon_{\mathrm{mach}} \max(\|R S R^\top\|_F^2, \|\delta_{\mathrm{low}}\|_2^2, \sum_i (\|A_{i\cdot}\|_2^2 + \|B_{i\cdot}\|_2^2)^2)`$, it is clamped to $`0.0`$ and flagged via `roundoff_clamped = True`.

When either matrix is a `DenseMatrix`, `_distance_sq` and `_norm_sq` stream over rows in contiguous blocks of 128 rows (`method = "streamed_dense"`), bounding temporary workspace to $`128 N`$ floats (with $`O(N^2(K + 1))`$ work when reconstructing 128-row blocks from a rank-$`K`$ `FactorMatrix`).

### 3.3 Woodbury Sum-of-Squares Identity for Mahalanobis Error

When `truth.covariance` is a `FactorMatrix(B, d)` with $`d > 0`$, we must evaluate $`\frac{1}{N} (\hat{\mu} - \mu)^\top \Sigma^{-1} (\hat{\mu} - \mu)`$ where $`\Sigma = B B^\top + \mathrm{diag}(d)`$.

Let $`u = \sqrt{\max_i \Sigma_{ii}}`$, $`\delta = (\hat{\mu} - \mu)/u`$, $`s_i = \delta_i / (\sqrt{d_i}/u) = (\hat{\mu}_i - \mu_i)/\sqrt{d_i}`$, and $`F = \mathrm{diag}(d)^{-1/2} B \in \mathbb{R}^{N \times K}`$. Then

```math
(\hat{\mu} - \mu)^\top \Sigma^{-1} (\hat{\mu} - \mu) = s^\top (I_N + F F^\top)^{-1} s.
```

By the Woodbury matrix identity, $`(I_N + F F^\top)^{-1} = I_N - F G^{-1} F^\top`$ where $`G = I_K + F^\top F \in \mathbb{R}^{K \times K}`$. To rewrite the difference between $`\|s\|_2^2`$ and $`s^\top F G^{-1} F^\top s`$ as a sum of squares, define $`c = G^{-1} F^\top s \in \mathbb{R}^K`$ and the residual $`r = s - F c \in \mathbb{R}^N`$. Using $`(I_K + F^\top F)c = F^\top s`$, expanding $`\|r\|_2^2 + \|c\|_2^2`$ gives:

```math
\|r\|_2^2 + \|c\|_2^2 = s^\top s - 2 s^\top F c + c^\top (F^\top F + I_K) c = s^\top s - s^\top F G^{-1} F^\top s = s^\top (I_N + F F^\top)^{-1} s.
```

Thus `_mean_mahalanobis` solves the $`K \times K`$ positive definite system $`G c = F^\top s`$ via Cholesky factorization and returns $`(\|r\|_2^2 + \|c\|_2^2)/N`$ in $`O(N K^2 + K^3)`$ operations (or $`\|s\|_2^2 / N`$ directly when $`K = 0`$).

### 3.4 Optional Rademacher Trace-Ratio Estimator

When `trace_probes` $`P \ge 2`$ and $`N > N_{\mathrm{exact}}`$, `covariance_nmse` and `second_moment_nmse` are estimated using Girard–Hutchinson trace estimation with $`P`$ independent Rademacher probe vectors $`v_1, \dots, v_P \in \{-1, +1\}^N`$ ($`\mathbb{P}(v_{p,i} = \pm 1) = 1/2`$, so $`\mathbb{E}[v_p v_p^\top] = I_N`$).

For any matrix $`A \in \mathbb{R}^{N \times N}`$, $`\mathbb{E}[\|A v_p\|_2^2] = \mathrm{tr}(A^\top A) = \|A\|_F^2`$. Setting $`N_p = \|(\hat{\Sigma} - \Sigma) v_p\|_2^2`$ and $`D_p = \|\Sigma v_p\|_2^2`$, `_trace_ratio` computes

```math
\widehat{\mathrm{NMSE}}_P = \frac{\frac{1}{P} \sum_{p=1}^P N_p}{\frac{1}{P} \sum_{p=1}^P D_p},
```

using only matrix-vector products `matvec`. The numerator and denominator sample means are unbiased estimators of $`\|\hat{\Sigma} - \Sigma\|_F^2`$ and $`\|\Sigma\|_F^2`$, whereas their finite-probe ratio can be biased. `metric_details` records all $`P`$ probe values $`(N_p)_{p=1}^P`$, $`(D_p)_{p=1}^P`$, and their $`2 \times 2`$ sample covariance matrix to quantify the Monte Carlo variance of the trace approximation.

## 4. Structural Diagnostics and Trial Statuses

### 4.1 Positive-Semidefiniteness Diagnostics

For both `prediction` and `truth`, `_diagnose` records `minimum_diagonal`, `negative_diagonal_count`, `zero_diagonal_count`, and a `psd` dictionary:

- **`FactorMatrix`**: `status = "psd"`, `method = "factor_construction"`, `scope = "full"`, `certificate = True`, and `strict_pd_sufficient_condition = bool(np.all(specific > 0.0))`.
- **`DenseMatrix` with $`N \le N_{\mathrm{exact}}`$**: computes all eigenvalues $`\lambda_0 \le \dots \le \lambda_{N-1}`$ of the unit-scaled matrix $`u^{-2} \hat{\Sigma}`$ (`np.linalg.eigvalsh`). With tolerance $`\tau = 128 \varepsilon_{\mathrm{mach}} N \max(1, \max_i |\lambda_i|)`$, reports `status = "indefinite"` if $`\lambda_0 < -\tau`$ and `"psd_within_tolerance"` otherwise (`scope = "full"`, `certificate = False`).
- **`DenseMatrix` with $`N > N_{\mathrm{exact}}`$**: computes the minimum eigenvalue on each declared block $`\mathcal{B}_1, \dots, \mathcal{B}_m`$. Reports `status = "indefinite_checked_block"` if any block minimum eigenvalue is below $`-128 \varepsilon_{\mathrm{mach}}`$, and `"checked_blocks_nonnegative"` otherwise (`scope = "tested_blocks_only"`, `certificate = False`, since only the sampled principal blocks are inspected).

### 4.2 Status Classification

`score` assigns one of four status strings:

- `"ok"`: both inputs are valid and all ten metrics are finite.
- `"partial"`: both inputs pass structural validation, but at least one metric is undefined on the input domain; available metrics remain populated and `errors` lists the reason for each `None` metric.
- `"invalid_prediction"`: `prediction` fails shape, type, finiteness, symmetry, or nonnegative specific-variance validation.
- `"invalid_truth"`: `truth` fails validation.

All outputs are passed through `finite_json`, which converts any residual `NaN` or `±Inf` float to `None` and appends an error entry so serialized JSON reports conform to RFC 8259 (`allow_nan=False`).

## 5. Multi-Objective Pareto Frontiers

[`src/covbench/pareto.py`](../src/covbench/pareto.py) partitions summary records into homogeneous comparison groups and extracts the nondominated set within each group.

### 5.1 Grouping and Eligibility

Given `comparison_keys` (default `["comparison_key", "scenario_id", "mode", "track", "checkpoint"]`) and `objectives` (default `["completed_ns", "covariance_nmse"]`):

1. A record is eligible if and only if:
   - it is a mapping with `status` in `{"ok", "partial"}`,
   - every path in `comparison_keys` resolves to a non-null value, and
   - every path in `objectives` resolves to a finite real number (`not isinstance(v, bool)` and `math.isfinite(float(v))`).
2. Any record failing these checks is placed in `excluded` with its original index and a reason code (`"invalid_record"`, `"invalid_status"`, `"invalid_comparison_metadata"`, or `"invalid_objective"`).

Eligibility is evaluated on the requested `objectives`, so a record with `status == "partial"` is included whenever every requested objective is finite, and excluded only when a requested objective is `null` or non-finite.

### 5.2 Dominance Relation

Within a single comparison group, let $`v^{(a)} = (v_1^{(a)}, \dots, v_M^{(a)}) \in \mathbb{R}^M`$ denote the vector of selected objective values for eligible record $`a`$. Because all timing and loss objectives are minimized, record $`b`$ **dominates** record $`a`$ ($`v^{(b)} \prec v^{(a)}`$) if and only if

```math
\forall m \in \{1, \dots, M\}, \quad v_m^{(b)} \le v_m^{(a)} \qquad \text{and} \qquad \exists m \in \{1, \dots, M\}, \quad v_m^{(b)} < v_m^{(a)}.
```

The group's `frontier` consists of all eligible records $`a`$ in the group for which no record $`b`$ in the group satisfies $`v^{(b)} \prec v^{(a)}`$. Records with identical objective vectors ($`v^{(a)} = v^{(b)}`$) do not dominate one another and are both retained in `frontier`.
