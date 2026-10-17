# Example Core-Suite Run Reports

This directory contains compact summary reports from evaluating the three reference baselines on [`suites/core.json`](../suites/core.json):

- [`zero-core.json`](zero-core.json): [`candidates/zero`](../candidates/zero/README.md)
- [`tick-core.json`](tick-core.json): [`candidates/previous-tick`](../candidates/previous-tick/README.md)
- [`native-core.json`](native-core.json): [`candidates/async-native`](../candidates/async-native/README.md) (using `async-covariance` v0.1.0)
- [`frontier.json`](frontier.json): per-regime Pareto frontiers for `completed_ns` and `covariance_nmse`

## Run Configuration

The three baselines were evaluated across all 12 core scenarios, $`R_{\mathrm{rep}} = 3`$ independent synthetic replications (`--root-seed 20261013`), $`R_{\mathrm{time}} = 2`$ cold-process timing repetitions, and both `offline` and `online` modes (`batch_size = 128`, `threads = 1`), totaling 432 worker trials and 2,592 query predictions. All 432 trials completed without timeouts or crashes.

Metrics whose domain conditions fail remain `null` (`status = "partial"`): for `zero`, zero marginal variances leave `correlation_mse`, `gaussian_kl`, and `covariance_qlike` `null` across all 144 regime summaries; for `async-native`, positive floored diagonals keep `correlation_mse` finite everywhere, and `gaussian_kl` and `covariance_qlike` are finite on 48 summaries (the 12 `gaussian-clean-16` summaries and the 36 factor-mode summaries at $`N = 1024`$) and `null` on the remaining 96 summaries ($`N \le 256`$ scenarios where unprojected dense overlap estimates are indefinite).

Timed execution used source commit `f6bc432`. The saved predictions were subsequently regraded (`covbench regrade`) after adding a diagonal-only truth fast path in `_mean_mahalanobis` for rank-0 scenarios (`student-noise-edits-16`), preserving all measured timings and Frobenius losses while recording separate measurement and scoring evaluator hashes.

## Reproducing a Full Run

These example JSON files retain run metadata and per-regime summaries (`records`), omitting the per-trial `.npz` prediction files and worker logs. To run a full evaluation from the repository root:

```sh
covbench run --candidate candidates/previous-tick --suite suites/core.json \
  --root-seed 20261013 --replications 3 --timing-repeats 2 --output runs/tick-core
```

Use matching flags, `--root-seed`, and environment across candidates when computing a new frontier with `covbench frontier`.
