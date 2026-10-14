# Example core run

Three baselines, 12 scenarios, three independent simulated draws, two fresh
process repetitions, and both execution modes: 432 trials and 2,592 predictions.
All execution trials completed. Undefined SPD metrics remain explicitly null.

The JSON files retain run metadata and per-regime summaries. These are small
development examples, not validated solver rankings: three draws give limited
loss precision. `frontier.json` uses completed checkpoint work and covariance
NMSE; other objectives can be selected with `covbench frontier`.

Measurement used the source at commit `f6bc432`. Scoring was regraded from the
same measured outputs after adding a diagonal-only truth fast path compatible
with older SciPy. Covariance and second-moment losses and all measured timings
are unchanged. The reports preserve separate measurement/scoring identities.

Raw predictions and logs are retained in the original local run directories,
not duplicated in these compact examples. Reproduce a full run from the root:

```sh
covbench run --candidate candidates/previous-tick --suite suites/core.json \
  --root-seed 20261013 --replications 3 --timing-repeats 2 --output runs/tick-core
```

Use matching policies, root seed and environment for every candidate. The
optional native baseline used `async-covariance` v0.1.0. Its adapter refits at
each query; it is not a measurement of that package's local online engine.
