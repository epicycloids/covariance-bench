"""Worker subprocess executing candidate API stages over a JSON stdin/stdout protocol.

Standard output is reserved for JSON protocol replies; candidate output to
``sys.stdout`` is redirected to ``sys.stderr`` during stage execution.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns
import traceback

import numpy as np

from covbench.api import DenseMatrix, Events, FactorMatrix, Observations, Prediction, Problem, Query


def materialize(prediction, n):
    """Validate a candidate Prediction and copy its arrays to contiguous float64 host memory."""
    if not isinstance(prediction, Prediction):
        raise TypeError("predict must return covbench.api.Prediction")
    def host(value):
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
            raise TypeError("prediction components must be real host NumPy arrays")
        return np.array(value, dtype=np.float64, copy=True, order="C")
    mean = host(prediction.mean)
    if mean.shape != (n,) or not np.all(np.isfinite(mean)):
        raise ValueError("mean must be a complete finite vector of length n_assets")
    matrix = prediction.covariance
    if isinstance(matrix, DenseMatrix):
        value = host(matrix.value)
        if value.shape != (n, n) or not np.all(np.isfinite(value)):
            raise ValueError("dense covariance must be a finite n_assets square matrix")
        scale = max((float(np.max(np.abs(value[i:i+128]), initial=0.))
                     for i in range(0, n, 128)), default=0.)
        if scale:
            for i in range(0, n, 128):
                difference = value[i:i+128]/scale - value.T[i:i+128]/scale
                if np.max(np.abs(difference), initial=0.) > 64*np.finfo(float).eps:
                    raise ValueError("dense covariance must be symmetric within 64 machine epsilons of its scale")
        return {"mean": mean, "dense": value}
    if isinstance(matrix, FactorMatrix):
        b = host(matrix.loadings)
        d = host(matrix.specific)
        if b.ndim != 2 or b.shape[0] != n or d.shape != (n,):
            raise ValueError("factor covariance must have shapes (n_assets, rank) and (n_assets,)")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("factor arrays must be finite and specific variances nonnegative")
        if not np.all(np.isfinite(np.einsum("ij,ij->i", b, b) + d)):
            raise ValueError("factor covariance diagonal overflows")
        return {"mean": mean, "loadings": b, "specific": d}
    raise TypeError("covariance must be DenseMatrix or FactorMatrix")


def peak_rss():
    """Return the worker's peak resident set size (VmHWM) in bytes on Linux, or None."""
    status = Path("/proc/self/status")
    if status.exists():
        return int(next(line.split()[1] for line in status.read_text().splitlines() if line.startswith("VmHWM:"))) * 1024
    return None


def main():
    candidate = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(candidate))
    estimator = None
    factory = None
    problem = None
    output_id = 0

    def sync():
        """Synchronize the candidate estimator and any initialized PyTorch/CuPy CUDA streams."""
        if estimator is not None:
            estimator.synchronize()
        # Synchronize the active CUDA device if torch or cupy has been imported.
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        cupy = sys.modules.get("cupy")
        if cupy is not None:
            cupy.cuda.runtime.deviceSynchronize()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            result = {}
            with redirect_stdout(sys.stderr):
                if command == "load":
                    module, name = request["entrypoint"].split(":")
                    start = perf_counter_ns()
                    factory = getattr(importlib.import_module(module), name)
                    result["duration_ns"] = perf_counter_ns() - start
                elif command == "create":
                    problem = Problem(**request["problem"])
                    sync()
                    start = perf_counter_ns()
                    estimator = factory(problem, request["config"])
                    for method in (problem.mode == "offline" and "fit" or "update", "predict", "synchronize"):
                        if not callable(getattr(estimator, method, None)):
                            raise TypeError(f"estimator is missing {method}")
                    sync()
                    result["duration_ns"] = perf_counter_ns() - start
                elif command in ("fit", "update"):
                    with np.load(request["input"], allow_pickle=False) as saved:
                        values = {k: saved[k] for k in saved.files}
                    for value in values.values():
                        value.setflags(write=False)
                    data = Observations(**values) if command == "fit" else Events(**values)
                    sync()
                    start = perf_counter_ns()
                    getattr(estimator, command)(data)
                    sync()
                    result["duration_ns"] = perf_counter_ns() - start
                elif command == "predict":
                    query = Query(**request["query"])
                    sync()
                    start = perf_counter_ns()
                    prediction = estimator.predict(query)
                    sync()
                    arrays = materialize(prediction, problem.n_assets)
                    sync()
                    result["duration_ns"] = perf_counter_ns() - start
                    path = output / f"prediction-{output_id:04d}.npz"
                    output_id += 1
                    np.savez_compressed(path, **arrays)
                    result["output"] = str(path)
                elif command == "close":
                    break
                else:
                    raise ValueError(f"unknown command {command}")
            result.update(status="ok", peak_rss_bytes=peak_rss())
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            result = dict(status="candidate_error", error_type=type(exc).__name__, error=str(exc), peak_rss_bytes=peak_rss())
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
