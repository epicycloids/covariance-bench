"""Subprocess trial runner, summary aggregation, and lossless report regrading."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
from time import monotonic, perf_counter_ns
import uuid

import numpy as np

from .api import API_VERSION, DenseMatrix, FactorMatrix, Prediction
from .metrics import score
from .provenance import EXCLUDED, derived_seed, digest, environment, file_hash, tree_hash
from .scenarios import generate, load_suite


class CandidateFailure(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def arrays(record):
    return {name: getattr(record, name) for name in record.__dataclass_fields__}


def manifest(path):
    """Load and validate ``candidate.json`` from a candidate directory."""
    data = json.loads((Path(path) / "candidate.json").read_text())
    if data.get("api_version") != API_VERSION:
        raise ValueError("candidate api_version must be 1")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ValueError("candidate needs a nonempty name")
    entry = data.get("entrypoint", "")
    if not isinstance(entry, str) or entry.count(":") != 1:
        raise ValueError("entrypoint must be module:factory")
    module, factory = entry.split(":")
    if not all(x.isidentifier() for x in module.split(".")) or not factory.isidentifier():
        raise ValueError("entrypoint must be module:factory")
    if not isinstance(data.get("config", {}), dict):
        raise ValueError("candidate config must be an object")
    modes = data.get("modes", ["offline", "online"])
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"offline", "online"}:
        raise ValueError("modes must select offline and/or online")
    return dict(data, config=data.get("config", {}), modes=modes)


class Worker:
    """Manage a single-trial candidate worker subprocess and enforce timeout/RSS limits."""

    def __init__(self, candidate, folder, *, threads, timeout, memory_mib):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True)
        self.timeout = timeout
        self.memory_bytes = int(memory_mib * 1024**2) if memory_mib else None
        self.buffer = b""
        env = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
            env[name] = str(threads)
        env["PYTHONHASHSEED"] = "0"
        # Candidate code is imported from the frozen copy; package imports are
        # resolved from this interpreter's installed environment.
        self.log = (self.folder / "worker.log").open("wb")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "covbench.worker", str(candidate), str(self.folder)],
            cwd=self.folder, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, start_new_session=True,
        )
        self.affinity = None
        if hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0))[:threads]
            try:
                os.sched_setaffinity(self.process.pid, cpus)
                self.affinity = cpus
            except (OSError, PermissionError):
                self.affinity = None

    def request(self, command, **values):
        start = perf_counter_ns()
        deadline = monotonic() + self.timeout
        try:
            self.process.stdin.write((json.dumps(dict(command=command, **values), allow_nan=False) + "\n").encode())
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            raise CandidateFailure("worker_crash", str(exc)) from exc
        while b"\n" not in self.buffer:
            if monotonic() >= deadline:
                raise CandidateFailure("timeout", f"{command} exceeded {self.timeout}s")
            if self.memory_bytes:
                status = Path(f"/proc/{self.process.pid}/status")
                try:
                    peak = next(int(line.split()[1]) * 1024 for line in status.read_text().splitlines() if line.startswith("VmHWM:"))
                    if peak > self.memory_bytes:
                        raise CandidateFailure("memory_limit", f"worker peak RSS exceeded {self.memory_bytes} bytes")
                except (FileNotFoundError, ProcessLookupError, StopIteration):
                    pass
            ready, _, _ = select.select([self.process.stdout], [], [], min(.05, max(0., deadline - monotonic())))
            if ready:
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise CandidateFailure("worker_crash", f"worker exited during {command}")
                self.buffer += data
                if len(self.buffer) > 1024**2:
                    raise CandidateFailure("protocol_error", "worker response exceeded one MiB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            result = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise CandidateFailure("protocol_error", "stdout was not a JSON protocol reply") from exc
        result["rpc_wall_ns"] = perf_counter_ns() - start
        if result["status"] != "ok":
            raise CandidateFailure(result["status"], f"{result.get('error_type')}: {result.get('error')}")
        if self.memory_bytes and (result.get("peak_rss_bytes") or 0) > self.memory_bytes:
            raise CandidateFailure("memory_limit", "worker peak RSS exceeded the configured limit")
        return result

    def close(self):
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def read_prediction(path):
    with np.load(path, allow_pickle=False) as saved:
        mean = saved["mean"]
        matrix = DenseMatrix(saved["dense"]) if "dense" in saved else FactorMatrix(saved["loadings"], saved["specific"])
    return Prediction(mean, matrix)


def _input(folder, arrays):
    path = Path(folder) / "current-input.npz"
    np.savez(path, **arrays)
    return str(path)


def run_trial(dataset, candidate, spec, folder, *, mode, threads, timeout, memory_mib, batch_size, scoring_seed):
    """Execute one cold-process trial for ``dataset`` and ``mode``, returning stage timings and query scores."""
    worker = Worker(candidate, folder, threads=threads, timeout=timeout, memory_mib=memory_mib)
    records, updates, fits, creations = [], [], [], []
    peak = 0
    current = 0
    processed_queries = 0
    import_result = None
    try:
        import_result = worker.request("load", entrypoint=spec["entrypoint"])
        peak = max(peak, import_result.get("peak_rss_bytes") or 0)
        if mode == "online":
            creation = worker.request("create", problem=dict(n_assets=dataset.n_assets, mode=mode), config=spec["config"])
            creations.append(creation)
        for asof in sorted({query.asof for query in dataset.queries}):
            group = [i for i, q in enumerate(dataset.queries) if q.asof == asof]
            phase_update_start = len(updates)
            if mode == "offline":
                creation = worker.request("create", problem=dict(n_assets=dataset.n_assets, mode=mode), config=spec["config"])
                creations.append(creation)
                obs = dataset.observations_at(asof)
                fit = worker.request("fit", input=_input(folder, arrays(obs)))
                fit.update(asof=asof, n_points=len(obs.time))
                fits.append(fit)
            else:
                end = int(np.searchsorted(dataset.arrival, asof, side="right"))
                while current < end:
                    stop = min(current + batch_size, end)
                    chunk = {k: v[current:stop] for k, v in arrays(dataset.events).items()}
                    response = worker.request("update", input=_input(folder, chunk))
                    response.update(n_events=stop-current, asof=asof,
                                    n_deletes=int(np.count_nonzero(chunk["operation"])))
                    updates.append(response)
                    current = stop
            phase = []
            for index in group:
                query = dataset.queries[index]
                result = worker.request("predict", query=asdict(query))
                peak = max(peak, result.get("peak_rss_bytes") or 0)
                path = Path(result["output"])
                prediction = read_prediction(path)
                losses = score(prediction, dataset.truth[index], seed=(scoring_seed + index) % 2**64)
                record = dict(query=asdict(query), status=losses["status"], metrics=losses["metrics"],
                                  metric_details=losses.get("metric_details", {}), diagnostics=losses.get("diagnostics", {}),
                                  errors=losses.get("errors", []), predict_ns=result["duration_ns"],
                                  peak_rss_bytes=result.get("peak_rss_bytes"),
                                  output=path.name, output_sha256=file_hash(path),
                                  fit_ns=fit["duration_ns"] if mode == "offline" else None,
                                  create_ns=creation["duration_ns"] if mode == "offline" or asof == min(q.asof for q in dataset.queries) else 0,
                                  checkpoint=index // 2)
                phase.append(record)
                records.append(record)
                processed_queries += 1
            phase_updates = updates[phase_update_start:]
            update_ns = sum(r["duration_ns"] for r in phase_updates)
            event_count = sum(r["n_events"] for r in phase_updates)
            # Sum all checkpoint work (creation, fit or intervening update batches,
            # and both checkpoint predictions) into completed_ns.
            work = update_ns + sum(r["predict_ns"] for r in phase)
            if mode == "offline":
                work += fit["duration_ns"] + creation["duration_ns"]
            elif asof == min(q.asof for q in dataset.queries):
                work += creation["duration_ns"]
            for record in phase:
                record.update(completed_ns=work, update_total_ns=update_ns,
                              n_events=event_count, amortized_event_ns=work/event_count if event_count else None,
                              update_ns_per_event=update_ns/event_count if event_count else None,
                              update_batch_median_ns=float(np.median([r['duration_ns'] for r in phase_updates])) if phase_updates else None,
                              update_batch_p95_ns=float(np.quantile([r['duration_ns'] for r in phase_updates], .95)) if phase_updates else None)
    except CandidateFailure as exc:
        for row in records:
            if "completed_ns" not in row:
                row.update(status="incomplete_checkpoint", error=str(exc), completed_ns=None)
        for query in dataset.queries[processed_queries:]:
            records.append(dict(query=asdict(query), checkpoint=processed_queries // 2,
                                status=exc.status, error=str(exc), metrics={}, completed_ns=None))
            processed_queries += 1
    finally:
        worker.close()
    peak = max([peak] + [r.get("peak_rss_bytes") or 0 for r in creations + fits + updates])
    return dict(mode=mode, status="ok" if all(r["status"] in ("ok", "partial") for r in records) else "failed",
                import_result=import_result, cpu_affinity=worker.affinity,
                peak_rss_bytes=peak, creations=creations, fits=fits, updates=updates, records=records)


def summarize(trials, common):
    """Aggregate per-trial records by (scenario_id, mode, track, checkpoint).

    Timings are summarized by their sample median across all trials in the group,
    peak RSS by the maximum, and statistical losses by averaging timing repeats
    within each replication before computing the mean and Monte Carlo standard
    error across replications.
    """
    grouped = {}
    for trial in trials:
        for row in trial["records"]:
            key = (trial["scenario_id"], trial["mode"], row["query"]["track"], row["checkpoint"])
            grouped.setdefault(key, []).append((trial, row))
    output = []
    for (scenario, mode, track, checkpoint), rows in grouped.items():
        failures = sum(row["status"] not in ("ok", "partial") for _, row in rows)
        record = dict(common, scenario_id=scenario, mode=mode, track=track, checkpoint=checkpoint,
                      status="failed" if failures else ("partial" if any(r["status"] == "partial" for _, r in rows) else "ok"),
                      trials=len(rows), failures=failures,
                      replications=len({t["replication"] for t, _ in rows}), uncertainty={})
        for name in ("completed_ns", "fit_ns", "predict_ns", "update_total_ns", "amortized_event_ns",
                     "update_ns_per_event", "update_batch_median_ns", "update_batch_p95_ns"):
            values = [r.get(name) for _, r in rows]
            finite = [v for v in values if v is not None and np.isfinite(v)]
            record[name] = float(np.median(finite)) if len(finite) == len(values) else None
        peaks = [r.get("peak_rss_bytes") for _, r in rows]
        record["peak_rss_bytes"] = max(peaks) if all(p is not None for p in peaks) else None
        names = sorted({name for _, row in rows for name in row["metrics"]})
        for name in names:
            per_replication = {}
            for trial, row in rows:
                value = row["metrics"].get(name)
                per_replication.setdefault(trial["replication"], []).append(value)
            complete = all(v is not None and np.isfinite(v) for values in per_replication.values() for v in values)
            if complete and not failures:
                values = np.array([np.mean(v) for v in per_replication.values()])
                record[name] = float(values.mean())
                record["uncertainty"][name] = dict(n=len(values), mc_se=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)
            else:
                record[name] = None
                record["uncertainty"][name] = dict(n_complete=sum(all(v is not None for v in values) for values in per_replication.values()), mc_se=None)
        output.append(record)
    return output


def evaluate(candidate, suite, output, *, root_seed, replications=3, timing_repeats=3,
             modes=None, threads=1, timeout=60., memory_mib=1024, batch_size=128):
    """Freeze ``candidate``, run all scenarios in ``suite``, and write ``<output>/results.json``."""
    if min(replications, timing_repeats, threads, batch_size) < 1 or timeout <= 0 or memory_mib < 0:
        raise ValueError("counts and timeout must be positive; memory_mib must be nonnegative")
    candidate, suite, output = Path(candidate).resolve(), Path(suite).resolve(), Path(output).resolve()
    spec = manifest(candidate)
    selected_modes = modes or spec["modes"]
    if not set(selected_modes) <= set(spec["modes"]):
        raise ValueError("requested mode is not declared by candidate")
    scenarios = load_suite(suite)
    candidate_hash = tree_hash(candidate)
    evaluator_root = Path(__file__).parent
    evaluator_hash = tree_hash(evaluator_root)
    policy = dict(threads=threads, timeout=timeout, memory_mib=memory_mib, batch_size=batch_size,
                  timing_repeats=timing_repeats, replications=replications, cold_worker=True,
                  query_cost="all required checkpoint predictions", input_transport_timed=False,
                  output_materialization_timed=True, output_file_io_timed=False)
    observed = environment(threads)
    comparison_key = digest(dict(suite=file_hash(suite), evaluator=evaluator_hash,
                                 environment=observed, policy=policy, root_seed=root_seed))
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "candidate"
    shutil.copytree(candidate, artifact, ignore=shutil.ignore_patterns(*EXCLUDED))
    if tree_hash(artifact) != candidate_hash:
        raise RuntimeError("candidate changed during freezing")
    trials = []
    for scenario in scenarios:
        for replication in range(replications):
            seed = derived_seed(root_seed, scenario.id, replication)
            dataset = generate(scenario, seed)
            inputs_hash = digest(dict(arrays={name: [str(value.dtype), value.shape, file_hash_bytes(value.tobytes())]
                                             for name, value in dict(arrays(dataset.events), arrival=dataset.arrival).items()},
                                      queries=[asdict(q) for q in dataset.queries]))
            # Each timing repetition and execution mode runs in a fresh worker process.
            for repeat in range(timing_repeats):
                for mode in selected_modes:
                    folder = output / "trials" / uuid.uuid4().hex
                    result = run_trial(dataset, artifact, spec, folder, mode=mode, threads=threads,
                                       timeout=timeout, memory_mib=memory_mib, batch_size=batch_size,
                                       scoring_seed=seed ^ 0xD7AD5C32)
                    result.update(scenario_id=scenario.id, scenario_config=scenario.config, instance_seed=seed,
                                  replication=replication, timing_repeat=repeat, inputs_sha256=inputs_hash,
                                  folder=str(folder.relative_to(output)))
                    (folder / "trial.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
                    trials.append(result)
                    print(f"{scenario.id} {mode} r{replication} t{repeat}: {result['status']}", file=sys.stderr, flush=True)
    if tree_hash(candidate) != candidate_hash or tree_hash(artifact) != candidate_hash or tree_hash(evaluator_root) != evaluator_hash:
        raise RuntimeError("candidate or evaluator changed during evaluation")
    common = dict(candidate=spec["name"], candidate_sha256=candidate_hash, comparison_key=comparison_key)
    report = dict(schema_version=1, run_id=str(uuid.uuid4()), api_version=API_VERSION,
                  candidate=dict(spec, sha256=candidate_hash), evaluator_sha256=evaluator_hash,
                  suite_sha256=file_hash(suite), root_seed=root_seed, environment=observed, policy=policy,
                  trials=trials, records=summarize(trials, common))
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def file_hash_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def regrade(report_path, output_path):
    """Recompute losses from saved ``prediction-XXXX.npz`` files under the current scorer.

    Verifies that regenerated dataset inputs and saved prediction files match
    their recorded SHA-256 digests, updates metric scores and summaries, and
    writes a new report alongside ``report_path`` preserving measurement timings.
    """
    report_path, output_path = Path(report_path).resolve(), Path(output_path).resolve()
    if output_path.parent != report_path.parent or output_path == report_path or output_path.exists():
        raise ValueError("regrade output must be a new file beside the original report")
    report = json.loads(report_path.read_text())
    scoring_hash = tree_hash(Path(__file__).parent)
    cache = {}
    for trial in report["trials"]:
        key = digest([trial["scenario_config"], trial["instance_seed"]])
        if key not in cache:
            from .scenarios import Scenario
            cache[key] = generate(Scenario(trial["scenario_id"], trial["scenario_config"]), trial["instance_seed"])
        dataset = cache[key]
        fingerprint = digest(dict(arrays={name: [str(value.dtype), value.shape, file_hash_bytes(value.tobytes())]
                                           for name, value in dict(arrays(dataset.events), arrival=dataset.arrival).items()},
                                  queries=[asdict(q) for q in dataset.queries]))
        if fingerprint != trial["inputs_sha256"]:
            raise ValueError("generator changed the measured input; regrade refused")
        for index, row in enumerate(trial["records"]):
            if "output" not in row or row["status"] not in ("ok", "partial"):
                continue
            path = report_path.parent / trial["folder"] / row["output"]
            if file_hash(path) != row["output_sha256"]:
                raise ValueError("measured prediction changed; regrade refused")
            if row["query"] != asdict(dataset.queries[index]):
                raise ValueError("query changed; regrade refused")
            losses = score(read_prediction(path), dataset.truth[index],
                           seed=((trial["instance_seed"] ^ 0xD7AD5C32) + index) % 2**64)
            row.update(status=losses["status"], metrics=losses["metrics"],
                       metric_details=losses["metric_details"], diagnostics=losses["diagnostics"], errors=losses["errors"])
    old_key = report["records"][0]["comparison_key"]
    comparison_key = digest([old_key, scoring_hash])
    common = dict(candidate=report["candidate"]["name"], candidate_sha256=report["candidate"]["sha256"],
                  comparison_key=comparison_key)
    report["measurement_evaluator_sha256"] = report.get("measurement_evaluator_sha256", report["evaluator_sha256"])
    report["scoring_evaluator_sha256"] = scoring_hash
    report["regraded_from_sha256"] = file_hash(report_path)
    report["records"] = summarize(report["trials"], common)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
