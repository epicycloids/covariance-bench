"""Project workspace and regime-vector Pareto views over native benchmark reports.

No estimator, generator, runner or regrader is executed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from research_market.project_workspace import Workspace, digest

from .pareto import frontiers

KEYS = ["comparison_key", "scenario_id", "mode", "track", "checkpoint"]
OBJECTIVES = {"completed_ns", "peak_rss_bytes", "covariance_nmse", "second_moment_nmse", "diagonal_nmse", "correlation_mse", "portfolio_risk_relative_mse", "portfolio_second_moment_relative_mse", "mean_variance_scaled_mse", "mean_mahalanobis", "gaussian_kl", "covariance_qlike"}
SOURCE_REVISION = "7ef8e9bece93eba6bed621e996967a04b1c3e8ec"


def regime(row):
    return (row["scenario_id"], row["mode"], row["track"], row["checkpoint"])


def check_report(report):
    required = {"schema_version", "api_version", "candidate", "suite_sha256", "evaluator_sha256", "root_seed", "environment", "policy", "records"}
    if not required.issubset(report) or type(report["schema_version"]) is not int or type(report["api_version"]) is not int or report["schema_version"] != 1 or report["api_version"] != 1:
        raise ValueError("Use the native covariance-bench API-v1 result envelope.")
    if not isinstance(report["records"], list) or not report["records"]:
        raise ValueError("Native results must retain per-regime records, including failures.")
    seen = set()
    for row in report["records"]:
        if any(key not in row for key in KEYS) or row["mode"] not in {"offline", "online"} or row["track"] not in {"historical", "forecast"} or type(row["checkpoint"]) is not int or row["checkpoint"] < 0:
            raise ValueError("Native records need comparison identity, mode, track and checkpoint.")
        key = (row["comparison_key"], regime(row))
        if key in seen or row.get("candidate_sha256") != report["candidate"]["sha256"] or row.get("candidate") != report["candidate"]["name"]:
            raise ValueError("Duplicate regime or candidate identity mismatch.")
        seen.add(key)
    return report


def comparable_profiles(reports, objectives, comparison_key=None, panel=None):
    """Every selected regime×objective is a coordinate, never an averaged scalar."""
    if not objectives or len(set(objectives)) != len(objectives) or not set(objectives).issubset(OBJECTIVES):
        raise ValueError("Select distinct native loss/runtime/host-memory objectives.")
    rows, provenance = [], {}
    for source in reports:
        report = check_report(source["report"])
        source_id = source["id"]
        environment_key = digest({"suite": report["suite_sha256"], "environment": report["environment"], "policy": report["policy"], "root_seed": report["root_seed"], "measurement_evaluator": report.get("measurement_evaluator_sha256", report["evaluator_sha256"]), "scoring_evaluator": report.get("scoring_evaluator_sha256", report["evaluator_sha256"])})
        provenance[source_id] = {"source": source.get("provenance"), "candidate": report["candidate"], "environment_key": environment_key, "environment": report["environment"], "policy": report["policy"], "suite_sha256": report["suite_sha256"], "measurement_evaluator_sha256": report.get("measurement_evaluator_sha256", report["evaluator_sha256"]), "scoring_evaluator_sha256": report.get("scoring_evaluator_sha256", report["evaluator_sha256"]), "measurement_source_commit": report.get("measurement_source_commit"), "note": report.get("note")}
        for row in report["records"]:
            if comparison_key is None or row["comparison_key"] == comparison_key:
                rows.append({**row, "source_id": source_id, "environment_key": environment_key})
    keys = sorted({(row["comparison_key"], row["environment_key"]) for row in rows})
    output = []
    for native_key, environment_key in keys:
        group = [row for row in rows if (row["comparison_key"], row["environment_key"]) == (native_key, environment_key)]
        regimes = sorted({regime(row) for row in group}) if panel is None else [tuple(item) for item in panel]
        if not regimes or len(set(regimes)) != len(regimes):
            raise ValueError("Select an explicit nonempty distinct common regime panel.")
        dimensions = [{"scenario_id": item[0], "mode": item[1], "track": item[2], "checkpoint": item[3], "metric": metric, "direction": "minimize"} for item in regimes for metric in objectives]
        candidates = {}
        for row in group:
            candidates.setdefault(row["candidate_sha256"], []).append(row)
        profiles = []
        for candidate, candidate_rows in candidates.items():
            table = {}
            conflicts = []
            for row in candidate_rows:
                key = regime(row)
                if key in table and {k: v for k, v in table[key].items() if k != "source_id"} != {k: v for k, v in row.items() if k != "source_id"}:
                    conflicts.append(key)
                table[key] = row
            matrix, vector, excluded = [], [], []
            for key in regimes:
                row = table.get(key)
                reason = None
                if row is None:
                    reason = "missing_regime"
                elif row.get("status") not in {"ok", "partial"} or row.get("failures", 0):
                    reason = "failed_or_incomplete_regime"
                elif any(type(row.get(metric)) not in (int, float) or not math.isfinite(row[metric]) for metric in objectives):
                    reason = "selected_objective_unavailable"
                elif key in conflicts:
                    reason = "conflicting_runs_require_explicit_source_selection"
                cell = {"regime": {"scenario_id": key[0], "mode": key[1], "track": key[2], "checkpoint": key[3]}, "metrics": {metric: row.get(metric) if row else None for metric in objectives}, "uncertainty": row.get("uncertainty") if row else None, "status": row.get("status") if row else "missing", "failures": row.get("failures") if row else None, "source_id": row.get("source_id") if row else None, "exclusion": reason}
                matrix.append(cell)
                if reason:
                    excluded.append({"regime": list(key), "reason": reason})
                else:
                    vector.extend(float(row[metric]) for metric in objectives)
            profiles.append({"candidate_sha256": candidate, "candidate_name": candidate_rows[0]["candidate"], "matrix": matrix, "coverage": {"expected_regimes": len(regimes), "comparable_regimes": len(regimes) - len(excluded), "excluded": excluded}, "eligible_cross_regime": not excluded, "vector": vector if not excluded else None})
        eligible = [profile for profile in profiles if profile["eligible_cross_regime"]]
        dominance = []
        for left in eligible:
            for right in eligible:
                if left is not right and all(a <= b for a, b in zip(left["vector"], right["vector"])) and any(a < b for a, b in zip(left["vector"], right["vector"])):
                    dominance.append({"dominates": left["candidate_sha256"], "dominated": right["candidate_sha256"]})
        dominated = {edge["dominated"] for edge in dominance}
        native_rows = [{**row, "status": "failed" if row.get("failures", 0) else row.get("status")} for row in group]
        output.append({"comparison_key": native_key, "environment_key": environment_key, "dimensions": dimensions, "candidate_profiles": profiles, "cross_regime_frontier": [profile["candidate_sha256"] for profile in eligible if profile["candidate_sha256"] not in dominated], "dominance": dominance, "per_regime": frontiers(native_rows, objectives, KEYS + ["environment_key"]), "scope": "Point-estimate dominance across every selected common regime×metric; specialists remain nondominated when another regime trades off. Uncertainty is retained, not converted into proof."})
    return {"objectives": objectives, "panels": output, "provenance": provenance, "memory_scope": "peak_rss_bytes is cumulative worker host VmHWM at the checkpoint; GPU memory is not measured by this coordinate."}


class CovarianceWorkspace(Workspace):
    methods = Workspace.methods | {"frontier", "mark_suite_tuned"}

    def __init__(self, repository, state):
        self.repository = repository
        self.suites = {path.stem: json.loads(path.read_text()) for path in (repository / "suites").glob("*.json")}
        self.examples = []
        for name in ("zero-core.json", "tick-core.json", "native-core.json"):
            path = repository / "examples" / name
            report = json.loads(path.read_text())
            self.examples.append({"id": "example:" + name, "report": report, "provenance": {"repository": "https://github.com/epicycloids/covariance-bench", "source_revision": SOURCE_REVISION, "path": "examples/" + name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "origin": "existing_benchmark_development_example_not_reproduced_here"}})
        tasks = [{"id": "covbench/" + name, "title": name.title() + " estimator panel", "family": "moment-estimation", "status": "defined", "contract": {"suite": data, "suite_sha256": hashlib.sha256((repository / "suites" / (name + ".json")).read_bytes()).hexdigest(), "api_version": 1, "query_tracks": ["historical", "forecast"], "execution_modes": ["offline", "online"], "targets": ["covariance", "second_moment"], "input_units": "levels already in estimation units", "memory": "worker host RSS; accelerator allocation is not inferred"}} for name, data in self.suites.items()]
        super().__init__("covariance-bench", {"title": "Covariance & second-moment estimators", "goal": "Develop useful estimators and compare complete regime performance profiles in error, completed runtime and honestly measured memory.", "methods": "Frozen estimator candidates; native benchmark requests; per-regime frontiers and selected-panel regime-vector dominance.", "native_capabilities": ["API-v1 candidates", "native results", "regime profiles", "Pareto tradeoffs", "uncertainty", "development/heldout provenance"], "multivac_adapter": "Project-native evaluation requests; the workspace never executes covbench or a model."}, tasks, state)

    def validate_candidate(self, data, task):
        manifest = data["payload"]["candidate_manifest"]
        if manifest.get("api_version") != 1 or not manifest.get("entrypoint") or not set(manifest.get("modes", [])).issubset({"offline", "online"}) or not manifest.get("modes"):
            raise ValueError("Use the benchmark's actual candidate.json API-v1 factory/mode contract.")
        tree_sha = data["payload"].get("candidate_tree_sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", tree_sha):
            raise ValueError("Name the candidate tree SHA256 using the benchmark's native tree_hash protocol.")
        return {"candidate_manifest": manifest, "candidate_tree_sha256": tree_sha}

    def evaluation_context(self, args, task, candidate):
        policy = args["policy"]
        if not {"batch_size", "threads", "replications", "timing_repeats", "root_seed", "environment"}.issubset(policy):
            raise ValueError("Name the actual paired seed, batch/incremental, timing, thread and environment policy.")
        return {"suite_sha256": task["contract"]["suite_sha256"], "policy": policy, "candidate_artifact_sha256": candidate["data"]["artifact"]["sha256"], "candidate_tree_sha256": candidate["data"]["payload"]["candidate_tree_sha256"], "api_version": 1}

    def validate_evidence(self, report, request, candidate):
        super().validate_evidence(report, request, candidate)
        native = check_report(report["native_report"])
        expected = request["data"]["context"]
        if native["suite_sha256"] != expected["suite_sha256"] or native["candidate"]["sha256"] != expected["candidate_tree_sha256"]:
            raise ValueError("Native covariance results differ from the frozen suite/candidate identity.")
        policy = expected["policy"]
        expected_policy = {key: value for key, value in policy.items() if key not in {"root_seed", "environment"}}
        if native["root_seed"] != policy["root_seed"] or native["environment"] != policy["environment"] or native["policy"] != expected_policy:
            raise ValueError("Native measurement policy/environment differs from the evaluation request.")

    def reports(self):
        return self.examples + [{"id": item["id"], "report": item["data"]["report"]["native_report"], "provenance": {"origin": "project_owner_reviewed_donor_report", "evidence_sha256": item["sha256"]}} for item in self.accepted()]

    def frontier(self, args, actor):
        reports = self.reports()
        if args.get("source_ids"):
            reports = [source for source in reports if source["id"] in args["source_ids"]]
        return comparable_profiles(reports, args.get("objectives", ["covariance_nmse", "completed_ns", "peak_rss_bytes"]), args.get("comparison_key"), args.get("panel"))

    def mark_suite_tuned(self, args, actor):
        actor.require(self.id, owner=True)
        if args["suite"] not in self.suites:
            raise ValueError("Unknown native suite.")
        return self.store.put("tuning", {"suite": args["suite"], "reason": args["reason"], "effective_partition": "development_after_tuning"}, actor, args.get("id"))

    def snapshot(self, actor):
        output = super().snapshot(actor)
        output.update(native_suites=self.suites, regime_catalog={scenario["id"]: scenario for suite in self.suites.values() for scenario in suite["scenarios"]}, tuning=self.store.list("tuning"), regime_profiles=self.frontier({}, actor), target_semantics={"covariance": "latent increment covariance excluding observation noise", "second_moment": "covariance + outer(mean,mean)", "timing": "completed_ns includes required predictions, synchronization/materialization and applicable fit/update/construction", "online": "replacement/deletion/delayed edits; scalar batch_size=1 and batched delivery remain different policies"})
        return output


def create(repository: Path, state: Path):
    return CovarianceWorkspace(repository, state)
