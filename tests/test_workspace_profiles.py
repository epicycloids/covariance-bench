"""Synthetic regime bookkeeping only; no estimator or generator executes."""
import copy
import unittest

try:
    from covbench.workspace import comparable_profiles
except ModuleNotFoundError:
    comparable_profiles = None


def source(name, values, failed=False, missing=False, environment="same"):
    sha = (name[0] * 64)
    rows = [{"candidate": name, "candidate_sha256": sha, "comparison_key": "paired", "scenario_id": regime, "mode": "offline", "track": "historical", "checkpoint": 0, "status": "failed" if failed and index == 0 else "ok", "failures": int(failed and index == 0), "covariance_nmse": value, "completed_ns": 10.0, "peak_rss_bytes": 100.0, "uncertainty": {"covariance_nmse": {"mc_se": 0.2}}} for index, (regime, value) in enumerate(zip(("small-dense", "large-sparse"), values))]
    if missing: rows.pop()
    return {"id": name, "report": {"schema_version": 1, "api_version": 1, "candidate": {"name": name, "sha256": sha}, "suite_sha256": "suite", "evaluator_sha256": "evaluator", "root_seed": 1, "environment": {"id": environment}, "policy": {"batch_size": 128}, "records": rows}}


@unittest.skipIf(comparable_profiles is None, "Workspace adapter needs the optional private Multivac workspace source")
class WorkspaceProfilesTests(unittest.TestCase):
    def test_specialists_remain_nondominated_over_common_regime_vector(self):
        result = comparable_profiles([source("alpha", [1, 10]), source("beta", [10, 1])], ["covariance_nmse", "completed_ns", "peak_rss_bytes"])
        panel = result["panels"][0]
        self.assertEqual(set(panel["cross_regime_frontier"]), {"a" * 64, "b" * 64})
        self.assertEqual(len(panel["dimensions"]), 6)
        self.assertIsNotNone(panel["candidate_profiles"][0]["matrix"][0]["uncertainty"])

    def test_missing_failed_profiles_do_not_win_by_omission(self):
        result = comparable_profiles([source("alpha", [1, 1]), source("beta", [0, 0], missing=True), source("gamma", [0, 0], failed=True)], ["covariance_nmse"])
        self.assertEqual(result["panels"][0]["cross_regime_frontier"], ["a" * 64])
        self.assertEqual(sum(profile["eligible_cross_regime"] for profile in result["panels"][0]["candidate_profiles"]), 1)

    def test_environment_policies_are_separate_even_if_named_key_repeats(self):
        result = comparable_profiles([source("alpha", [1, 1]), source("beta", [0, 0], environment="different")], ["covariance_nmse"])
        self.assertEqual(len(result["panels"]), 2)
