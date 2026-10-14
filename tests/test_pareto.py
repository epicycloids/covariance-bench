import pytest

from covbench.pareto import frontiers


KEYS = ["suite", "evaluator", "hardware.id", "config", "track", "scenario", "mode"]
OBJECTIVES = ["timing.seconds", "metrics.covariance_nmse"]


def _record(name, seconds, loss, **updates):
    record = dict(candidate=name, status="ok", suite="s1", evaluator="e1",
                  hardware={"id": "cpu1"}, config={"pairs": 20}, track="historical",
                  scenario="clean", mode="offline", timing={"seconds": seconds},
                  metrics={"covariance_nmse": loss})
    record.update(updates)
    return record


def _names(points):
    return [point["record"]["candidate"] for point in points]


def test_incomparable_points_and_ties_are_retained():
    records = [_record("fast", 1., .2), _record("accurate", 2., .1),
               _record("tie", 1., .2), _record("dominated", 3., .3)]
    result = frontiers(records, OBJECTIVES, KEYS)
    assert _names(result["groups"][0]["frontier"]) == ["fast", "accurate", "tie"]
    assert _names(result["groups"][0]["dominated"]) == ["dominated"]
    assert result["excluded"] == []


@pytest.mark.parametrize("field,value", [("scenario", "noisy"), ("mode", "online"),
                                         ("hardware", {"id": "gpu1"}), ("suite", "s2"),
                                         ("config", {"pairs": 40}), ("track", "forecast")])
def test_different_regimes_never_dominate_each_other(field, value):
    records = [_record("first", 1., .1), _record("second", 20., 4., **{field: value})]
    result = frontiers(records, OBJECTIVES, KEYS)
    assert len(result["groups"]) == 2
    assert all(len(group["frontier"]) == 1 and not group["dominated"] for group in result["groups"])


def test_failure_missing_and_nonfinite_points_remain_explicit():
    records = [_record("valid", 1., .1), _record("timeout", 0., 0., status="timeout"),
               _record("invalid", 0., 0., status="invalid_prediction"),
               _record("nan", 0., float("nan")), _record("none", 0., None),
               _record("missing_metadata", 0., 0.)]
    del records[-1]["evaluator"]
    result = frontiers(records, OBJECTIVES, KEYS)
    assert _names(result["groups"][0]["frontier"]) == ["valid"]
    assert len(result["excluded"]) == 5
    assert [failure["index"] for failure in result["excluded"]] == [1, 2, 3, 4, 5]
    assert {failure["reason"] for failure in result["excluded"]} == {"invalid_status", "invalid_objective", "invalid_comparison_metadata"}


def test_partial_status_requires_selected_metric_to_exist():
    records = [_record("raw", 1., .1, status="partial")]
    records[0]["metrics"]["gaussian_kl"] = None
    assert _names(frontiers(records, OBJECTIVES, KEYS)["groups"][0]["frontier"]) == ["raw"]
    assert len(frontiers(records, ["metrics.gaussian_kl"], KEYS)["excluded"]) == 1


@pytest.mark.parametrize("objectives,keys", [([], KEYS), (OBJECTIVES, []),
                                            (["a", "a"], KEYS), (["a..b"], KEYS)])
def test_comparison_definition_is_explicit(objectives, keys):
    with pytest.raises(ValueError):
        frontiers([], objectives, keys)


def test_empty_records_are_well_defined():
    result = frontiers([], OBJECTIVES, KEYS)
    assert result["groups"] == []
    assert result["excluded"] == []
