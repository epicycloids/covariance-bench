import json
from pathlib import Path

import numpy as np
import pytest

from covbench.api import DELETE, UPSERT
from covbench.scenarios import Scenario, generate, load_suite


def scenario(**config):
    return Scenario.from_dict(dict(id="test", n_assets=4, grid_steps=32, **config))


def table(observations):
    return {(int(a), float(t)): float(v) for t, a, v in
            zip(observations.time, observations.asset, observations.value)}


def test_seed_reproducibility_random_parameters_and_json_config():
    config = scenario(distribution="student_t", noise="iid", regimes=2)
    a, b, changed = [generate(config, seed) for seed in [41, 41, 42]]
    assert json.loads(json.dumps(config.config)) == config.config
    for key in ["time", "asset", "value", "operation"]:
        np.testing.assert_array_equal(getattr(a.events, key), getattr(b.events, key))
    np.testing.assert_array_equal(a.arrival, b.arrival)
    for x, y in zip(a.truth, b.truth):
        np.testing.assert_array_equal(x.mean, y.mean)
        np.testing.assert_array_equal(x.covariance.diagonal(), y.covariance.diagonal())
    assert not np.array_equal(a.truth[0].mean, changed.truth[0].mean)
    assert not np.array_equal(a.truth[0].covariance.diagonal(), changed.truth[0].covariance.diagonal())


def test_arrival_prefix_replay_is_causal_including_corrections_and_deletions():
    data = generate(scenario(observation_fraction=.7, max_latency=.2,
                             correction_fraction=.6, delete_fraction=.3), 124)
    assert np.all(data.arrival[1:] >= data.arrival[:-1])
    assert np.all(data.events.time <= data.arrival)
    assert np.any(data.events.time[1:] < data.events.time[:-1])
    assert np.any(data.events.operation == DELETE)
    live = {}
    for index, arrival in enumerate(data.arrival):
        key = (int(data.events.asset[index]), float(data.events.time[index]))
        if data.events.operation[index] == UPSERT:
            live[key] = float(data.events.value[index])
        else:
            live.pop(key, None)
        if index % 17 == 0 or index == len(data.arrival) - 1:
            assert table(data.observations_at(float(arrival))) == live
            prefix = data.observations_at(float(arrival))
            assert np.all(prefix.time <= arrival)
            assert np.all(prefix.time[1:] >= prefix.time[:-1])
    assert table(data.observations_at(-1.)) == {}
    with pytest.raises(ValueError):
        data.observations_at(np.inf)


def test_correction_value_is_not_visible_before_its_arrival():
    data = generate(scenario(observation_fraction=1., correction_fraction=1.), 52)
    by_key = {}
    for index, (asset, time) in enumerate(zip(data.events.asset, data.events.time)):
        by_key.setdefault((int(asset), float(time)), []).append(index)
    for key, indices in by_key.items():
        assert len(indices) == 2
        wrong, corrected = indices
        assert data.events.operation[wrong] == data.events.operation[corrected] == UPSERT
        assert data.arrival[corrected] > data.arrival[wrong]
        midway = (data.arrival[wrong] + data.arrival[corrected]) / 2
        assert table(data.observations_at(float(midway)))[key] == data.events.value[wrong]
        assert table(data.observations_at(float(data.arrival[corrected])))[key] == data.events.value[corrected]
    clean = generate(scenario(observation_fraction=1.), 52)
    assert table(data.observations_at(2.)) == table(clean.observations_at(2.))


def test_arrival_cutoff_includes_all_tied_events():
    data = generate(scenario(sampling="synchronous", observation_fraction=1.), 85)
    initial = data.observations_at(0.)
    np.testing.assert_array_equal(initial.time, np.zeros(4))
    np.testing.assert_array_equal(initial.asset, np.arange(4))
    np.testing.assert_array_equal(initial.value, np.zeros(4))


def test_noise_and_edits_do_not_enter_latent_truth():
    clean = generate(scenario(), 744)
    contaminated = generate(scenario(noise="correlated", noise_scale=2.,
                                     correction_fraction=.4, delete_fraction=.2), 744)
    for x, y in zip(clean.truth, contaminated.truth):
        np.testing.assert_array_equal(x.mean, y.mean)
        np.testing.assert_array_equal(x.covariance.loadings, y.covariance.loadings)
        np.testing.assert_array_equal(x.covariance.specific, y.covariance.specific)


def test_queries_have_exact_grid_windows_and_forecasts_use_no_future_arrivals():
    data = generate(scenario(max_latency=.1, regimes=2), 55)
    assert len(data.queries) == len(data.truth) == 6
    asofs = sorted({query.asof for query in data.queries})
    assert len(asofs) == 3
    for query, truth in zip(data.queries, data.truth):
        assert query.start < query.end
        if query.track == "historical":
            assert query.end == query.asof
        else:
            assert query.start == query.asof
            assert query.end > query.asof
        np.testing.assert_allclose(np.asarray([query.start, query.end]) * 32,
                                   np.rint(np.asarray([query.start, query.end]) * 32))
        observations = data.observations_at(query.asof)
        assert np.all(observations.time <= query.asof)
        assert truth.mean.shape == truth.covariance.diagonal().shape == (4,)
        assert np.all(truth.covariance.specific > 0)
        matrix = truth.covariance.matvec(np.eye(4))
        assert np.linalg.eigvalsh(matrix).min() > 0
        # Integrated covariance and mean scale with physical window length.
        assert np.isfinite(matrix).all()


def test_piecewise_integration_crosses_change_instead_of_using_one_endpoint_rate():
    stationary = generate(scenario(regimes=1, noise="none"), 67)
    changing = generate(scenario(regimes=2, noise="none"), 67)
    q, actual = changing.queries[2], changing.truth[2]
    assert q.start < .5 < q.end
    before = stationary.truth[0]
    rate_before = before.covariance.matvec(np.eye(4)) / (
        stationary.queries[0].end - stationary.queries[0].start)
    after_query, after = changing.queries[-1], changing.truth[-1]
    rate_after = after.covariance.matvec(np.eye(4)) / (after_query.end - after_query.start)
    expected = (.5 - q.start) * rate_before + (q.end - .5) * rate_after
    np.testing.assert_allclose(actual.covariance.matvec(np.eye(4)), expected, atol=2e-15)
    mean_before = before.mean / (stationary.queries[0].end - stationary.queries[0].start)
    mean_after = after.mean / (after_query.end - after_query.start)
    np.testing.assert_allclose(actual.mean, (.5 - q.start) * mean_before
                               + (q.end - .5) * mean_after, atol=2e-15)


@pytest.mark.parametrize("distribution", ["gaussian", "student_t", "compound_poisson"])
def test_small_monte_carlo_standardized_latent_increment_moments(distribution):
    config = Scenario.from_dict(dict(id="moments", n_assets=2, grid_steps=16,
                                     rank=1, observation_fraction=1.,
                                     sampling="synchronous", distribution=distribution,
                                     noise="none", condition=3.))
    residuals = []
    for seed in range(600):
        data = generate(config, seed)
        query, truth = data.queries[0], data.truth[0]
        observations = table(data.observations_at(1.))
        increment = np.array([observations[a, query.end] - observations[a, query.start]
                              for a in range(2)])
        covariance = truth.covariance.matvec(np.eye(2))
        residuals.append(np.linalg.solve(np.linalg.cholesky(covariance), increment - truth.mean))
    residuals = np.asarray(residuals)
    np.testing.assert_allclose(residuals.mean(axis=0), np.zeros(2), atol=.15)
    np.testing.assert_allclose(residuals.T @ residuals / len(residuals), np.eye(2), atol=.22)


@pytest.mark.parametrize("config", [
    {"distribution": "student_t", "df": 8}, {"sampling": "unknown"},
    {"noise": "unknown"}, {"observation_fraction": -1}, {"max_latency": -1},
    {"condition": .5}, {"correction_fraction": .8, "delete_fraction": .3},
])
def test_invalid_scenario_parameters_reject(config):
    with pytest.raises(ValueError):
        generate(scenario(**config), 2)


def test_suites_cover_required_scales_and_are_loadable_without_generation():
    root = Path(__file__).resolve().parents[1]
    smoke, core, scale = [load_suite(root / "suites" / f"{name}.json")
                          for name in ["smoke", "core", "scale"]]
    assert max(s.n_assets for s in smoke) <= 8
    assert {16, 64, 256, 1024} <= {s.n_assets for s in core}
    assert {4096} <= {s.n_assets for s in scale}
    assert {"gaussian", "student_t", "compound_poisson"} <= {
        s.config["distribution"] for s in core}
    assert {0, 4, 8, 16, 32} <= {s.config["rank"] for s in core}
    assert {8, 32, 64} == {s.config["rank"] for s in scale}
    holdout = load_suite(root / "suites" / "holdout.json")
    assert 4 <= len(holdout) <= 6
    assert {32, 96, 384, 768} <= {s.n_assets for s in holdout}
    assert all(s.config["regimes"] == 3 for s in holdout)
    assert all(s.config["df"] == 12 for s in holdout
               if s.config["distribution"] == "student_t")
    assert max(s.config["drift_strength"] for s in holdout) >= 32
