import numpy as np
import pytest

from covbench.api import DELETE, UPSERT, Events, FactorMatrix, Observations, Problem, Query
from covbench.baselines import create_async_native, create_previous_tick, create_zero


def observations(times, values):
    return Observations(np.asarray(times, float), np.zeros(len(times), dtype=int), np.asarray(values, float))


def covariance(prediction):
    return prediction.covariance.matvec(np.eye(len(prediction.mean)))


def assert_same(left, right):
    np.testing.assert_allclose(left.mean, right.mean)
    np.testing.assert_allclose(covariance(left), covariance(right))


@pytest.mark.parametrize("factory", [create_zero, create_previous_tick])
def test_forecast_uses_only_prefix_and_dilates_full_moments(factory):
    estimator = factory(Problem(1, "offline"), {"lookback": 1.0, "bins": 5})
    prefix = observations([0, 0.25, 0.5, 0.75, 1], [0, 0.2, 0.8, 0.9, 1.7])
    estimator.fit(prefix)
    first = estimator.predict(Query("forecast", 1, 1, 2))
    twice = estimator.predict(Query("forecast", 1, 1, 3))
    np.testing.assert_allclose(twice.mean, 2 * first.mean)
    np.testing.assert_allclose(covariance(twice), 2 * covariance(first))
    second = covariance(first) + np.outer(first.mean, first.mean)
    expected = 2 * second + 2 * np.outer(first.mean, first.mean)
    np.testing.assert_allclose(covariance(twice) + np.outer(twice.mean, twice.mean), expected)
    estimator.fit(observations([*prefix.time, 1.1, 2], [*prefix.value, 1e8, -1e8]))
    assert_same(first, estimator.predict(Query("forecast", 1, 1, 2)))


def test_previous_tick_revisions_match_current_table_and_fit_resets():
    estimator = create_previous_tick(Problem(1, "online"), {"bins": 5, "lookback": 1.0})
    estimator.fit(observations([0, 0.25, 0.5, 0.75, 1], [0, 1, 5, 2, 4]))
    query = Query("historical", 1, 0, 1)
    original = estimator.predict(query)
    estimator.update(Events(np.array([0.5]), np.array([0]), np.array([1.5]), np.array([UPSERT])))
    revised = estimator.predict(query)
    assert not np.allclose(covariance(original), covariance(revised))
    estimator.update(Events(np.array([0.25]), np.array([0]), np.array([np.nan]), np.array([DELETE])))
    fresh = create_previous_tick(Problem(1, "offline"), {"bins": 5})
    fresh.fit(observations([0, 0.5, 0.75, 1], [0, 1.5, 2, 4]))
    assert_same(estimator.predict(query), fresh.predict(query))
    estimator.fit(observations([0, 1], [2, 2]))
    np.testing.assert_allclose(estimator.predict(query).mean, 0)


def test_previous_tick_does_not_backfill_first_price_from_future():
    estimator = create_previous_tick(Problem(1, "offline"), {"bins": 5, "floor": 0})
    estimator.fit(observations([0.75, 1], [5, 6]))
    result = estimator.predict(Query("historical", 1, 0, 1))
    np.testing.assert_allclose(result.mean, 4)
    np.testing.assert_allclose(covariance(result), 0)
    assert isinstance(result.covariance, FactorMatrix)
    assert result.covariance.loadings.shape == (1, 4)


def test_previous_tick_multivariate_sample_covariance_rate():
    times = np.linspace(0, 1, 5)
    prices = np.array([[0, 1, 1.5, 1.2, 2], [0, 0.1, 0.6, 1.1, 1]])
    estimator = create_previous_tick(Problem(2, "offline"), {"bins": 5, "floor": 0})
    estimator.fit(Observations(np.tile(times, 2), np.repeat([0, 1], 5), prices.ravel()))
    result = estimator.predict(Query("historical", 1, 0, 1))
    returns = np.diff(prices, axis=1)
    np.testing.assert_allclose(result.mean, prices[:, -1] - prices[:, 0])
    np.testing.assert_allclose(covariance(result), np.cov(returns) / 0.25)


def test_native_optional_prefix_revisions_and_duration():
    pytest.importorskip("async_covariance")
    estimator = create_async_native(Problem(1, "online"), {"lookback": 1.0})
    estimator.fit(observations([0, 0.5, 1, 2], [0, 0.8, 2, 1e8]))
    query = Query("forecast", 1, 1, 2)
    first = estimator.predict(query)
    np.testing.assert_allclose(first.mean, 2)
    doubled = estimator.predict(Query("forecast", 1, 1, 3))
    np.testing.assert_allclose(doubled.mean, 2 * first.mean)
    np.testing.assert_allclose(covariance(doubled), 2 * covariance(first))
    estimator.update(Events(np.array([0.5, 0]), np.array([0, 0]), np.array([1, np.nan]), np.array([UPSERT, DELETE])))
    fresh = create_async_native(Problem(1, "offline"), {"lookback": 1.0})
    fresh.fit(observations([0.5, 1], [1, 2]))
    assert_same(estimator.predict(query), fresh.predict(query))


def test_native_large_universe_uses_factor_and_fit_errors_propagate(monkeypatch):
    native = pytest.importorskip("async_covariance")
    def forbid_dense(snapshot):
        raise AssertionError("large native baseline must not materialize dense covariance")
    monkeypatch.setattr(native.MomentSnapshot, "dense", forbid_dense)
    n = 257
    times = np.repeat([0, 0.5, 1], n)
    assets = np.tile(np.arange(n), 3)
    values = times * (1 + assets / n) + (times == 0.5) * 0.1
    estimator = create_async_native(Problem(n, "offline"), {"max_iter": 2})
    estimator.fit(Observations(times, assets, values))
    query = Query("historical", 1, 0, 1)
    result = estimator.predict(query)
    assert isinstance(result.covariance, FactorMatrix)
    assert result.covariance.loadings.shape == (n, 8)
    assert np.all(np.isfinite(result.covariance.diagonal()))
    assert np.all(result.covariance.specific > 0)
    def fail_fit(*args, **kwargs):
        raise ValueError("deliberate native fit failure")
    estimator.native_fit = fail_fit
    with pytest.raises(ValueError, match="deliberate native fit failure"):
        estimator.predict(query)
