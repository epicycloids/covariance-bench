import json
from pathlib import Path

import numpy as np
import pytest

from covbench.runner import evaluate, read_prediction, regrade
from covbench.scenarios import Scenario, generate
from covbench.provenance import derived_seed
from covbench.worker import materialize
from covbench.api import DenseMatrix, Prediction


def make_candidate(path, *, slow_fit=False, deferred=False):
    path.mkdir()
    (path / "candidate.json").write_text(json.dumps(dict(name="test", api_version=1, entrypoint="candidate:create", config={})))
    code = '''
import time
import numpy as np
from covbench.api import Prediction, FactorMatrix
class Candidate:
    def __init__(self, problem):
        assert set(vars(problem)) == {'n_assets', 'mode', 'time_unit'}
        self.n = problem.n_assets
        self.live = {}
    def fit(self, obs):
        FIT_DELAY
        self.live = {(int(a),float(t)):float(v) for a,t,v in zip(obs.asset,obs.time,obs.value)}
    def update(self, events):
        for a,t,v,op in zip(events.asset,events.time,events.value,events.operation):
            key = (int(a),float(t))
            if op == 0: self.live[key] = float(v)
            else: self.live.pop(key,None)
    def predict(self, query):
        PREDICT_DELAY
        assert all(t <= query.asof for a,t in self.live)
        print('candidate stdout is redirected')
        mean = np.full(self.n, sum(self.live.values()))
        return Prediction(mean, FactorMatrix(np.empty((self.n,0)),np.ones(self.n)))
    def synchronize(self): pass
def create(problem, config): return Candidate(problem)
'''
    code = code.replace("FIT_DELAY", "time.sleep(2.)" if slow_fit else "pass")
    code = code.replace("PREDICT_DELAY", "time.sleep(.03)" if deferred else "pass")
    (path / "candidate.py").write_text(code)
    return path


def make_suite(path):
    scenario = dict(id="causal", n_assets=3, grid_steps=32, observation_fraction=.6,
                    max_latency=.08, correction_fraction=.2, delete_fraction=.1)
    path.write_text(json.dumps(dict(version=1, scenarios=[scenario])))
    return path, Scenario.from_dict(scenario)


def test_causal_offline_online_measured_outputs_and_freezing(tmp_path):
    candidate = make_candidate(tmp_path / "candidate")
    suite, scenario = make_suite(tmp_path / "suite.json")
    output = tmp_path / "run"
    report = evaluate(candidate, suite, output, root_seed=123, replications=1, timing_repeats=1, batch_size=7)
    assert len(report["records"]) == 12
    assert len(report["trials"]) == 2
    assert all(r["failures"] == 0 for r in report["records"])
    dataset = generate(scenario, derived_seed(123, scenario.id, 0))
    for trial in report["trials"]:
        for record in trial["records"]:
            asof = record["query"]["asof"]
            expected = dataset.observations_at(asof).value.sum()
            prediction = read_prediction(output / trial["folder"] / record["output"])
            np.testing.assert_allclose(prediction.mean, expected, atol=1e-12)
            assert record["completed_ns"] >= record["predict_ns"]
        log = (output / trial["folder"] / "worker.log").read_text()
        assert "stdout is redirected" in log
    assert report["trials"][0]["inputs_sha256"] == report["trials"][1]["inputs_sha256"]
    assert (output / "candidate" / "candidate.py").read_bytes() == (candidate / "candidate.py").read_bytes()
    rescored = regrade(output / "results.json", output / "regraded.json")
    for old, new in zip(report["records"], rescored["records"]):
        assert old["covariance_nmse"] == new["covariance_nmse"]
        assert old["completed_ns"] == new["completed_ns"]
    assert rescored["measurement_evaluator_sha256"] == report["evaluator_sha256"]
    first = report["trials"][0]
    predicted = output / first["folder"] / first["records"][0]["output"]
    predicted.write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction changed"):
        regrade(output / "results.json", output / "tampered.json")


def test_deferred_computation_is_in_completed_work(tmp_path):
    candidate = make_candidate(tmp_path / "candidate", deferred=True)
    suite, _ = make_suite(tmp_path / "suite.json")
    report = evaluate(candidate, suite, tmp_path / "run", root_seed=1, replications=1,
                      timing_repeats=1, modes=["online"])
    assert all(r["completed_ns"] > 50_000_000 for r in report["records"])
    assert all(r["predict_ns"] > 20_000_000 for r in report["records"])


def test_timeouts_are_complete_failure_records(tmp_path):
    candidate = make_candidate(tmp_path / "candidate", slow_fit=True)
    suite, _ = make_suite(tmp_path / "suite.json")
    report = evaluate(candidate, suite, tmp_path / "run", root_seed=1, replications=1,
                      timing_repeats=1, modes=["offline"], timeout=.3)
    assert len(report["trials"][0]["records"]) == 6
    assert all(r["status"] == "timeout" for r in report["trials"][0]["records"])
    assert all(r["status"] == "failed" and r["completed_ns"] is None for r in report["records"])


@pytest.mark.parametrize("mean", [[0., 0.], np.zeros(2, dtype=complex)])
def test_worker_rejects_wrong_array_contract_instead_of_casting_it(mean):
    with pytest.raises(TypeError, match="real host"):
        materialize(Prediction(mean, DenseMatrix(np.eye(2))), 2)
