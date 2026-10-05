import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from local_agent.agent.tolerance_calibration import calibrate_tolerances, apply_calibrated_tolerances, stable_losses
from local_agent.agent.validators import _validate_raw_settings


@pytest.mark.parametrize('reference,proposed,accepted', [
    ([[1.], [2.]], [[1.005], [2.01]], True),
    ([[1.], [2.]], [[1.02], [2.]], False),
    ([[1.], [1.015]], [[1.01], [1.005]], False),
    ([[0.], [1.]], [[1e-9], [1.]], True),
    ([[1., 100.], [2., 200.]], [[2., 99.], [2., 200.]], False),
    ([[1.], [2.]], [[float('nan')], [2.]], False),
])
def test_loss_and_order_guards(reference, proposed, accepted):
    assert stable_losses(reference, proposed)[0] == accepted


def setup_case(tmp_path, monkeypatch, failed=False, settings=None, candidates=2):
    from diffrax import RESULTS
    settings = settings or {}
    folder = tmp_path/'generated'
    folder.mkdir()
    (tmp_path/'inputs').mkdir()
    gradient = dict(max_steps=20000, stepsize_rtol=[1e-7], stepsize_atol=[1e-9])
    (tmp_path/'inputs/user_input.yaml').write_text(yaml.safe_dump(dict(gradient_opt=gradient, population_opt=settings)))
    (folder/'user_model.py').write_text('# source')
    (folder/'solver_coverage.json').write_text(json.dumps([dict(candidates=[
        dict(sample=i, successful=True, normalized_parameters=[i], experiment_losses=[i+1.])
        for i in range(candidates)])]))
    reader = SimpleNamespace(max_steps=20000, stepsize_rtol=[1e-7], stepsize_atol=[1e-9],
        trainable_parameter_names=["k"], min_axis_values=[0.], max_axis_values=[1.], axis_logscale=[False], error_loss=1e10)
    monkeypatch.setattr('lib.utils.experiments.experiment_constants', lambda record, reader: dict(max_steps=reader.max_steps))
    calls = []
    def integrate(c, p):
        assert c['max_steps'] == 20000
        np.testing.assert_allclose(c['stepsize_rtol'], [1e-6])
        np.testing.assert_allclose(c['stepsize_atol'], [1e-8])
        calls.append(p.copy())
        return None, np.ones((2, 1)), RESULTS.max_steps_reached if failed else RESULTS.successful, dict(num_steps=3)
    module = SimpleNamespace(_integrate_system_with_stats=integrate,
                             _compute_loss_problem=lambda c,p: (p[0]+1.)*1.001)
    script = folder/'generated_script.py'
    result = calibrate_tolerances(module, reader, script, [dict(index=1)], settings)
    return result, calls, script, gradient


def test_accept_and_apply_only_population_settings(tmp_path, monkeypatch):
    result, calls, script, gradient = setup_case(tmp_path, monkeypatch)
    assert result['status'] == 'accepted'
    assert len(calls) == 2  # strict losses reused
    assert apply_calibrated_tolerances(tmp_path, script)
    raw = yaml.safe_load((tmp_path/'inputs/user_input.yaml').read_text())
    assert raw['gradient_opt'] == gradient
    np.testing.assert_allclose(raw['population_opt']['stepsize_rtol'], [1e-6])
    assert not apply_calibrated_tolerances(tmp_path, script)


def test_failed_loose_solve_retains_strict_without_recovery(tmp_path, monkeypatch):
    result, calls, script, _ = setup_case(tmp_path, monkeypatch, failed=True)
    assert result['status'] == 'retained'
    assert len(calls) == 1
    assert not apply_calibrated_tolerances(tmp_path, script)


@pytest.mark.parametrize('settings,candidates', [({'auto_tolerances':False},2), ({'stepsize_rtol':[1e-5]},2), ({},1)])
def test_preserve_overrides_and_skip_insufficient_points(tmp_path, monkeypatch, settings, candidates):
    result, calls, _, _ = setup_case(tmp_path, monkeypatch, settings=settings, candidates=candidates)
    assert result['status'] == 'retained'
    assert not calls


def test_changed_sources_prevent_applying_report(tmp_path, monkeypatch):
    _, _, script, _ = setup_case(tmp_path, monkeypatch)
    (tmp_path/'generated/user_model.py').write_text('# changed')
    assert not apply_calibrated_tolerances(tmp_path, script)


def test_auto_setting_requires_boolean():
    with pytest.raises(ValueError, match='auto_tolerances'):
        _validate_raw_settings({'population_opt':{'auto_tolerances':'true'}})
