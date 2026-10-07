import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from local_agent.agent.tolerance_calibration import calibrate_tolerances, apply_calibrated_tolerances, stable_losses, state_scaled_atol
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


def setup_case(tmp_path, monkeypatch, failed=False, settings=None, candidates=2, scales=None, gradient_settings=None, bad_scale_loss=False, fail_scaled=False, fail_de=False):
    from diffrax import RESULTS
    settings = settings or {}
    folder = tmp_path/'generated'
    folder.mkdir()
    (tmp_path/'inputs').mkdir()
    gradient_settings = gradient_settings or {}
    gradient = dict(max_steps=20000, stepsize_rtol=[1e-7], stepsize_atol=[1e-9], **gradient_settings)
    (tmp_path/'inputs/user_input.yaml').write_text(yaml.safe_dump(dict(gradient_opt=gradient, population_opt=settings)))
    (folder/'user_model.py').write_text('# source')
    (folder/'solver_coverage.json').write_text(json.dumps([dict(candidates=[
        dict(sample=i, successful=True, normalized_parameters=[i], experiment_losses=[i+1.], experiment_state_scales=None if scales is None else [scales])
        for i in range(candidates)])]))
    reader = SimpleNamespace(max_steps=20000, stepsize_rtol=[1e-7], stepsize_atol=[1e-9],
        trainable_parameter_names=["k"], min_axis_values=[0.], max_axis_values=[1.], axis_logscale=[False], error_loss=1e10)
    monkeypatch.setattr('lib.utils.experiments.experiment_constants', lambda record, reader: dict(max_steps=reader.max_steps))
    calls = []
    def integrate(c, p):
        assert c['max_steps'] == 20000
        if scales is None:
            np.testing.assert_allclose(c['stepsize_rtol'], [1e-6])
            np.testing.assert_allclose(c['stepsize_atol'], [1e-8])
        calls.append(dict(rtol=c['stepsize_rtol'].copy(), atol=c['stepsize_atol'].copy()))
        return None, np.ones((2, 1)), RESULTS.max_steps_reached if failed or (fail_scaled and c['stepsize_rtol'][0] == 1e-7) or (fail_de and c['stepsize_rtol'][0] > 1e-7) else RESULTS.successful, dict(num_steps=3)
    module = SimpleNamespace(_integrate_system_with_stats=integrate,
                             _compute_loss_problem=lambda c,p: (p[0]+1.)*(1.1 if bad_scale_loss and c['stepsize_rtol'][0] == 1e-7 else 1.001))
    script = folder/'generated_script.py'
    result = calibrate_tolerances(module, reader, script, [dict(index=1)], settings, gradient_settings)
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


def test_scale_aggregation_robust_across_candidates_and_experiments():
    candidates = [dict(experiment_state_scales=v) for v in [
        [[1000., 1e-6, 0.], [2000., 2e-6, 0.]],
        [[1000., 1e-6, 0.], [2000., 2e-6, 0.]],
        [[1e12, 1e6, 0.], [1e12, 1e6, 0.]],
    ]]
    result = state_scaled_atol(candidates, [1e-7], [1e-9])
    np.testing.assert_allclose(result['state_scales'], [2000., 2e-6, 0.], atol=0)
    np.testing.assert_allclose(result['proposed_atol'], [2e-6, 2e-15, 1e-9], atol=0)
    assert result['fallback_state_indices'] == [2]


def test_scaled_refinement_then_de_uses_selected_scales(tmp_path, monkeypatch):
    result, calls, script, gradient = setup_case(tmp_path, monkeypatch, scales=[1000., 1e-6, 0.])
    assert result['state_scaling']['status'] == 'accepted'
    assert len(calls) == 4
    np.testing.assert_allclose(calls[0]['atol'], [1e-6, 1e-15, 1e-9], atol=0)
    np.testing.assert_allclose(calls[2]['atol'], [1e-5, 1e-14, 1e-8], atol=0)
    assert apply_calibrated_tolerances(tmp_path, script)
    config = yaml.safe_load((tmp_path/'inputs/user_input.yaml').read_text())
    assert config['gradient_opt']['max_steps'] == gradient['max_steps']
    assert config['gradient_opt']['stepsize_rtol'] == gradient['stepsize_rtol']
    np.testing.assert_allclose(config['gradient_opt']['stepsize_atol'], [1e-6, 1e-15, 1e-9], atol=0)


@pytest.mark.parametrize('options', [{'bad_scale_loss':True}, {'fail_scaled':True}])
def test_unstable_state_scaling_falls_back_before_de(tmp_path, monkeypatch, options):
    result, calls, script, gradient = setup_case(tmp_path, monkeypatch, scales=[1000.], **options)
    assert result['state_scaling']['status'] == 'retained'
    assert 'gradient_stepsize_atol' not in result
    np.testing.assert_allclose(calls[-1]['atol'], [1e-8], atol=0)
    assert apply_calibrated_tolerances(tmp_path, script)
    assert yaml.safe_load((tmp_path/'inputs/user_input.yaml').read_text())['gradient_opt'] == gradient


@pytest.mark.parametrize('population', [{'auto_tolerances':False}, {'stepsize_atol':[1e-5]}])
def test_state_scaling_independent_of_population_override(tmp_path, monkeypatch, population):
    result, calls, script, _ = setup_case(tmp_path, monkeypatch, scales=[1000.], settings=population)
    assert result['state_scaling']['status'] == 'accepted'
    assert len(calls) == 2
    assert apply_calibrated_tolerances(tmp_path, script)
    config = yaml.safe_load((tmp_path/'inputs/user_input.yaml').read_text())
    assert config['population_opt'] == population
    np.testing.assert_allclose(config['gradient_opt']['stepsize_atol'], [1e-6], atol=0)


def test_state_scaling_opt_out(tmp_path, monkeypatch):
    result, calls, script, gradient = setup_case(tmp_path, monkeypatch, scales=[1000.],
        gradient_settings={'auto_state_tolerances':False})
    assert result['state_scaling']['status'] == 'retained'
    assert len(calls) == 2
    np.testing.assert_allclose(calls[0]['atol'], [1e-8], atol=0)
    assert apply_calibrated_tolerances(tmp_path, script)
    assert yaml.safe_load((tmp_path/'inputs/user_input.yaml').read_text())['gradient_opt'] == gradient


def test_state_scaling_flag_requires_boolean():
    with pytest.raises(ValueError, match='auto_state_tolerances'):
        _validate_raw_settings({'gradient_opt':{'auto_state_tolerances':'true'}})


def test_scaled_refinement_survives_failed_de_comparison(tmp_path, monkeypatch):
    result, calls, script, _ = setup_case(tmp_path, monkeypatch, scales=[1000.], fail_de=True)
    assert result['state_scaling']['status'] == 'accepted'
    assert result['de_status'] == 'retained'
    assert result['status'] == 'accepted'
    assert len(calls) == 3
    assert apply_calibrated_tolerances(tmp_path, script)
    config = yaml.safe_load((tmp_path/'inputs/user_input.yaml').read_text())
    assert config['population_opt'] == {}
    np.testing.assert_allclose(config['gradient_opt']['stepsize_atol'], [1e-6], atol=0)
