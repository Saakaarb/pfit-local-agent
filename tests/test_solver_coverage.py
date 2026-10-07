import json
from types import SimpleNamespace

import numpy as np
import pytest

from local_agent.agent.solver_coverage import parameter_samples, assess_solver_coverage
from local_agent.agent.validators import SolverValidationError, _validate_raw_settings


def test_nested_samples_reproducible_and_cover_bounds():
    small = parameter_samples(32, 4, 7)
    large = parameter_samples(128, 4, 7)
    np.testing.assert_array_equal(small, large[:32])
    np.testing.assert_array_equal(parameter_samples(64, 4, 7), large[:64])
    assert np.all(np.abs(large) <= 1)
    assert not np.array_equal(large, parameter_samples(128, 4, 19))
    # Each dimension of the first added block has one point in each stratum.
    for axis in range(4):
        np.testing.assert_array_equal(np.sort(((small[1:, axis]+1)*31/2).astype(int)), np.arange(31))


def check(tmp_path, monkeypatch, success, count=32, experiments=1, loss=1., initial_state=0.):
    from diffrax import RESULTS
    records = [dict(index=i+1, filename=f'{i}.csv', t_eval=np.array([0.,1.])) for i in range(experiments)]
    monkeypatch.setattr('lib.utils.experiments.experiment_constants',
                        lambda record, reader: dict(index=record['index'], init_time=0., init_cond=np.array([initial_state])))
    reader = SimpleNamespace(n_search_axes=2, trainable_parameter_names=['a','b'],
        min_axis_values=[1.,0.], max_axis_values=[100.,10.], axis_logscale=[True,False],
        integrator='test', max_steps=10000, error_loss=1e10)
    calls = []
    def integrate(constants, point):
        index = len(calls)//experiments
        calls.append(point.copy())
        result = RESULTS.successful if success(index, constants['index']) else RESULTS.max_steps_reached
        return np.array([0.,1.]), np.ones((2,1)), result, dict(num_steps=10)
    module = SimpleNamespace(_integrate_system_with_stats=integrate,
                             _compute_loss_problem=lambda c,p: loss)
    result = assess_solver_coverage(module, reader, tmp_path, tmp_path/'generated.py', records,
        dict(solver_validation_samples=count, solver_validation_success_fraction=1.0))
    return result, calls


@pytest.mark.parametrize("count", [32, 64, 128])
def test_ten_suffices_regardless_of_sample_set_size(tmp_path, monkeypatch, count):
    points, calls = check(tmp_path, monkeypatch, lambda i,e: 1 <= i <= 10, count=count)
    assert len(calls) == count
    assert len(points) == 2
    assert not np.array_equal(points[0], np.zeros(2))  # midpoint need not pass
    report = json.loads((tmp_path/'solver_coverage.json').read_text())[-1]
    assert report['successful_samples'] == 10
    assert report['required_successful_samples'] == 10
    assert report['success_fraction'] == 10/count
    assert report['candidates'][1]['experiment_state_scales'] == [[1.0]]
    detail = json.loads((tmp_path/'solver_diagnostics.json').read_text())[0]
    assert detail['physical_parameters'] == {'a':10., 'b':5.}


def test_success_requires_same_particle_to_complete_all_experiments(tmp_path, monkeypatch):
    with pytest.raises(SolverValidationError) as exc:
        check(tmp_path, monkeypatch, lambda i,e: (i < 16) == (e == 1), experiments=2)
    assert exc.value.diagnostics['successful_samples'] == 0
    assert exc.value.diagnostics['failure_counts']['step_limit'] == 32


def test_finite_integration_with_invalid_loss_does_not_count(tmp_path, monkeypatch):
    with pytest.raises(SolverValidationError) as exc:
        check(tmp_path, monkeypatch, lambda i,e: True, loss=np.nan)
    assert exc.value.diagnostics['failure_counts'] == {'invalid_loss':32}


@pytest.mark.parametrize('field,value', [('solver_validation_samples',0),('solver_validation_samples',True),
    ('solver_validation_samples',9),('solver_validation_seed',-1),
    ('solver_validation_min_successful',0),('solver_validation_min_successful',True),
    ('solver_validation_samples',-1)])
def test_bad_policy_rejected(field,value):
    with pytest.raises(ValueError):
        _validate_raw_settings({'gradient_opt':{field:value}})


def test_scale_includes_initial_state_even_when_saved_trajectory_has_decayed(tmp_path, monkeypatch):
    check(tmp_path, monkeypatch, lambda i,e: True, initial_state=1000.)
    report = json.loads((tmp_path/'solver_coverage.json').read_text())[-1]
    assert report['candidates'][0]['experiment_state_scales'] == [[1000.]]


@pytest.mark.parametrize('count', [32, 128])
def test_nine_successes_never_suffice(tmp_path, monkeypatch, count):
    with pytest.raises(SolverValidationError) as exc:
        check(tmp_path, monkeypatch, lambda i,e: i < 9, count=count)
    assert exc.value.diagnostics['required_successful_samples'] == 10


@pytest.mark.parametrize('dimensions,expected', [(1,32),(2,32),(4,64),(6,96),(8,128),(50,128)])
def test_one_sample_design_scales_with_search_dimensions(dimensions, expected):
    from local_agent.agent.solver_coverage import validation_sample_count
    assert validation_sample_count({}, dimensions) == expected
    # A legacy growth cap does not change the fixed design.
    assert validation_sample_count({'solver_validation_max_samples': 1}, dimensions) == expected


def test_explicit_sample_count_is_preserved():
    from local_agent.agent.solver_coverage import validation_sample_count
    assert validation_sample_count({'solver_validation_samples': 64}, 8) == 64
