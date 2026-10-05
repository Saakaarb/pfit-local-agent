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


def check(tmp_path, monkeypatch, success, target=.25, experiments=1, loss=1.):
    from diffrax import RESULTS
    records = [dict(index=i+1, filename=f'{i}.csv', t_eval=np.array([0.,1.])) for i in range(experiments)]
    monkeypatch.setattr('lib.utils.experiments.experiment_constants',
                        lambda record, reader: dict(index=record['index'], init_time=0.))
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
        dict(solver_validation_samples=32, solver_validation_success_fraction=target))
    return result, calls


def test_fraction_suffices_and_all_samples_are_measured(tmp_path, monkeypatch):
    points, calls = check(tmp_path, monkeypatch, lambda i,e: 1 <= i <= 8)
    assert len(calls) == 32
    assert len(points) == 2
    assert not np.array_equal(points[0], np.zeros(2))  # midpoint need not pass
    report = json.loads((tmp_path/'solver_coverage.json').read_text())[-1]
    assert report['successful_samples'] == 8
    assert report['success_fraction'] == .25
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
    ('solver_validation_max_samples',16),('solver_validation_seed',-1),
    ('solver_validation_success_fraction',0),('solver_validation_success_fraction',1.1),
    ('solver_validation_success_fraction',True),('solver_validation_success_fraction',float('nan')),
    ('solver_recovery_stagnation_patience',0)])
def test_bad_policy_rejected(field,value):
    with pytest.raises(ValueError):
        _validate_raw_settings({'gradient_opt':{field:value}})
