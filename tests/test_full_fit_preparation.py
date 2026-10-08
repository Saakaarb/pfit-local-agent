import copy
import pytest
from scripts.run_qwen38_cpu_fits import full_fit_config, check_optimizer_only_change


def original():
    return {'population_opt': {'num_iters': 1, 'stepsize_rtol': [1e-6]},
            'gradient_opt': {'num_iters': 5, 'integrator': 'Kvaerno5', 'max_steps': 7000,
                             'stepsize_atol': [1e-9]},
            'model': {'trainable_parameters': [{'name': 'k', 'bounds': [1, 10]}]},
            'experiments': [{'data_file': 'data.csv'}]}


@pytest.mark.parametrize('dimensions,population', [(1,64),(9,80),(20,160),(50,160)])
def test_full_fit_budgets_and_lr_endpoint(dimensions, population):
    import optax
    source = original(); before = copy.deepcopy(source)
    result = full_fit_config(source, dimensions, 16)
    assert source == before
    assert result['population_opt']['population_size'] == population
    assert result['population_opt']['num_iters'] == 100
    assert result['population_opt']['processors'] == 16
    g = result['gradient_opt']
    assert g['num_iters'] == 3000
    lr = optax.exponential_decay(g['init_value_lr'], g['transition_steps_lr'],
                                g['decay_rate_lr'], end_value=g['end_value_lr'])
    assert float(lr(0)) == pytest.approx(1e-4)
    assert float(lr(2999)) == pytest.approx(1e-6)
    assert g['max_steps'] == 7000
    assert g['integrator'] == 'Kvaerno5'
    assert g['stepsize_atol'] == [1e-9]


@pytest.mark.parametrize('section,key,value', [('gradient_opt','max_steps',50000),
                                             ('gradient_opt','stepsize_atol',[1e-5]),
                                             ('model','trainable_parameters',[])])
def test_rebinding_rejects_scientific_and_solver_changes(section,key,value):
    source=original(); updated=full_fit_config(source, 5, 16)
    updated[section][key]=value
    with pytest.raises(ValueError, match='scientific or solver'):
        check_optimizer_only_change(source, updated)
