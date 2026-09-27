"""Configuration-to-optimizer regressions, using a differentiable objective."""
import numpy as np
import pytest
import yaml
import jax.numpy as jnp

from lib.algorithms.NODE.classes import FitParamsNODE
from local_agent.agent.validators import parse_input_yaml, ValidationError
from tests.test_session_validation import _minimal_yaml, _write_yaml

pytestmark = pytest.mark.unit


def reader_for(tmp_path, settings):
    config = yaml.safe_load(_minimal_yaml())
    config['gradient_opt'].pop('num_iters')
    config['gradient_opt'].update(settings)
    return parse_input_yaml(_write_yaml(tmp_path, yaml.safe_dump(config)))


def test_defaults_and_explicit_zero(tmp_path):
    reader = reader_for(tmp_path / 'default', {})
    assert reader.gradient_optimizer == 'adam'
    assert reader.n_iters_grad == 1000
    reader = reader_for(tmp_path / 'zero', {'num_iters': 0, 'gradient_optimizer': ' LBFGS '})
    assert reader.n_iters_grad == 0
    assert reader.gradient_optimizer == 'lbfgs'


@pytest.mark.parametrize('settings', [
    {'gradient_optimizer': 'sgd'}, {'gradient_optimizer': None},
    {'num_iters': -1}, {'num_iters': 1.5},
    {'init_value_lr': 0}, {'end_value_lr': float('nan')},
    {'transition_steps_lr': -2}, {'decay_rate_lr': float('inf')},
])
def test_invalid_settings_rejected(tmp_path, settings):
    with pytest.raises(ValidationError):
        reader_for(tmp_path, settings)


@pytest.mark.parametrize('name', ['adam', 'lbfgs'])
def test_selected_optimizer_reduces_loss_with_configured_budget(tmp_path, name):
    reader = reader_for(tmp_path, dict(gradient_optimizer=name, num_iters=20,
        init_value_lr=.05, end_value_lr=.01, transition_steps_lr=100, decay_rate_lr=.9))
    reader.output_dir = tmp_path
    class Problem:
        def _compute_loss(self, params):
            return jnp.sum((params - .25) ** 2)
    optimizer = FitParamsNODE(reader, Problem(), init_guess=np.array([1.]))
    initial = float(optimizer.compute_loss(optimizer.trainable_params))
    result, loss = optimizer.train_NODE()
    assert optimizer.optimizer_name == name
    assert len(optimizer.loss_history) == 20
    assert float(loss) < initial * .1
    assert np.all(np.abs(result) <= 1)
    if name == 'adam':
        assert float(optimizer.learning_rate(0)) == pytest.approx(.05)
        assert float(optimizer.learning_rate(100)) == pytest.approx(.045)


def test_nonfinite_gradient_explains_early_stop_to_user(tmp_path, capsys):
    reader = reader_for(tmp_path, dict(num_iters=5))
    reader.output_dir = tmp_path
    class Problem:
        def _compute_loss(self, params):
            # Finite forward value, undefined autodiff gradient.
            return 1 + jnp.sum(jnp.sqrt(jnp.maximum(params - params, 0)))
    optimizer = FitParamsNODE(reader, Problem(), init_guess=np.array([1.]))
    optimizer.train_NODE()
    assert optimizer.termination_reason == 'invalid_loss_or_gradient'
    assert 'non-finite gradient (NaN/Inf)' in optimizer.termination_detail
    output = capsys.readouterr().out
    assert 'gradient refinement did not complete' in output
    assert 'Keeping the best valid point' in output
    assert 'non-finite gradient' in (tmp_path / 'NODE_fitting.log').read_text()
