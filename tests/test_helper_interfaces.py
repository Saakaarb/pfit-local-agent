import json

import pytest

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.jax_fragments import JaxFragments
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.validators import ValidationError
from local_agent.agent.workflow import LocalWorkflow, _allowed_repair_fields, _validate_helper_interfaces
from local_agent.llm.fake import FakeLLMClient
from tests.test_workflow import make_session, VALID_RHS

SOURCE_HELPER = 'def _observables(solution, trainable_parameters, fixed_parameters):\n    return solution[:, 0]'
BAD_HELPER = 'def _observables(solution, mu):\n    return solution[:, 0]'


def fragments(helper, call='_observables(solution, trainable_parameters, fixed_parameters)'):
    return JaxFragments(rhs=('x2', '-mu*x1'), helper_functions=(helper,),
                        loss_body='return ' + call, writeout_body='return solution')


def test_source_dictionary_signature_cannot_be_replaced_by_scalars():
    with pytest.raises(ValidationError, match='helper _observables signature changed'):
        _validate_helper_interfaces(fragments(BAD_HELPER), SOURCE_HELPER)
    _validate_helper_interfaces(fragments(SOURCE_HELPER), SOURCE_HELPER)


def test_missing_call_argument_is_detected_before_runtime():
    with pytest.raises(ValidationError, match='helper _observables call is incompatible'):
        _validate_helper_interfaces(fragments(SOURCE_HELPER, '_observables(solution)'), SOURCE_HELPER)


def test_keyword_defaults_and_positional_only_arguments():
    helper = 'def convert(x, /, scale=1, *, offset=0):\n    return x * scale + offset'
    _validate_helper_interfaces(fragments(helper, 'convert(solution, offset=2)'), helper)
    with pytest.raises(ValidationError, match='call is incompatible'):
        _validate_helper_interfaces(fragments(helper, 'convert(x=solution)'), helper)


def test_runtime_helper_argument_error_takes_precedence_over_loss_context():
    response = json.dumps({'helper_functions': [BAD_HELPER], 'loss_body': 'return 0'})
    assert _allowed_repair_fields(response, "loss smoke test: _observables() missing 1 required positional argument: 'mu'") == {'helper_functions'}
    assert _allowed_repair_fields(response, 'loss returned NaN') == {'loss_body'}


def test_bounded_helper_repair_preserves_deterministic_callers(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    generated = session / 'generated'
    generated.mkdir()
    source = 'import numpy as np\n' + SOURCE_HELPER + '''

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    return np.array([y[1], -trainable_parameters['mu'] * y[0]])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    prediction = _observables(solution, trainable_parameters, fixed_parameters)
    return np.mean((prediction - dataset[:, 0]) ** 2)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return solution
'''
    (generated / 'user_model.py').write_text(source)
    llm = FakeLLMClient([json.dumps({'helper_functions': [BAD_HELPER]}), VALID_RHS,
                         json.dumps({'helper_functions': [SOURCE_HELPER]})])
    workflow = LocalWorkflow(llm, PromptRenderer(), WorkflowConfig(max_repair_attempts=1))
    monkeypatch.setattr(workflow, '_validate_generated_script', lambda *args: True)
    result = workflow.generate_script(session)
    assert result.success
    assert len(llm.requests) == 3
    repair = '\n'.join(message.content for message in llm.requests[-1])
    assert 'Allowed repair fields: helper_functions' in repair
    script = (generated / 'generated_script.py').read_text()
    assert 'prediction = _observables(solution, trainable_parameters, fixed_parameters)' in script
    assert 'return jnp.mean((prediction - dataset[:, 0]) ** 2)' in script
    assert any('signature changed' in event.message for event in result.events)
