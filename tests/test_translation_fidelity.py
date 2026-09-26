"""User functions survive translation; numerical counterexamples drive repair."""
import json
from pathlib import Path
import numpy as np
import pytest
from local_agent.agent.session_init import init_session
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.workflow import LocalWorkflow, _deterministic_custom_loss_body
from local_agent.agent.config import WorkflowConfig
from local_agent.agent.validators import import_generated_script, parse_input_yaml
from local_agent.llm.fake import FakeLLMClient
from lib.utils.experiments import load_experiments, experiment_constants
from lib.utils.run_artifacts import parameter_axes
from lib.utils.source_stamp import verify_stamp

pytestmark=pytest.mark.unit


def source_session(tmp_path):
    from tests.test_cli import _new_session_response
    session=tmp_path/'session';(session/'inputs').mkdir(parents=True)
    t=np.linspace(0,1,6)
    np.savetxt(session/'inputs/data.csv',np.column_stack([t,10*np.exp(-t)]),delimiter=',',header='time,y',comments='')
    d=json.loads(_new_session_response());d['states'][0]['initial_value']=10.
    init_session(session,FakeLLMClient([json.dumps(d)]),PromptRenderer())
    path=session/'generated/user_model.py';source=path.read_text()
    start=source.index('def writeout_description')
    source=source[:start]+'''def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return np.column_stack((solution_time, solution[:, 0] - dataset[:, 0], 3 * dataset[:, 0]))
'''
    path.write_text(source)
    return session


def fragments(loss='return jnp.mean(jnp.square(solution[:, 0]-dataset[:, 0]))',rhs='-k*y',writeout=None):
    return json.dumps({'rhs':[rhs],'helper_functions':[], 'loss_body':loss,
        'writeout_body':writeout or 'return jnp.column_stack((solution_time, solution[:, 0]-dataset[:, 0], 3*dataset[:, 0]))', 'review':''})


def test_raw_mse_and_custom_output_preserved_independently(tmp_path):
    session=source_session(tmp_path)
    source=(session/'generated/user_model.py').read_text()
    assert 'return jnp.mean' in _deterministic_custom_loss_body(source)
    client=FakeLLMClient([json.dumps({'helper_functions':[]}),json.dumps({'rhs':['-k*y']})])
    result=LocalWorkflow(client,PromptRenderer()).generate_script(session)
    assert result.success
    report=json.loads((session/'generated/translation_fidelity.json').read_text())
    assert report['status']=='passed' and len(report['points'])==2
    module=import_generated_script(session/'generated/generated_script.py')
    reader=parse_input_yaml(session/'inputs/user_input.yaml');c=experiment_constants(load_experiments(session,reader)[0],reader)
    lo,hi,logs=parameter_axes(reader);c.update(min_limits=lo,max_limits=hi,is_logscale=logs)
    point=np.zeros(1);out=np.asarray(module._write_problem_result(c,point))
    assert out.shape==(6,3)
    np.testing.assert_allclose(out[:,2],3*c['dataset'][:,0])
    assert float(module._compute_loss_problem(c,point))==pytest.approx(np.mean(out[:,1]**2),rel=1e-10)


@pytest.mark.parametrize('bad', ['loss','rhs','writeout'])
def test_numerical_mismatch_enters_repair_loop(tmp_path,bad):
    session=source_session(tmp_path)
    kwargs={'loss':'return jnp.sqrt(jnp.mean(jnp.square(solution[:, 0]-dataset[:, 0])))'} if bad=='loss' else ({'rhs':'-2*k*y'} if bad=='rhs' else {'writeout':'return jnp.column_stack((solution_time, dataset, solution))'})
    # A malformed split reply enters the full-fragment repair path; the first repair
    # is executable but scientifically wrong, requiring a second targeted repair.
    client=FakeLLMClient(["not JSON",fragments(**kwargs),fragments()])
    result=LocalWorkflow(client,PromptRenderer(),WorkflowConfig(max_repair_attempts=2)).generate_script(session)
    assert result.success and len(client.requests)==3
    assert any(e.status=='failed' and 'fidelity' in e.message for e in result.events)
    assert 'differs from source Python' in client.requests[2][1].content
    assert verify_stamp(session)[0]


def test_unrepaired_loss_change_never_receives_source_stamp(tmp_path):
    session=source_session(tmp_path)
    client=FakeLLMClient(['not JSON',fragments(loss='return 0.0')])
    result=LocalWorkflow(client,PromptRenderer(),WorkflowConfig(max_repair_attempts=1)).generate_script(session)
    assert not result.success
    assert not verify_stamp(session)[0]
    assert json.loads((session/'generated/translation_fidelity.json').read_text())['status']=='failed'


def test_default_loss_does_not_replace_custom_writeout(tmp_path):
    from tests.test_workflow import make_session
    from local_agent.agent.user_model import render_user_model_skeleton
    from local_agent.agent.session_spec import load_session_spec
    session=make_session(tmp_path);(session/'generated').mkdir()
    spec=load_session_spec(session/'inputs/user_input.yaml')
    source=render_user_model_skeleton(spec)
    source=source[:source.index('def writeout_description')]+'''def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return np.column_stack((solution_time, 7 * solution[:, 0]))
'''
    (session/'generated/user_model.py').write_text(source)
    client=FakeLLMClient([json.dumps({'helper_functions':[]}),json.dumps({'rhs':['x2','-mu*x1']})])
    result=LocalWorkflow(client,PromptRenderer()).generate_script(session)
    assert result.success
    script=(session/'generated/generated_script.py').read_text()
    assert '7 * solution[:, 0]' in script
    report=json.loads((session/'generated/translation_fidelity.json').read_text())
    assert all('writeout' in p['components'] for p in report['points'])
    assert any('_compute_loss_problem' in x for x in report['limitations'])


def test_unmarked_constant_loss_is_preserved():
    body=_deterministic_custom_loss_body('def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):\n    return 0.0\n')
    assert body=='return 0.0'
