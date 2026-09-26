"""Multi-experiment correctness; reference contract from deployed pfit-claude."""
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from lib.utils.experiments import load_experiments, experiment_constants, mean_experiment_loss
from lib.utils.helper_functions import CreatedClass
from lib.utils.run_artifacts import parameter_axes, scale_parameters
from lib.utils.source_stamp import write_stamp, verify_stamp
from lib.utils.yamlread import YAMLReader
from local_agent.agent.validators import validate_session, parse_input_yaml, smoke_test_generated_script, import_generated_script, ValidationError
from local_agent.agent.checks import check_session
from local_agent.agent.workflow import LocalWorkflow
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.config import WorkflowConfig
from local_agent.llm.fake import FakeLLMClient

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).parent / 'fixtures'


def copy_fixture(tmp_path, name='decay_multiexp'):
    session = tmp_path / name
    shutil.copytree(FIXTURES / name, session)
    return session


def change_config(session, mutate):
    path = session / 'inputs/user_input.yaml'
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def make_problem(session):
    reader = parse_input_yaml(session / 'inputs/user_input.yaml')
    module = import_generated_script(session / 'generated/generated_script.py')
    problem = CreatedClass(experiments=load_experiments(session, reader), input_reader=reader,
                           compute_loss_problem=module._compute_loss_problem,
                           write_problem_result=module._write_problem_result)
    lo, hi, logs = parameter_axes(reader)
    problem.set_min_limit(lo)
    problem.set_max_limit(hi)
    problem.set_is_logscale(logs)
    return reader, module, problem


def test_reader_and_loader_preserve_every_record_and_override(tmp_path):
    session = copy_fixture(tmp_path)
    reader = parse_input_yaml(session / 'inputs/user_input.yaml')
    assert len(reader.experiments) == 2
    assert reader.get_y0(0) == [1.0, 0.0]
    assert reader.get_y0(1) == [2.0, 0.5]
    change_config(session, lambda c: c['experiments'][1]['initial_conditions'].pop('B'))
    assert parse_input_yaml(session / 'inputs/user_input.yaml').get_y0(1) == [2.0, 0.0]
    path = session / 'inputs/decay_run_B.csv'
    data = np.loadtxt(path, delimiter=',')
    np.savetxt(path, data[::2], delimiter=',')
    validation = validate_session(session)
    assert len(validation.experiments) == 2
    assert len(validation.experiments[1]['t_eval']) < len(validation.experiments[0]['t_eval'])


@pytest.mark.parametrize('damage,match', [
    ('missing', 'Dataset file not found'), ('time', 'strictly increasing'),
    ('ic', 'not an integrated variable'), ('finite', 'finite'),
    ('columns', 'same ordered column'), ('meaning', 'same ordered column'),
])
def test_bad_second_record_is_rejected(tmp_path, damage, match):
    session = copy_fixture(tmp_path)
    path = session / 'inputs/decay_run_B.csv'
    if damage == 'missing':
        path.unlink()
    elif damage == 'time':
        data = np.loadtxt(path, delimiter=','); data[1, 0] = data[0, 0]
        np.savetxt(path, data, delimiter=',')
    else:
        def mutate(c):
            exp = c['experiments'][1]
            if damage == 'ic': exp['initial_conditions']['unknown'] = 1
            elif damage == 'finite': exp['initial_conditions']['A'] = float('nan')
            elif damage == 'columns': exp['columns'][1:] = exp['columns'][1:][::-1]
            else: exp['columns'][1]['observes'] = 'B'
        change_config(session, mutate)
    with pytest.raises(ValidationError, match=match) as exc:
        validate_session(session)
    assert 'xperiment 2' in str(exc.value)


def test_objective_gradient_and_hessian_use_equal_record_weights():
    constants = [{'dataset': jnp.full((2,), 1.), 'error_loss': 1e30},
                 {'dataset': jnp.full((9,), 3.), 'error_loss': 1e30}]
    loss = lambda c, x: jnp.mean((x[0] - c['dataset']) ** 2)
    aggregate = lambda x: mean_experiment_loss(loss, constants, x)
    x = jnp.array([0.])
    assert float(aggregate(x)) == pytest.approx(5.)
    np.testing.assert_allclose(jax.grad(aggregate)(x), [-4.])
    np.testing.assert_allclose(jax.hessian(aggregate)(x), [[2.]])
    assert float(aggregate(jnp.array([2.]))) == pytest.approx(1.)
    constants[1]['dataset'] = jnp.full((9,), 5.)
    assert float(aggregate(x)) == pytest.approx(13.)
    np.testing.assert_allclose(jax.grad(aggregate)(x), [-6.])


@pytest.mark.parametrize('bad', [1e30, float('nan'), float('inf')])
def test_failure_in_second_record_is_not_averaged_down(bad):
    constants = [{'value': 0., 'error_loss': 1e30}, {'value': bad, 'error_loss': 1e30}]
    loss = lambda c, x: jnp.asarray(c['value']) + x[0] * 0
    assert float(mean_experiment_loss(loss, constants, jnp.zeros(1))) == pytest.approx(1e30)


def test_decay_numerical_objective_and_output_have_both_trajectories(tmp_path):
    session = copy_fixture(tmp_path)
    path = session / 'inputs/decay_run_B.csv'
    data = np.loadtxt(path, delimiter=','); np.savetxt(path, data[::2], delimiter=',')
    reader, module, problem = make_problem(session)
    point = scale_parameters([1., .3], reader)
    losses = [float(module._compute_loss_problem(c, point)) for c in problem.constants_list]
    assert max(losses) < 1e-6
    np.testing.assert_allclose(float(problem._compute_loss(point)), np.mean(losses), atol=1e-10)
    assert not np.array_equal(problem.constants_list[0]['init_cond'], problem.constants_list[1]['init_cond'])
    reader.output_dir = tmp_path / 'results'; reader.output_dir.mkdir()
    problem.write_problem_result(point, reader, label='result')
    for index, record in enumerate(problem.experiments):
        output = np.loadtxt(reader.output_dir / f'result_solution_exp{index+1}.csv', delimiter=',')
        np.testing.assert_allclose(output[:, 0], record['t_eval'])
        np.testing.assert_allclose(output[:, 1:3], output[:, 3:5], atol=1e-6)
    assert len(json.loads((reader.output_dir / 'result_experiments.json').read_text())) == 2


def test_later_record_failure_prevents_translation_acceptance(tmp_path, monkeypatch):
    session = copy_fixture(tmp_path)
    import types
    stub = types.SimpleNamespace(
        _compute_loss_problem=lambda c, x: 1e30 if c['init_cond'][0] == 2. else 0.,
        _write_problem_result=lambda c, x: np.column_stack((c['t_eval'], c['dataset'])),
    )
    monkeypatch.setattr('local_agent.agent.validators.import_generated_script', lambda p: stub)
    workflow = LocalWorkflow(FakeLLMClient([]), PromptRenderer())
    from lib.utils.source_stamp import build_stamp
    workflow._generation_source_stamp = build_stamp(session)
    events = []
    assert not workflow._validate_generated_script(session / 'generated/generated_script.py', session, events)
    assert 'Experiment 2' in events[-1].message
    assert verify_stamp(session)[0] is False


def test_sneyd_has_nine_independent_clamp_conditions(tmp_path):
    session = copy_fixture(tmp_path, 'sneyd_ipr')
    validation = validate_session(session)
    assert len(validation.experiments) == 9
    reader = parse_input_yaml(session / 'inputs/user_input.yaml')
    overrides = [reader.get_y0(i)[-2:] for i in range(9)]
    assert overrides == [[10., .1], [10., .2], [10., .4], [10., 1.], [10., 3.], [10., 10.], [3., .4], [5., .4], [10., .4]]


def test_fake_llm_new_and_jax_preserve_experiments(tmp_path):
    from local_agent.agent.session_init import init_session
    session = tmp_path / 'new'
    inputs = session / 'inputs'; inputs.mkdir(parents=True)
    for name, initial, times in [('a.csv', 1., [0., 1., 2.]), ('b.csv', 2., [0., .5, 1.5, 2.])]:
        np.savetxt(inputs / name, np.column_stack((times, initial*np.exp(-np.asarray(times)))), delimiter=',', header='time,y', comments='')
    (inputs / 'user_info.txt').write_text('Fit a.csv and b.csv with shared k. dy/dt=-k*y. Global initial y=1; b.csv initial y=2. Loss: MSE.')
    common = {'missing_inputs': [], 'review': '', 'user_info_txt': ''}
    responses = [
        {**common, 'experiments': [{'data_file':'a.csv'}, {'data_file':'b.csv', 'initial_conditions': {'y':2.}}]},
        {**common, 'parameters':[{'name':'k','min_value':.1,'max_value':2.,'logscale':False}], 'fixed_parameters':[]},
        {**common, 'states':[{'name':'y','initial_value':1.}]},
        {**common, 'formulas':[], 'rhs':[{'state':'y','expression':'-k*y'}]},
        {**common, 'observables':[{'measured':'y','simulated':'y','expression':''}]},
        {**common, 'custom_loss':True,'data_terms':[{'measured':'y','simulated':'y','metric':'mse'}], 'penalties':[]},
    ]
    llm = FakeLLMClient([json.dumps(r) for r in responses])
    init_session(session, llm, PromptRenderer())
    reader = parse_input_yaml(inputs / 'user_input.yaml')
    assert len(reader.experiments) == 2
    assert reader.get_y0(1) == [2.]
    assert all('b.csv' in '\n'.join(m.content for m in request) for request in llm.requests[1:])
    llm = FakeLLMClient([json.dumps({'helper_functions':[], 'review':''}),
                         json.dumps({'rhs':['-k*y'],'review':''}),
                         json.dumps({'writeout_body':'return jnp.column_stack((solution_time, dataset, solution))','review':''})])
    result = LocalWorkflow(llm, PromptRenderer(), WorkflowConfig(max_repair_attempts=0)).generate_script(session)
    assert result.success, result.events
    assert verify_stamp(session)[0] is True
    assert check_session(session).passed
    assert 'b.csv' in '\n'.join(m.content for m in llm.requests[0])
    reader, module, problem = make_problem(session)
    point = scale_parameters([1.], reader)
    assert float(problem._compute_loss(point)) < 1e-10


def test_header_swap_in_second_file_is_rejected(tmp_path):
    session = copy_fixture(tmp_path)
    path = session/'inputs/decay_run_B.csv'
    path.write_text('time,B,A\n'+path.read_text())
    with pytest.raises(ValidationError, match='Experiment 2.*CSV header'):
        validate_session(session)


def test_legacy_constructor_cannot_silently_drop_records(tmp_path):
    session = copy_fixture(tmp_path)
    reader = parse_input_yaml(session/'inputs/user_input.yaml')
    first = load_experiments(session, reader)[0]
    with pytest.raises(ValueError, match='all experiment records'):
        CreatedClass(first['dataset'], first['t_eval'], first['y0'], reader)


def test_each_record_uses_own_default_initial_time(tmp_path):
    session = copy_fixture(tmp_path)
    path = session/'inputs/decay_run_B.csv'
    data = np.loadtxt(path,delimiter=',');data[:,0]+=2.
    np.savetxt(path,data,delimiter=',')
    reader=parse_input_yaml(session/'inputs/user_input.yaml')
    records=load_experiments(session,reader)
    constants=[experiment_constants(record,reader) for record in records]
    assert constants[1]['init_time'] == constants[0]['init_time']+2.
    change_config(session,lambda c:c['gradient_opt'].update(initial_time=1.))
    with pytest.raises(ValidationError,match='Experiment 1.*initial_time'):
        validate_session(session)


def test_default_loss_preserves_missing_measurements_but_not_invalid_simulations(tmp_path):
    from local_agent.agent.session_spec import load_session_spec
    from local_agent.agent.workflow import _standard_loss_and_writeout_bodies
    session=copy_fixture(tmp_path)
    spec=load_session_spec(session/'inputs/user_input.yaml')
    loss_body,_=_standard_loss_and_writeout_bodies(spec)
    namespace={'jnp':jnp}
    exec('def loss(solution,dataset):\n'+'\n'.join('    '+line for line in loss_body.splitlines()),namespace)
    measured=jnp.array([[1.,2.],[jnp.nan,4.]])
    simulated=jnp.array([[1.5,2.],[3.,4.]])
    assert float(namespace['loss'](simulated,measured)) == pytest.approx(np.sqrt(.25/3))
    assert np.isnan(namespace['loss'](simulated.at[1,0].set(jnp.inf),measured))
    assert np.isnan(namespace['loss'](simulated,measured.at[:,0].set(jnp.nan)))


def test_new_full_response_preserves_overrides_through_normalization(tmp_path):
    from local_agent.agent.session_init import init_session
    from tests.test_session_init import _new_session_response
    session=tmp_path/'full';inputs=session/'inputs';inputs.mkdir(parents=True)
    (inputs/'data.csv').write_text('time,y\n0,1\n1,.5\n')
    (inputs/'other.csv').write_text('time,y\n0,2\n1,1\n')
    response=json.loads(_new_session_response(rhs='-k * abs(y)'))
    response['experiments']=[{'data_file':'data.csv'}, {'data_file':'other.csv','initial_conditions':{'y':2}}]
    init_session(session,FakeLLMClient([json.dumps(response)]),PromptRenderer())
    reader=parse_input_yaml(inputs/'user_input.yaml')
    assert reader.get_y0(1)==[2.]
    assert 'np.abs(y)' in (session/'generated/user_model.py').read_text()


def test_record_loss_must_be_scalar():
    with pytest.raises(ValueError,match='must be scalar'):
        mean_experiment_loss(lambda c,x:jnp.array([1.,2.]),[{'error_loss':1e30}],jnp.zeros(1))


def test_sneyd_translation_uses_all_nine_records(tmp_path):
    session=copy_fixture(tmp_path,'sneyd_ipr')
    # Deterministic fake fragments exercise the actual local translation pipeline,
    # including source-intermediate expansion and the derived observable helper.
    rhs=['dOdt','dRdt','dI1dt','dSdt','dAdt','dI2dt','dIP3dt','dCadt']
    writeout="observables = _observables(solution, trainable_parameters, fixed_parameters)\nreturn jnp.column_stack((solution_time, dataset[:, 0], observables['Po']))"
    llm=FakeLLMClient([json.dumps({'helper_functions':[], 'review':''}),
                      json.dumps({'rhs':rhs, 'review':''}),
                      json.dumps({'writeout_body':writeout, 'review':''})])
    workflow=LocalWorkflow(llm,PromptRenderer(),WorkflowConfig(max_repair_attempts=0))
    result=workflow.generate_script(session)
    assert result.success, result.events
    assert verify_stamp(session)[0] is True
    assert check_session(session).passed
    prompt='\n'.join(m.content for m in llm.requests[0])
    assert 'ca01.csv' in prompt and 'ip10.csv' in prompt
    reader,module,problem=make_problem(session)
    point=scale_parameters(np.loadtxt(session/'reference_parameters.csv',delimiter=','),reader)
    losses=[float(module._compute_loss_problem(c,point)) for c in problem.constants_list]
    assert len(losses)==9 and np.all(np.isfinite(losses)) and max(losses)<reader.error_loss
    assert float(problem._compute_loss(point)) == pytest.approx(np.mean(losses),rel=1e-7)
