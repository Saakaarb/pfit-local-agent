import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lib.utils.source_stamp import verify_stamp
from local_agent.agent.config import WorkflowConfig
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.validators import SolverValidationError, parse_input_yaml
from local_agent.agent.workflow import LocalWorkflow, _restrict_repair_response
from local_agent.llm.fake import FakeLLMClient


SOURCE = '''import numpy as np
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    eps1 = trainable_parameters['eps1']
    eps2 = trainable_parameters['eps2']
    q = trainable_parameters['q']
    f = trainable_parameters['f']
    X, Y, Z = y
    return np.array([(q*Y-X*Y+X*(1-X))/eps1, (-q*Y-X*Y+f*Z)/eps2, X-Z])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = np.mean(np.square((solution[:, 0]-dataset[:, 0])/dataset[:, 2]))
    loss += np.mean(np.square((solution[:, 2]-dataset[:, 1])/dataset[:, 3]))
    return float(np.sqrt(loss/2))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return np.column_stack((solution_time, dataset, solution))
'''
RHS = ['(q*Y-X*Y+X*(1-X))/eps1', '(-q*Y-X*Y+f*Z)/eps2', 'X-Z']


def oregonator(tmp_path, cap=None, timeout=120, multi=False):
    session = tmp_path / 'session'
    (session / 'inputs').mkdir(parents=True)
    (session / 'generated').mkdir()
    experiments = []
    for name, stop in ([('short.csv', 0.1), ('long.csv', 30)] if multi else [('long.csv', 30)]):
        data = np.column_stack((np.linspace(0, stop, 300), np.full((300, 2), .2), np.full((300, 2), .02)))
        np.savetxt(session / 'inputs' / name, data, delimiter=',', header='time,X,Z,X_sd,Z_sd', comments='')
        experiments.append(dict(data_file=name, columns=[{'name': 'time'}, {'name': 'X', 'observes': 'X'},
            {'name': 'Z', 'observes': 'Z'}, {'name': 'X_sd', 'uncertainty_of': 'X'}, {'name': 'Z_sd', 'uncertainty_of': 'Z'}]))
    config = dict(experiments=experiments, model=dict(
        trainable_parameters=[dict(name=n, min_val=lo, max_val=hi, logscale=True) for n, lo, hi in
            [('eps1', .0004, 4), ('eps2', .000004, .04), ('q', .000008, .08), ('f', .01, 100)]],
        integrated_variables=[dict(name=n, init_val=v) for n, v in
            [('X', .778383715217), ('Y', .267086592467), ('Z', .208522420421)]]),
        population_opt=dict(algorithm='DE', population_size=4, num_iters=1, processors=1),
        gradient_opt=dict(max_steps=10000, integrator='Kvaerno5', stepsize_rtol=1e-7,
            stepsize_atol=1e-9, initial_timestep=1e-6, num_iters=5, solver_recovery_timeout_seconds=timeout,
            solver_validation_samples=1, solver_validation_max_samples=1, solver_validation_success_fraction=1.0))
    if cap is not None:
        config['gradient_opt']['solver_recovery_max_steps'] = cap
    (session / 'inputs/user_input.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (session / 'generated/user_model.py').write_text(SOURCE)
    llm = FakeLLMClient([json.dumps({'helper_functions': []}), json.dumps({'rhs': RHS})])
    return session, config, llm


def test_real_oregonator_recovers_all_records_without_llm_repair(tmp_path):
    session, original, llm = oregonator(tmp_path, cap=50000, multi=True)
    result = LocalWorkflow(llm, PromptRenderer()).generate_script(session)
    assert result.success, result.events
    assert len(llm.requests) == 2
    updated = yaml.safe_load((session / 'inputs/user_input.yaml').read_text())
    assert updated['gradient_opt']['max_steps'] == 20000
    updated['gradient_opt']['max_steps'] = 10000
    assert updated == original
    assert (session / 'generated/user_model.py').read_text() == SOURCE
    assert verify_stamp(session)[0] is True
    diagnostics = json.loads((session / 'generated/solver_diagnostics.json').read_text())
    failure = next(d for d in diagnostics if d['code'] == 'step_limit')
    assert failure['experiment'] == 2
    assert failure['stats']['num_steps'] == 10000
    assert failure['finite_rows'] < failure['requested_rows']
    assert {d['experiment'] for d in diagnostics if d['code'] == 'successful' and d['max_steps'] == 20000} == {1, 2}
    assert json.loads((session / 'generated/translation_fidelity.json').read_text())['status'] == 'passed'


@pytest.mark.parametrize('cap,timeout', [(None, 120), (11000, 120), (50000, .001)])
def test_exhausted_or_timed_out_recovery_restores_config_and_never_repairs_code(tmp_path, cap, timeout):
    session, _, llm = oregonator(tmp_path, cap=cap, timeout=timeout)
    before = (session / 'inputs/user_input.yaml').read_bytes()
    result = LocalWorkflow(llm, PromptRenderer(), WorkflowConfig(max_repair_attempts=5)).generate_script(session)
    assert not result.success
    assert len(llm.requests) == 2
    assert (session / 'inputs/user_input.yaml').read_bytes() == before
    assert verify_stamp(session)[0] is False
    history = json.loads((session / 'generated/solver_recovery.json').read_text())
    assert history['status'] == 'exhausted'
    if cap is None:
        assert history['attempts'] == []
    if timeout < 1:
        assert any('recovery_timeout' in event.message for event in result.events)


def test_non_step_solver_failure_does_not_trigger_budget_retries_or_code_repair(tmp_path, monkeypatch):
    session, _, llm = oregonator(tmp_path, cap=50000)
    def failed(*args):
        raise SolverValidationError({'code': 'integration_failure', 'result': 'Nonlinear solve failed'})
    monkeypatch.setattr('local_agent.agent.workflow.smoke_test_generated_script', failed)
    result = LocalWorkflow(llm, PromptRenderer()).generate_script(session)
    assert not result.success
    assert len(llm.requests) == 2
    assert not any(e.step == 'solver_recovery' and e.status == 'attempted' for e in result.events)


def test_recovery_never_stamps_a_concurrent_source_edit(tmp_path, monkeypatch):
    session, _, llm = oregonator(tmp_path, cap=50000)
    def changed(*args):
        (session / 'generated/user_model.py').write_text(SOURCE + '\n# edited during validation\n')
        raise SolverValidationError({'code': 'step_limit', 'result': 'Step limit reached'})
    monkeypatch.setattr('local_agent.agent.workflow.smoke_test_generated_script', changed)
    result = LocalWorkflow(llm, PromptRenderer()).generate_script(session)
    assert not result.success
    assert len(llm.requests) == 2
    assert any(e.step == 'source_freshness' and e.status == 'failed' for e in result.events)
    assert verify_stamp(session)[0] is False


def test_repair_can_fix_rhs_but_cannot_change_loss_or_writeout():
    old = dict(rhs=['unknown'], loss_body='return loss', writeout_body='return out', helper_functions=[])
    good = _restrict_repair_response(json.dumps(old), json.dumps({'rhs': ['-k*x']}), {'rhs'}, {'loss_body': 'return loss'})
    assert json.loads(good) == dict(old, rhs=['-k*x'])
    for field in ['loss_body', 'writeout_body']:
        with pytest.raises(ValueError, match='protected/unrelated'):
            _restrict_repair_response(json.dumps(old), json.dumps({'rhs': ['-k*x'], field: 'return 0'}),
                                     {'rhs'}, {'loss_body': 'return loss'})


def test_repeated_unrelated_repairs_stop_without_rendering_changed_loss(tmp_path):
    session, _, _ = oregonator(tmp_path)
    bad = json.dumps({'rhs': ['missing', *RHS[1:]]})
    unrelated = json.dumps({'rhs': RHS, 'loss_body': 'return 0'})
    llm = FakeLLMClient([json.dumps({'helper_functions': []}), bad, unrelated, unrelated])
    result = LocalWorkflow(llm, PromptRenderer(), WorkflowConfig(max_repair_attempts=5)).generate_script(session)
    assert not result.success
    assert len(llm.requests) == 4
    assert any('protected/unrelated' in e.message for e in result.events)
    assert any('Repeated failed repair' in e.message for e in result.events)
    assert not (session / 'generated/generated_script.py').exists()


@pytest.mark.parametrize('field,value', [('solver_recovery_max_steps', 9999), ('solver_recovery_max_steps', True),
    ('solver_recovery_timeout_seconds', 0), ('solver_recovery_timeout_seconds', float('inf'))])
def test_invalid_recovery_policy_rejected(tmp_path, field, value):
    session, config, _ = oregonator(tmp_path)
    config['gradient_opt'][field] = value
    path = session / 'inputs/user_input.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='solver_recovery'):
        parse_input_yaml(path)


def test_flat_coverage_stops_after_two_increases_and_restores_inputs(tmp_path, monkeypatch):
    session, config, llm = oregonator(tmp_path, cap=50000)
    config['gradient_opt'].update(solver_validation_samples=32, solver_validation_max_samples=128,
                                 solver_validation_success_fraction=.25)
    path = session/'inputs/user_input.yaml'
    path.write_text(yaml.safe_dump(config)); original=path.read_bytes()
    seen=[]
    def flat(script, session):
        g=yaml.safe_load(path.read_text())['gradient_opt'];n=g['solver_validation_samples']
        seen.append((g['max_steps'],n))
        raise SolverValidationError(dict(code='coverage_below_target', result='No successful candidates',
            sample_count=n, success_fraction=0., failure_counts={'step_limit':n},
            candidates=[{'successful':False} for _ in range(n)]))
    monkeypatch.setattr('local_agent.agent.workflow.smoke_test_generated_script',flat)
    monkeypatch.setattr('local_agent.agent.solver_recovery.smoke_with_deadline',lambda script,session,timeout:flat(script,session))
    result=LocalWorkflow(llm,PromptRenderer()).generate_script(session)
    assert not result.success
    assert seen==[(10000,32),(20000,64),(40000,128)]
    history=json.loads((session/'generated/solver_recovery.json').read_text())
    assert history['status']=='coverage_stagnated'
    assert 'inadequate search coverage' in history['diagnosis']
    assert path.read_bytes()==original
    assert verify_stamp(session)[0] is False
    assert len(llm.requests)==2


def test_coverage_improvement_resets_stagnation_counter(tmp_path, monkeypatch):
    session, config, llm = oregonator(tmp_path, cap=50000)
    config['gradient_opt'].update(solver_validation_samples=32, solver_validation_max_samples=128,
                                 solver_validation_success_fraction=.25)
    path=session/'inputs/user_input.yaml';path.write_text(yaml.safe_dump(config));seen=[]
    def improving(script, session):
        g=yaml.safe_load(path.read_text())['gradient_opt'];n=g['solver_validation_samples'];budget=g['max_steps']
        seen.append((budget,n))
        if budget==50000:return
        fraction={10000:0.,20000:0.,40000:.125}[budget]
        raise SolverValidationError(dict(code='coverage_below_target',result='Below target',sample_count=n,
            success_fraction=fraction,failure_counts={'step_limit':n},
            candidates=[{'successful':i<int(n*fraction)} for i in range(n)]))
    monkeypatch.setattr('local_agent.agent.workflow.smoke_test_generated_script',improving)
    monkeypatch.setattr('local_agent.agent.solver_recovery.smoke_with_deadline',lambda script,session,timeout:improving(script,session))
    result=LocalWorkflow(llm,PromptRenderer()).generate_script(session)
    assert result.success
    assert seen==[(10000,32),(20000,64),(40000,128),(50000,128)]
    assert yaml.safe_load(path.read_text())['gradient_opt']['solver_validation_samples']==128
    assert verify_stamp(session)[0] is True
