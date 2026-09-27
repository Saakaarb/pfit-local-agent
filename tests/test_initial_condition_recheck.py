import json

import pytest
import yaml

from local_agent.agent.config import WorkflowConfig
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.session_init import init_session
from local_agent.llm.fake import FakeLLMClient


def setup_case(tmp_path, broad=True):
    session = tmp_path / 'session'
    inputs = session / 'inputs'
    inputs.mkdir(parents=True)
    for name in ['first.csv', 'second.csv']:
        (inputs / name).write_text('time,A\n0,1\n1,0.9\n')
    (inputs / 'user_info.txt').write_text(
        'Fit states A, B and U with shared k in [0.1,2]. A starts at 1. '
        + ('All other reaction states start at zero. ' if broad else '')
        + 'U is a zero-derivative input state, clamped to 2 in first.csv and 4 in second.csv. '
        + 'dA/dt=-k*A*U, dB/dt=k*A*U, dU/dt=0. Use RMSE for observed A.')
    def response(**fields):
        return json.dumps(dict(missing_inputs=[], review='Extracted.', user_info_txt='Extracted.', **fields))
    prefix = [response(filename_data='first.csv', experiments=[
        dict(data_file='first.csv', initial_conditions={'U': 2}),
        dict(data_file='second.csv', initial_conditions={'U': 4})]),
        response(parameters=[dict(name='k', min_value=.1, max_value=2, logscale=False)], fixed_parameters=[])]
    missing = json.dumps(dict(missing_inputs=['B'], review='Missing B initial value.', states=[], user_info_txt='Missing B.'))
    states = response(states=[dict(name='A', initial_value=1), dict(name='B', initial_value=0), dict(name='U', initial_value=2)])
    suffix = [response(formulas=[], rhs=[dict(state='A', expression='-k*A*U'),
        dict(state='B', expression='k*A*U'), dict(state='U', expression='0')]),
        response(observables=[dict(measured='A', simulated='A', expression='')]),
        response(custom_loss=True, data_terms=[dict(simulated='A', measured='A', metric='rmse')], penalties=[])]
    return session, prefix, missing, states, suffix


def test_group_initial_values_rechecked_without_losing_experiment_overrides(tmp_path):
    session, prefix, missing, states, suffix = setup_case(tmp_path)
    llm = FakeLLMClient(prefix + [missing, states] + suffix)
    init_session(session, llm, PromptRenderer())
    config = yaml.safe_load((session / 'inputs/user_input.yaml').read_text())
    assert {v['name']: v['init_val'] for v in config['model']['integrated_variables']} == {'A': 1, 'B': 0, 'U': 2}
    assert [e['initial_conditions']['U'] for e in config['experiments']] == [2, 4]
    assert len(llm.requests) == 7
    retry = '\n'.join(m.content for m in llm.requests[3])
    assert 'All other reaction states start at zero' in retry
    assert 'Never invent a zero' in retry
    records = [json.loads(s) for s in (session / 'generated/agent_logs/llm_calls.jsonl').read_text().splitlines()]
    assert records[3]['step'] == 'repair_new_session_initial_conditions'


@pytest.mark.parametrize('attempts,expected_calls', [(0, 3), (5, 4)])
def test_genuinely_missing_value_remains_missing_and_retry_is_bounded(tmp_path, attempts, expected_calls):
    session, prefix, missing, _, _ = setup_case(tmp_path, broad=False)
    llm = FakeLLMClient(prefix + [missing, missing])
    with pytest.raises(ValueError, match='missing required inputs'):
        init_session(session, llm, PromptRenderer(), WorkflowConfig(max_repair_attempts=attempts))
    assert len(llm.requests) == expected_calls
    assert not (session / 'inputs/user_input.yaml').exists()
    assert not (session / 'generated/user_model.py').exists()


@pytest.mark.parametrize("complete", [True, False])
def test_per_experiment_values_only_resolve_complete_declarations(complete):
    from local_agent.agent.session_init import _resolve_experiment_initial_values
    data = dict(states=[dict(name="A", initial_value=1)], missing_inputs=["U", "B"])
    experiments = [dict(initial_conditions={"U": 2}),
                   dict(initial_conditions={"U": 4} if complete else {})]
    _resolve_experiment_initial_values(data, experiments)
    assert data["missing_inputs"] == (["B"] if complete else ["U", "B"])
    assert data["states"] == ([dict(name="A", initial_value=1), dict(name="U", initial_value=2)]
                              if complete else [dict(name="A", initial_value=1)])
    assert experiments[0]["initial_conditions"]["U"] == 2
