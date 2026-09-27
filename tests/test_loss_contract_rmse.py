"""The original loss request must survive an LLM summary that drops RMSE."""
import json
from pathlib import Path

import numpy as np
import pytest

from local_agent.agent.checks import user_loss_contract
from local_agent.agent.session_init import init_session, _normalize_split_loss_data
from local_agent.agent.prompts import PromptRenderer
from local_agent.llm.fake import FakeLLMClient

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('case', ['boehm_stat5', 'oregonator'])
def test_original_loss_controls_generated_pooled_rmse(tmp_path, case):
    weighted = case == 'oregonator'
    names = ['X', 'Z'] if weighted else ['pSTAT5A', 'pSTAT5B', 'rSTAT5A']
    n = len(names)
    measured = np.arange(1., 2*n+1).reshape(2, n)
    scale = np.array([[.5, 2.], [1., 4.]]) if weighted else measured.max(axis=0)
    residual = np.arange(3., 2*n+3).reshape(2, n)
    solution = measured + residual * scale
    headers = ['time', *names, *([f'{x}_sd' for x in names] if weighted else [])]
    dataset = np.column_stack((measured, scale)) if weighted else measured
    inputs = tmp_path/'inputs';inputs.mkdir()
    np.savetxt(inputs/'data.csv', np.column_stack(([0., 1.], dataset)), delimiter=',',
               header=','.join(headers), comments='')
    original = user_loss_contract(REPO/'sessions'/case)
    (inputs/'user_info.txt').write_text(original)
    terms = [dict(simulated=name, measured=name,
                  metric='sigma_weighted_mse' if weighted else 'max_abs_normalized_mse',
                  **({'sigma':f'{name}_sd'} if weighted else {})) for name in names]
    responses = [
        dict(filename_data='data.csv'),
        dict(parameters=[dict(name='k', min_value=.1, max_value=2., logscale=False)], fixed_parameters=[]),
        dict(states=[dict(name=name, initial_value=float(measured[0,i])) for i,name in enumerate(names)]),
        dict(rhs=[dict(state=name, expression=f'-k * {name}') for name in names]),
        dict(observables=[dict(measured=name, simulated=name, expression='') for name in names]),
        dict(custom_loss=True, data_terms=terms, penalties=[], review='Use scaled MSE.', user_info_txt='Squared error.'),
    ]
    client = FakeLLMClient([json.dumps(dict(missing_inputs=[], **r)) for r in responses])
    init_session(tmp_path, client, PromptRenderer())
    namespace = {}
    exec((tmp_path/'generated/user_model.py').read_text(), namespace)
    actual = namespace['_compute_loss_problem'](np.array([0.,1.]), solution, dataset, {'k':1.}, {})
    assert actual == pytest.approx(np.sqrt(np.mean(residual**2)), rel=1e-10)
    assert len(client.requests) == 6  # deterministic correction, no additional LLM call
    assert (inputs/'user_info.txt').read_text() == original
    assert 'original user loss specification' in (tmp_path/'generated/pfit_new_review.txt').read_text()


@pytest.mark.parametrize('contract', [
    'Loss: use MSE, not RMSE.',
    'Loss: RMSE for X and MAE for Z.',
    'Loss: sum of per-channel RMSE values.',
    'Loss: either RMSE or MSE.',
    'Loss: RMSE plus mean absolute error.',
])
def test_ambiguous_or_mixed_request_does_not_force_rmse(contract):
    data = dict(review='Use RMSE.', data_terms=[dict(simulated='X', measured='X', metric='mse')], penalties=[])
    result = _normalize_split_loss_data(data, [{'name':'X'}], [], [], ['time','X'], loss_contract=contract)
    assert result['data_terms'] == data['data_terms']


def test_authoritative_mse_is_not_overridden_by_llm_rmse_summary():
    data = dict(review='Use RMSE.', data_terms=[dict(simulated='X', measured='X', metric='mse')], penalties=[])
    result = _normalize_split_loss_data(data, [{'name':'X'}], [], [], ['time','X'], loss_contract='Loss: plain MSE.')
    assert result['data_terms'] == data['data_terms']
