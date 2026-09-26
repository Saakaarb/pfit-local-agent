import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from lib.utils.scientific_diagnosis import scientific_diagnosis, residual_statistics, probe_gradient
from local_agent.agent.diagnostics import diagnose_run

pytestmark=pytest.mark.unit

MODEL='''import jax.numpy as jnp
from diffrax import RESULTS

def _integrate_system(c,p):
    k=.1+(p[0]+1)/2*3.9
    return c['t_eval'], (k*c['t_eval']+.5)[:,None], RESULTS.successful

def _compute_loss_problem(c,p):
    _,y,_=_integrate_system(c,p)
    return jnp.mean((y[:,0]-c['dataset'][:,0])**2)
'''

def saved_run(tmp_path):
    session=tmp_path/'session';run=session/'outputs/run_1';snap=run/'snapshot'
    (snap/'inputs').mkdir(parents=True);(snap/'generated').mkdir()
    c=yaml.safe_load(Path('tests/fixtures/cascaded_tanks/inputs/user_input.yaml').read_text())
    c['model']={'trainable_parameters':[{'name':'k','min_val':.1,'max_val':4.,'logscale':False}], 'integrated_variables':[{'name':'x','init_val':.5}]}
    c['experiments']=[]
    for i in range(2):
        t=np.linspace(0,2,12+i)
        np.savetxt(snap/f'inputs/data{i}.csv',np.column_stack([t,2*t]),delimiter=',')
        c['experiments'].append({'data_file':f'data{i}.csv','columns':[{'name':'time'},{'name':'x','observes':'x'}]})
    (snap/'inputs/run_config.yaml').write_text(yaml.safe_dump(c))
    (snap/'generated/generated_script.py').write_text(MODEL)
    (snap/'generated/user_model.py').write_text(MODEL)
    (run/'final_parameters.json').write_text(json.dumps({'parameters':[{'name':'k','value':3.99}]}))
    (run/'final_design_point.csv').write_text('3.99\n')
    (run/'fit_summary.json').write_text(json.dumps({'termination':'iteration_budget','refinement_error':None}))
    (run/'run_manifest.json').write_text(json.dumps({'status':'completed'}))
    return session,run


def test_diagnosis_snapshot_residuals_bounds_and_next_steps(tmp_path):
    session,run=saved_run(tmp_path)
    # The working inputs differ and must never contaminate historical diagnosis.
    (session/'inputs').mkdir();(session/'inputs/user_input.yaml').write_text('{}')
    before=(run/'snapshot/inputs/run_config.yaml').read_bytes()
    report=scientific_diagnosis(session,run,probe_gradients=True)
    assert report['gradient_probe']['status']=='ok'
    assert len(report['experiments'])==2 and len(report['plots'])==2
    assert {'BOUND','STATIONARITY','BUDGET','RESIDUAL'} <= {f['id'] for f in report['findings']}
    assert report['parameters'][0]['value']==3.99
    for exp in report['experiments']:
        assert exp['channels'][0]['bias'] == pytest.approx(2.49)
        assert exp['channels'][0]['rmse']>2
    assert all((run/p).exists() for p in report['plots'])
    assert before==(run/'snapshot/inputs/run_config.yaml').read_bytes()
    assert 'Next:' in diagnose_run(session,'run_1').read_text()


def test_failed_solve_stops_scientific_interpretation(tmp_path):
    session,run=saved_run(tmp_path)
    p=run/'snapshot/generated/generated_script.py'
    p.write_text(MODEL.replace('return jnp.mean((y[:,0]-c[\'dataset\'][:,0])**2)',"return c['error_loss']"))
    report=scientific_diagnosis(session,run)
    assert report['findings'][0]['id']=='SOLVE'
    assert report['experiments']==[] and report['plots']==[]


def test_missing_snapshot_never_uses_working_model(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    report=scientific_diagnosis(tmp_path,tmp_path)
    assert report['status']=='limited'
    assert report['findings'][0]['id']=='SNAPSHOT'


def test_residual_masks_and_gradient_probe_detect_pathology():
    import jax
    import jax.numpy as jnp
    time=np.arange(10.)
    measured=time.copy();measured[2]=np.nan
    stats=residual_statistics(time,measured,time+1)
    assert stats['count']==9 and stats['rmse']==1
    with pytest.raises(ValueError,match='nonfinite'):
        residual_statistics(time,measured,np.full(10,np.nan))
    probe=probe_gradient(lambda x: jax.lax.stop_gradient(x[0]**2),np.array([.5]),5000.)
    assert probe['status']=='disagreement'
    probe=probe_gradient(lambda x: jnp.where(x[0]>.5,5000.,x[0]**2),np.array([.5]),5000.)
    assert probe['status']=='unavailable' and 'Nearby solve' in probe['reason']


def test_ollama_interpretation_uses_snapshot_and_valid_evidence(tmp_path):
    from local_agent.llm.fake import FakeLLMClient
    session,run=saved_run(tmp_path)
    response={'verdict':'Budget and bounds need review.', 'recommendations':[{'evidence_ids':['BOUND'], 'action':'Review the physical upper bound before refitting.', 'reason':'The final parameter is near its upper limit.'}], 'limitations':['A short fit does not establish convergence.']}
    client=FakeLLMClient([json.dumps(response)])
    report=diagnose_run(session,'run_1',llm_client=client).read_text()
    assert 'Ollama interpretation' in report and 'Review the physical upper bound' in report
    assert 'k' in client.requests[0][1].content
    assert json.loads((run/'ollama_diagnosis.json').read_text())['status']=='ok'
    assert (run/'agent_logs/llm_calls.jsonl').exists()


def test_ollama_cannot_cite_invented_evidence(tmp_path):
    from local_agent.llm.fake import FakeLLMClient
    session,run=saved_run(tmp_path)
    response={'verdict':'Claim', 'recommendations':[{'evidence_ids':['INVENTED'], 'action':'Change loss.', 'reason':'Unsupported.'}], 'limitations':[]}
    report=diagnose_run(session,'run_1',llm_client=FakeLLMClient([json.dumps(response)])).read_text()
    assert 'unknown evidence IDs' in report
    assert '1. Change loss.' not in report
    assert (run/'scientific_diagnosis.json').exists()
