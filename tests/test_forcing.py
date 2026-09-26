"""Measured inputs drive each record, without becoming fitted observations."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import yaml
from lib.utils.experiments import load_experiments, experiment_constants
from lib.utils.run_artifacts import parameter_axes, scale_parameters
from lib.utils.source_stamp import write_stamp
from local_agent.agent.validators import parse_input_yaml, validate_session, import_generated_script, ValidationError
from local_agent.agent.session_spec import load_session_spec
from local_agent.agent.jax_fragments import parse_jax_fragments_response, render_generated_script_from_fragments
from local_agent.agent.session_init import init_session, _assemble_split_new_session_response
from local_agent.agent.prompts import PromptRenderer
from local_agent.llm.fake import FakeLLMClient
from local_agent.core.fitting import run_driver

pytestmark = pytest.mark.unit


def forced_session(tmp_path):
    session=tmp_path/'forced';(session/'inputs').mkdir(parents=True);(session/'generated').mkdir()
    c=yaml.safe_load(Path('tests/fixtures/cascaded_tanks/inputs/user_input.yaml').read_text())
    columns=[{'name':'time'}, {'name':'x','observes':'x'}, {'name':'drive','role':'forcing'}]
    c['model']={'trainable_parameters':[{'name':'k','min_val':.1,'max_val':4.,'logscale':False}], 'integrated_variables':[{'name':'x','init_val':0.}]}
    c['experiments']=[]
    for i,t in enumerate((np.linspace(0,2,9),np.array([0.,.3,.8,1.5,2.]))):
        factor=i+1;initial=float(i)
        data=np.column_stack([t,initial+factor*t*t,factor*t])
        filename=f'record{i}.csv'
        np.savetxt(session/'inputs'/filename,data,delimiter=',',header='time,x,drive',comments='')
        c['experiments'].append({'data_file':filename,'columns':columns,'initial_conditions':{'x':initial}})
    c['population_opt'].update(algorithm='DE',population_size=4,num_iters=1,processors=1,random_seed=9)
    c['gradient_opt'].update(num_iters=2,integrator='Dopri5',initial_time=0.,initial_timestep=.01)
    (session/'inputs/user_input.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    response={'rhs':['k * drive'], 'loss_body':'return jnp.mean(jnp.square(solution[:, 0] - dataset[:, 0]))', 'writeout_body':'return jnp.column_stack((solution_time, dataset[:, 0], solution[:, 0], dataset[:, 1]))', 'review':'forced linear system'}
    spec=load_session_spec(session/'inputs/user_input.yaml')
    fragments=parse_jax_fragments_response(json.dumps(response),spec)
    source=render_generated_script_from_fragments(fragments,spec)
    (session/'generated/generated_script.py').write_text(source)
    (session/'generated/user_model.py').write_text(source)
    return session,c


def test_forcing_interpolation_per_record_fit_and_snapshot(tmp_path):
    session,c=forced_session(tmp_path)
    reader=parse_input_yaml(session/'inputs/user_input.yaml')
    module=import_generated_script(session/'generated/generated_script.py')
    point=scale_parameters([2.],reader);lo,hi,logs=parameter_axes(reader)
    records=load_experiments(session,reader)
    for record in records:
        constants=experiment_constants(record,reader);constants.update(min_limits=lo,max_limits=hi,is_logscale=logs)
        _,ys,result=module._integrate_system(constants,point)
        np.testing.assert_allclose(np.asarray(ys)[:,0],record['dataset'][:,0],atol=1e-6)
        # At a non-grid time, forcing is linearly interpolated for this record.
        derivative=module.user_defined_system(.4,np.array([0.]),{'constants':constants,'trainable_variables':point})
        assert float(derivative[0])==pytest.approx(.8*record['index'])
    changed=dict(constants);changed['dataset']=np.array(constants['dataset']);changed['dataset'][:,1]*=2
    assert float(module._compute_loss_problem(changed,point))>1
    write_stamp(session)
    run=session/'outputs/full'
    fit=run_driver(session,reader,run,sloppiness=False)
    np.testing.assert_allclose(fit,[2.],atol=1e-5)
    for i in (1,2): assert (run/f'result_solution_exp{i}.csv').exists()
    from lib.utils.run_store import run_config
    snapshot=parse_input_yaml(run_config(run,session))
    assert snapshot.experiments[1]['columns'][2]['role']=='forcing'
    # A restart reuses every forcing history; a historical snapshot is self-contained.
    restart=session/'outputs/restart'
    run_driver(session,reader,restart,gradient_only=True,from_run='full',sloppiness=False)
    for rec in records: rec['path'].unlink()
    assert len(load_experiments(run/'snapshot',snapshot))==2


@pytest.mark.parametrize('damage,match',[('nan','Forcing drive'),('coverage','coverage'),('role','same ordered'),('observes','separate'),('interpolation','linear')])
def test_invalid_forcing_rejected(tmp_path,damage,match):
    session,c=forced_session(tmp_path)
    if damage=='nan':
        p=session/'inputs/record1.csv';a=np.loadtxt(p,delimiter=',',skiprows=1);a[1,2]=np.nan;np.savetxt(p,a,delimiter=',',header='time,x,drive',comments='')
    elif damage=='coverage': c['gradient_opt']['initial_time']=-1
    elif damage=='role': c['experiments'][1]['columns']=[*c['experiments'][1]['columns'][:2],{'name':'drive'}]
    elif damage=='observes': c['experiments'][0]['columns'][2]['observes']='x'
    else: c['experiments'][0]['columns'][2]['interpolation']='previous'
    (session/'inputs/user_input.yaml').write_text(yaml.safe_dump(c))
    with pytest.raises(ValidationError,match=match): validate_session(session)


def test_split_extraction_preserves_forcing_and_excludes_it_from_loss(tmp_path):
    session=tmp_path/'session';(session/'inputs').mkdir(parents=True)
    (session/'inputs/data.csv').write_text('time,x,drive\n0,0,0\n1,1,1\n')
    data=_assemble_split_new_session_response(filename_data='data.csv',csv_header=['time','x','drive'],
        parameters=[{'name':'k','min_value':.1,'max_value':4.,'logscale':False}],fixed_parameters=[],
        states_data={'states':[{'name':'x','initial_value':0.}]},
        equations_data={'forcing_columns':['drive'],'formulas':[],'rhs':[{'state':'x','expression':'k * drive'}]},
        observables_data={'observables':[{'measured':'x','simulated':'x','expression':''}]},
        loss_data={'custom_loss':True,'data_terms':[{'simulated':'x','measured':'x','metric':'rmse'}],'penalties':[]})
    init_session(session,FakeLLMClient([json.dumps(data)]),PromptRenderer())
    reader=parse_input_yaml(session/'inputs/user_input.yaml')
    assert reader.experiments[0]['columns'][2]['role']=='forcing'
    source=(session/'generated/user_model.py').read_text()
    assert 'drive = np.interp(t, t_eval, dataset[:, 1])' in source
    loss=source.split('def _compute_loss_problem')[1].split('def writeout_description')[0]
    assert 'dataset[:, 1]' not in loss
