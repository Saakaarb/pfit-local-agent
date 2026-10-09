from pathlib import Path
import importlib.util,json,sys,subprocess,os,re,shutil,yaml
ROOT=Path(__file__).resolve().parent;REPO=next(p for p in ROOT.parents if (p/'.git').exists());sys.path.insert(0,str(REPO))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=load('fit_runner',REPO/'scripts/run_qwen38_cpu_fits.py');r.ROOT=ROOT
session=ROOT/'cases/sneyd_ipr/session';status=session.parent/'status.json'
if sys.argv[1]=='prepare':
 from lib.utils.source_stamp import verify_stamp
 source=ROOT.parent/'sneyd_extended_20261009/cases/sneyd_ipr/session';assert verify_stamp(source)[0] is True
 session.mkdir(parents=True,exist_ok=False)
 for folder in ['inputs','generated']:shutil.copytree(source/folder,session/folder,ignore=shutil.ignore_patterns('__pycache__','agent_logs'))
 archive=session.parent/'original_validation';archive.mkdir()
 for p in (session/'generated').glob('*.json'):shutil.move(p,archive/p.name)
 path=session/'inputs/user_input.yaml';config=yaml.safe_load(path.read_text());assert config['gradient_opt']['max_steps']==1000;config['gradient_opt']['max_steps']=5000;path.write_text(yaml.safe_dump(config,sort_keys=False))
 script=session/'generated/generated_script.py';code,n=re.subn(r'max_steps=1000\b','max_steps=5000',script.read_text());assert n==1;script.write_text('# pfit-sources: pending=true\n'+code)
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)==config['population_opt']['processors']
 entry=dict(case='sneyd_ipr',session=str(session),source=str(source),status='prepared',parameters=14,population=config['population_opt'],gradient=config['gradient_opt'])
 r.save(status,entry);r.save(ROOT/'plan.json',dict(created_at=r.now(),cases=[entry],cpu_slots=[cpus],change='Only max_steps raised1000->5000; same Tsit5, tolerances, nine experiments, bounds, model, objective, DE320x300 seed29 and Adam3000. No automatic tolerance calibration or further budget increase for this comparison.'))
 print('Prepared fixed5000-step comparison')
elif sys.argv[1]=='validate':
 import numpy as np
 from lib.utils.source_stamp import write_stamp
 from lib.utils.run_artifacts import accuracy_seed_context,parameter_axes
 from lib.utils.experiments import load_experiments,experiment_constants
 from local_agent.agent.validators import parse_input_yaml,validate_generated_script_contract,import_generated_script
 from local_agent.agent.solver_coverage import assess_solver_coverage
 from local_agent.agent.solver_accuracy import assess_solver_accuracy
 from lib.utils.translation_fidelity import compare_source_and_jax
 from local_agent.agent.readiness import check_ready
 script=session/'generated/generated_script.py';validate_generated_script_contract(script);m=import_generated_script(script);path=session/'inputs/user_input.yaml';reader=parse_input_yaml(path);records=load_experiments(session,reader);config=yaml.safe_load(path.read_text());g=config['gradient_opt'];pop=config['population_opt']
 points=assess_solver_coverage(m,reader,session,script,records,g)
 lo,hi,logs=parameter_axes(reader)
 for record in records:
  c=experiment_constants(record,reader);c.update(min_limits=lo,max_limits=hi,is_logscale=logs)
  loss=np.asarray(m._compute_loss_problem(c,points[0]));assert loss.shape==() and np.isfinite(loss) and loss!=reader.error_loss
  out=np.asarray(m._write_problem_result(c,points[0]));assert out.ndim==2 and len(out)==len(record['t_eval'])
 compare_source_and_jax(m,reader,session,script,parameter_points=points)
 accuracy=assess_solver_accuracy(m,reader,session,script,records,g,pop,{})
 assert accuracy.get('validated_seeds');write_stamp(session);accuracy['seed_context']=accuracy_seed_context(session,reader);r.save(script.parent/'solver_accuracy.json',accuracy)
 ready=check_ready(session);assert ready.passed,ready.critical_errors
 print('Fresh validation passed at5000steps with unchanged tolerances',flush=True)
else:
 plan=json.loads((ROOT/'plan.json').read_text());entry=plan['cases'][0];cpus=plan['cpu_slots'][0];state=json.loads(status.read_text());assert state['status']=='prepared';state.update(status='validating',started_at=r.now());r.save(status,state)
 env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
 with (session.parent/'validation.log').open('w') as log:
  result=subprocess.run(['taskset','-c',','.join(map(str,cpus)),sys.executable,__file__,'validate'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  state.update(status='validation_failed',returncode=result.returncode);r.save(status,state);sys.exit(result.returncode)
 state['status']='queued';r.save(status,state);r.worker([entry],cpus);plan['finished_at']=r.now();r.save(ROOT/'plan.json',plan)
