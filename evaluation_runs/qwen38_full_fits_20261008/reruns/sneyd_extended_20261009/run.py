from pathlib import Path
import importlib.util,json,sys,subprocess,os,yaml
ROOT=Path(__file__).resolve().parent
REPO=next(p for p in ROOT.parents if (p/'.git').exists());sys.path.insert(0,str(REPO))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=load('fit_runner',REPO/'scripts/run_qwen38_cpu_fits.py');r.ROOT=ROOT
plan=json.loads((ROOT/'plan.json').read_text());entry=plan['cases'][0];session=Path(entry['session']);status=session.parent/'status.json';cpus=plan['cpu_slots'][0]
if len(sys.argv)>1 and sys.argv[1]=='validate':
 from local_agent.agent.validators import validate_generated_script_contract
 validate_generated_script_contract(session/'generated/generated_script.py')
 recovery=load('recovery_runner',ROOT.parent/'targeted_refits_20261008/run_refits.py');recovery.validate(session)
 print('Fresh numerical validation passed',flush=True)
else:
 state=json.loads(status.read_text());assert state['status']=='prepared'
 state.update(status='validating',started_at=r.now());r.save(status,state)
 env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
 with (session.parent/'validation.log').open('w') as log:
  result=subprocess.run(['taskset','-c',','.join(map(str,cpus)),sys.executable,__file__,'validate'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  state.update(status='validation_failed',returncode=result.returncode);r.save(status,state);sys.exit(result.returncode)
 config=yaml.safe_load((session/'inputs/user_input.yaml').read_text());state.update(status='queued',population=config['population_opt'],gradient=config['gradient_opt']);r.save(status,state)
 plan['run_framework_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();r.save(ROOT/'plan.json',plan)
 r.worker([entry],cpus);plan.update(finished_at=r.now());r.save(ROOT/'plan.json',plan)
