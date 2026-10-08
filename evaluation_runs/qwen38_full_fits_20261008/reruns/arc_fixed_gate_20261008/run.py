from pathlib import Path
import importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent;REPO=next(p for p in ROOT.parents if (p / '.git').exists());sys.path.insert(0,str(REPO))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=module('fit_runner',REPO/'scripts/run_qwen38_cpu_fits.py');r.ROOT=ROOT
plan=json.loads((ROOT/'plan.json').read_text());entry=plan['cases'][0];session=Path(entry['session']);status=session.parent/'status.json';cpus=plan['cpu_slots'][0]
if len(sys.argv)>1 and sys.argv[1]=='validate':
 import numpy as np
 import jax,jax.numpy as jnp
 import yaml
 from local_agent.agent.validators import validate_generated_script_contract
 new=module('gated_source',session/'generated/user_model.py');old=module('old_source',Path(entry['source'])/'generated/user_model.py')
 config=yaml.safe_load((session/'inputs/user_input.yaml').read_text())
 pars={p['name']:float(np.sqrt(float(p['min_val'])*float(p['max_val'])) if p['logscale'] else (float(p['min_val'])+float(p['max_val']))/2) for p in config['model']['trainable_parameters']}
 fixed={p['name']:float(p['value']) for p in config['model']['fixed_parameters']}
 for T in [354.,460.,470.,480.,600.]:
  y=np.array([.7,.2,T]);a=old.user_defined_system(0.,y,pars,fixed,None,None);b=new.user_defined_system(0.,y,pars,fixed,None,None)
  gate=np.exp(-np.logaddexp(0.,-(T-470.)/5.));np.testing.assert_allclose(b,[a[0],gate*a[1],abs(pars['h1']*a[0])+gate*abs(pars['h2']*a[1])],rtol=1e-12)
  np.testing.assert_allclose(new._observables(y[None,:],pars,fixed)['dTdt'][0],b[2],rtol=1e-12)
 g=lambda t:jnp.exp(-jnp.logaddexp(0.,-(t-470.)/5.))
 np.testing.assert_allclose(float(g(470.)),.5);np.testing.assert_allclose(float(jax.grad(g)(470.)),.05,rtol=1e-6)
 validate_generated_script_contract(session/'generated/generated_script.py')
 print('Fixed-gate RHS/observable and derivative checks passed',flush=True)
 recovery=module('recovery_runner',REPO/'evaluation_runs/qwen38_full_fits_20261008/reruns/targeted_refits_20261008/run_refits.py');recovery.validate(session)
 print('Fresh numerical validation passed',flush=True)
else:
 state=json.loads(status.read_text());assert state['status']=='prepared'
 state.update(status='validating',started_at=r.now());r.save(status,state)
 env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
 with (session.parent/'validation.log').open('w') as log:
  result=subprocess.run(['taskset','-c',','.join(map(str,cpus)),sys.executable,__file__,'validate'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  state.update(status='validation_failed',returncode=result.returncode);r.save(status,state);sys.exit(result.returncode)
 import yaml
 actual=yaml.safe_load((session/'inputs/user_input.yaml').read_text());state.update(status='queued',gradient=actual['gradient_opt'],population=actual['population_opt']);r.save(status,state)
 r.worker([entry],cpus);plan['finished_at']=r.now();r.save(ROOT/'plan.json',plan)
