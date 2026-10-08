"""Isolated diagnostic reruns; reuse the established full-fit runner."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('cpu_fits',REPO/'scripts/run_qwen38_cpu_fits.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);runner.ROOT=ROOT
SOURCE=REPO/'evaluation_runs/qwen38_full_fits_20261008'
from lib.utils.source_stamp import verify_stamp,write_stamp
from lib.utils.yamlread import YAMLReader
from lib.utils.run_artifacts import load_accuracy_seeds,accuracy_seed_context
from local_agent.agent.readiness import check_ready

def prepare():
 entries=[]
 for name,population in [('lotka_volterra',128),('sneyd_ipr',160),('oregonator',128),('ARC_fitting',128)]:
  source=SOURCE/'cases'/name/'session';session=ROOT/'cases'/name/'session'
  assert verify_stamp(source)[0] is True
  source_reader=YAMLReader.from_file(source/'inputs/user_input.yaml')
  seeds=load_accuracy_seeds(source,source_reader);assert seeds
  session.mkdir(parents=True,exist_ok=False)
  for folder in ('inputs','generated'):
   shutil.copytree(source/folder,session/folder,ignore=shutil.ignore_patterns('__pycache__','agent_logs'))
  path=session/'inputs/user_input.yaml';original=yaml.safe_load(path.read_text())
  assert load_accuracy_seeds(session,YAMLReader.from_file(path))==seeds
  config=runner.full_fit_config(original,source_reader.n_search_axes,8)
  config['population_opt'].update(population_size=population,random_seed=17)
  runner.check_optimizer_only_change(original,config)
  if name=='oregonator':
   config['gradient_opt']['max_steps']=8000
   script=session/'generated/generated_script.py'
   code,n=re.subn(r'max_steps=2000\b','max_steps=8000',script.read_text());assert n==1
   script.write_text(code)
  path.write_text(yaml.safe_dump(config,sort_keys=False));write_stamp(session)
  accuracy_path=session/'generated/solver_accuracy.json'
  if name=='oregonator':
   # Archive inherited reports; never treat old seeds as validated under new settings.
   archive=session.parent/'original_validation';archive.mkdir()
   for report in ('solver_accuracy.json','solver_coverage.json','solver_diagnostics.json','tolerance_calibration.json','translation_fidelity.json'):
    p=session/'generated'/report
    if p.exists():shutil.move(p,archive/report)
  else:
   accuracy=json.loads(accuracy_path.read_text());accuracy['budget_source_seed_context']=accuracy['seed_context']
   accuracy['seed_context']=accuracy_seed_context(session,YAMLReader.from_file(path))
   accuracy['budget_change_note']='Only optimizer population, independent random seed and CPU count changed; scientific model and solver settings unchanged.'
   runner.save(accuracy_path,accuracy)
   assert check_ready(session).passed
  entry=dict(case=name,session=str(session),source=str(source),status='queued',parameters=source_reader.n_search_axes,
             population=config['population_opt'],gradient=config['gradient_opt'],requires_validation=name=='oregonator',
             original_result=json.loads((source.parent/'status.json').read_text())['run_dir'])
  runner.save(session.parent/'status.json',entry);entries.append(entry)
 slots=[[3,7,20,23,67,71,84,87],[27,31,46,55,91,95,110,119]]
 runner.save(ROOT/'plan.json',dict(created_at=runner.now(),cases=entries,cpu_slots=slots,
  note='Fresh seed 17, broader DE populations; 100 generations and 3000 Adam iterations retained. Oregonator ceiling raised to 8000 following exact first-update reproduction. Other ongoing fits retain their CPU allocation.'))

def validate(session):
 from local_agent.agent.validators import smoke_test_generated_script, SolverValidationError
 from local_agent.agent.tolerance_calibration import apply_calibrated_tolerances
 script=session/'generated/generated_script.py'
 while True:
  try:
   smoke_test_generated_script(script,session)
   break
  except SolverValidationError as exc:
   path=session/'inputs/user_input.yaml';config=yaml.safe_load(path.read_text());g=config['gradient_opt']
   failure=exc.diagnostics
   recoverable=(failure['code']=='step_limit' or
    (failure['code']=='coverage_below_target' and failure.get('failure_counts',{}).get('step_limit',0)>0) or
    (failure['code']=='accuracy_inconclusive' and failure.get('failure_cause')=='step_limit'))
   old=g['max_steps'];budget=min(g['solver_recovery_max_steps'],old*2)
   if not recoverable or budget<=old:raise
   g['max_steps']=budget;path.write_text(yaml.safe_dump(config,sort_keys=False))
   code,n=re.subn(r'max_steps='+str(old)+r'\b','max_steps='+str(budget),script.read_text());assert n==1
   script.write_text(code);write_stamp(session)
   print(f'Numerical recovery: {old} -> {budget} steps; same model and fixed sample count',flush=True)
 apply_calibrated_tolerances(session,script);write_stamp(session)
 path=script.parent/'solver_accuracy.json';accuracy=json.loads(path.read_text());assert accuracy.get('validated_seeds')
 accuracy['seed_context']=accuracy_seed_context(session,YAMLReader.from_file(session/'inputs/user_input.yaml'))
 runner.save(path,accuracy);report=check_ready(session)
 if not report.passed:raise RuntimeError(str(report.critical_errors))

def queue(entries,cpus):
 for entry in entries:
  session=Path(entry['session']);status_path=session.parent/'status.json'
  if entry['requires_validation']:
   state=json.loads(status_path.read_text());state.update(status='validating',started_at=runner.now());runner.save(status_path,state)
   env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JAX_ENABLE_X64='true',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
   with (session.parent/'validation.log').open('w') as log:
    result=subprocess.run(['taskset','-c',','.join(map(str,cpus)),sys.executable,__file__,'validate',str(session)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:
    state.update(status='validation_failed',returncode=result.returncode);runner.save(status_path,state);continue
   state['status']='queued';runner.save(status_path,state)
  runner.worker([entry],cpus)

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='validate':validate(Path(sys.argv[2]))
 else:
  plan=json.loads((ROOT/'plan.json').read_text());(ROOT/'runner.pid').write_text(str(os.getpid()))
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(queue,plan['cases'][i::2],cpus) for i,cpus in enumerate(plan['cpu_slots'])]
   for future in futures:future.result()
  plan.update(finished_at=runner.now());runner.save(ROOT/'plan.json',plan)
