from pathlib import Path
import sys,json,shutil,yaml,importlib.util,subprocess,os
repo=Path.cwd();sys.path.insert(0,str(repo))
spec=importlib.util.spec_from_file_location('runner',repo/'scripts/run_qwen38_cpu_fits.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from lib.utils.source_stamp import verify_stamp,write_stamp
from lib.utils.yamlread import YAMLReader
from lib.utils.run_artifacts import load_accuracy_seeds,accuracy_seed_context
from local_agent.agent.readiness import check_ready
root=repo/'evaluation_runs/arc_extended_de_20261008';source=repo/'evaluation_runs/qwen38_full_fits_20261008/cases/ARC_fitting/session';session=root/'cases/ARC_fitting/session'
assert verify_stamp(source)[0] is True
reader=YAMLReader.from_file(source/'inputs/user_input.yaml');seeds=load_accuracy_seeds(source,reader);assert seeds
session.mkdir(parents=True,exist_ok=False)
for d in ['inputs','generated']:shutil.copytree(source/d,session/d,ignore=shutil.ignore_patterns('__pycache__','agent_logs'))
p=session/'inputs/user_input.yaml';original=yaml.safe_load(p.read_text());assert load_accuracy_seeds(session,YAMLReader.from_file(p))==seeds
config=r.full_fit_config(original,reader.n_search_axes,16);config['population_opt'].update(population_size=128,num_iters=300,random_seed=29)
r.check_optimizer_only_change(original,config);p.write_text(yaml.safe_dump(config,sort_keys=False));write_stamp(session)
p=session/'generated/solver_accuracy.json';accuracy=json.loads(p.read_text());accuracy['budget_source_seed_context']=accuracy['seed_context'];accuracy['seed_context']=accuracy_seed_context(session,YAMLReader.from_file(session/'inputs/user_input.yaml'));accuracy['budget_change_note']='Only whitelisted optimizer budgets, processors and random seed changed. Scientific model, solver and data unchanged.';r.save(p,accuracy)
check=check_ready(session);assert check.passed,check.critical_errors
cpus=[5,18,21,26,29,44,47,62,69,82,85,90,93,108,111,126];assert set(cpus)<=os.sched_getaffinity(0)
entry=dict(case='ARC_fitting',session=str(session),source=str(source),status='queued',parameters=reader.n_search_axes,population=config['population_opt'],gradient=config['gradient_opt'])
r.save(session.parent/'status.json',entry);r.save(root/'plan.json',dict(created_at=r.now(),cases=[entry],cpu_slots=[cpus],reason='User-requested larger ARC DE budget; seed29,128particles,300generations,3000Adam iterations. Original results preserved; no solver/scientific changes.'))
(root/'run.py').write_text('''from pathlib import Path
import importlib.util,json,sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('runner',REPO/'scripts/run_qwen38_cpu_fits.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.ROOT=ROOT
plan=json.loads((ROOT/'plan.json').read_text())
r.worker(plan['cases'],plan['cpu_slots'][0])
plan['finished_at']=r.now();r.save(ROOT/'plan.json',plan)
''')
shutil.copy('/tmp/prepare_arc_extended.py',root/'prepare.py')
with (root/'runner.log').open('w') as log:
 proc=subprocess.Popen([str(repo/'.venv/bin/python'),str(root/'run.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','JAX_PLATFORMS':'cpu'})
(root/'runner.pid').write_text(str(proc.pid));print('Launched',proc.pid,root)
