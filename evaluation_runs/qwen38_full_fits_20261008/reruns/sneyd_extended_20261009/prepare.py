from pathlib import Path
import importlib.util,sys,json,shutil,os,re,yaml,subprocess
ROOT=Path(__file__).resolve().parent
REPO=next(p for p in ROOT.parents if (p/'.git').exists());sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('fit_runner',REPO/'scripts/run_qwen38_cpu_fits.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from lib.utils.source_stamp import verify_stamp
from lib.utils.yamlread import YAMLReader
from local_agent.agent.solver_selection import select_session_integrator
source=ROOT.parent/'targeted_refits_20261008/cases/sneyd_ipr/session';session=ROOT/'cases/sneyd_ipr/session'
assert verify_stamp(source)[0] is True
session.mkdir(parents=True,exist_ok=False)
for folder in ['inputs','generated']:shutil.copytree(source/folder,session/folder,ignore=shutil.ignore_patterns('__pycache__','agent_logs'))
archive=session.parent/'original_validation';archive.mkdir()
for p in (session/'generated').glob('*.json'):shutil.move(p,archive/p.name)
p=session/'inputs/user_input.yaml';original=yaml.safe_load(p.read_text());cpus=sorted(os.sched_getaffinity(0));reader=YAMLReader.from_file(p)
config=r.full_fit_config(original,reader.n_search_axes,len(cpus));config['population_opt'].update(population_size=320,num_iters=300,random_seed=29);r.check_optimizer_only_change(original,config);p.write_text(yaml.safe_dump(config,sort_keys=False))
selection=select_session_integrator(session);updated=yaml.safe_load(p.read_text());assert updated['gradient_opt']['integrator']==original['gradient_opt']['integrator']
old=original['gradient_opt']['max_steps'];new=updated['gradient_opt']['max_steps'];script=session/'generated/generated_script.py';code,n=re.subn(r'max_steps='+str(old)+r'\b','max_steps='+str(new),script.read_text());assert n==1
code='\n'.join(l for l in code.splitlines() if not l.startswith('# pfit-sources:'));script.write_text('# pfit-sources: pending=true\n'+code+'\n')
assert (session/'generated/user_model.py').read_bytes()==(source/'generated/user_model.py').read_bytes()
entry=dict(case='sneyd_ipr',session=str(session),source=str(source),status='prepared',parameters=reader.n_search_axes,population=updated['population_opt'],gradient=updated['gradient_opt'])
r.save(session.parent/'status.json',entry);r.save(ROOT/'plan.json',dict(created_at=r.now(),cases=[entry],cpu_slots=[cpus],framework_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),step_multiplier=100,memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),previous_best_loss=.02505909149215776,reason='User-requested broader Sneyd search: 320particles x300generations, seed29; Adam3000. Same scientific model, nine datasets, bounds and objective. New 100x time-scale estimate and fresh numerical validation.'))
print('Sneyd max_steps',old,'->',new,'CPUs',len(cpus),'experiments',len(updated['experiments']))
