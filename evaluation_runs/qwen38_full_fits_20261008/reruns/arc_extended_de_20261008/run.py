from pathlib import Path
import importlib.util,json,sys
ROOT=Path(__file__).resolve().parent
REPO=next(p for p in ROOT.parents if (p / '.git').exists())
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('runner',REPO/'scripts/run_qwen38_cpu_fits.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.ROOT=ROOT
plan=json.loads((ROOT/'plan.json').read_text())
r.worker(plan['cases'],plan['cpu_slots'][0])
plan['finished_at']=r.now();r.save(ROOT/'plan.json',plan)
