import json, os, sys, shutil
from pathlib import Path
repo=Path('/workspace/pfit-local-agent');sys.path.insert(0,str(repo/'scripts'))
from run_model_comparison import stage,save,utc
root=repo/'evaluation_runs/sliding_manual_sticking_20261007';session=root/'session';entry=json.loads((root/'metadata.json').read_text())
run=sorted((session/'outputs').glob('run_*'))[-1];fit=json.loads((run/'fit_summary.json').read_text())
entry.update(status='running',resumed_utc=utc(),resume_reason='Pod restart; completed fit retained',run_id=run.name,fit_summary=fit)
save(root/'metadata.json',entry)
old=root/'logs/diagnose.log'
if old.exists():shutil.copy2(old,root/'logs/diagnose.before_pod_restart.log')
args=['diagnose',str(session),run.name,'--model','qwen3.8:27b','--base-url','http://127.0.0.1:11434','--timeout-seconds','600','--max-tokens','12000','--temperature','0.1']
try:
 result=stage(root,'diagnose',args,1800,dict(os.environ,OPENBLAS_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true'))
 entry['stages']['diagnose']=result
 if result['status']!='pass':entry.update(status='fail',failed_stage='diagnose')
 else:
  diagnosis=json.loads((run/'ollama_diagnosis.json').read_text());entry['diagnosis_status']=diagnosis['status'];entry['status']='pass' if diagnosis['status']=='ok' and fit['termination']=='iteration_budget' and not fit.get('refinement_error') else 'degraded'
except Exception as exc:entry.update(status='fail',failed_stage='diagnose',error=str(exc))
entry.update(finished_utc=utc(),total_wall_seconds=sum(v.get('seconds',0) for v in entry['stages'].values()),timing_note='Sum of completed stages, excluding pod downtime and interrupted diagnosis')
save(root/'metadata.json',entry);print(entry['status'],flush=True)
