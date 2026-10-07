"""Run the final fresh Qwen3.8 batch, then save the three-model comparison."""
from pathlib import Path
import json, os, subprocess, sys, time
root=Path(__file__).resolve().parent
repo=root.parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',PYTHONUNBUFFERED='1')
command=[str(repo/'.venv/bin/python'),str(repo/'scripts/run_model_comparison.py'),
         '--root',str(root),'--models','qwen38_27b',
         '--report-models','qwen38_27b','qwen14b','qwen32b',
         '--first-case','ARC_fitting','--rotate-models']
result=subprocess.run(command,cwd=repo,env=env)
if result.returncode:
    raise SystemExit(result.returncode)
# Only this batch's artifacts; leave unrelated workspace changes alone.
files=[str(p.relative_to(repo)) for p in root.rglob('*') if p.is_file()
       and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.lock','.pid')
       and p.name not in ('active_stage.json','progress.log')]
subprocess.run(['git','add','-f','--',*files],cwd=repo,check=True)
subprocess.run(['git','commit','--only','-m','Record final fresh three-model smoke comparison', '--',str(root.relative_to(repo))],cwd=repo,check=True)
for attempt in range(3):
    pushed = subprocess.run(['git','push','origin','main'],cwd=repo)
    if pushed.returncode == 0:
        break
    if attempt == 2:
        raise SystemExit(pushed.returncode)
    time.sleep(20)
print('Finished comparison and pushed results.',flush=True)
