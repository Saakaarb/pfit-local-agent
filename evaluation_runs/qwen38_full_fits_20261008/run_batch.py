"""Run the full CPU queues and push only their results on completion."""
from pathlib import Path
import os, subprocess, time
root=Path(__file__).resolve().parent
repo=root.parents[1]
env=dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
subprocess.run([str(repo/'.venv/bin/python'),str(repo/'scripts/run_qwen38_cpu_fits.py'),'run'],cwd=repo,env=env,check=True)
paths=[str(p.relative_to(repo)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
       and p.suffix not in ('.pyc','.pid','.lock') and p.name!='progress.log']
subprocess.run(['git','add','-f','--',*paths],cwd=repo,check=True)
subprocess.run(['git','commit','--only','-m','Record full Qwen3.8 CPU fits with 100 DE and 3000 Adam iterations','--',str(root.relative_to(repo))],cwd=repo,check=True)
for attempt in range(3):
    result=subprocess.run(['git','push','origin','main'],cwd=repo)
    if result.returncode==0:break
    if attempt==2:raise SystemExit(result.returncode)
    time.sleep(20)
