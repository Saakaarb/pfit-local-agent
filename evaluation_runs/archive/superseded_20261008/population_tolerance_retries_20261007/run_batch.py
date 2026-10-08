from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
repo=root.parents[1]
for case in ('borghans_calcium','raia_il13'):
    subprocess.run([str(repo/'.venv/bin/python'),str(root/'retry_case.py'),str(root/'cases'/case)],cwd=repo,check=True)
