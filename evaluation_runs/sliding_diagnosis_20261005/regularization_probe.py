"""Diagnostic counterfactual only: smooth sign(v2); never change benchmark artifacts."""
import os
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
import json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from local_agent.agent.validators import parse_input_yaml
from lib.utils.experiments import load_experiments,experiment_constants
from lib.utils.run_artifacts import parameter_axes
from diffrax import RESULTS
out=[]
for case in ['sliding_basepoint','sliding_basepoint_headered']:
 session=REPO/'evaluation_runs/qwen38_comparison_20261005/cases'/case/'qwen38_27b/session'
 reader=parse_input_yaml(session/'inputs/user_input.yaml')
 record=load_experiments(session,reader)[0]
 constants=experiment_constants(record,reader)
 lo,hi,logs=parameter_axes(reader)
 constants.update(min_limits=lo,max_limits=hi,is_logscale=logs)
 source=(session/'generated/generated_script.py').read_text()
 assert source.count('jnp.sign(v2)')==1
 source=source.replace('jnp.sign(v2)','jnp.tanh(v2 / (0.01 * vf))')
 ns={'__name__':f'diagnostic_{case}'}
 exec(compile(source,'diagnostic_in_memory','exec'),ns)
 ts,ys,result,stats=ns['_integrate_system_with_stats'](constants,np.zeros(reader.n_search_axes))
 item=dict(case=case,diagnostic_only=True,change='Replace sign(v2) with tanh(v2/(0.01*vf)); all other settings unchanged',success=bool(result==RESULTS.successful),finite_rows=int(np.isfinite(np.asarray(ys)).all(axis=1).sum()),requested_rows=len(ts),stats={k:int(v) for k,v in stats.items()})
 out.append(item); print(json.dumps(item),flush=True)
(Path(__file__).parent/'regularization_probe.json').write_text(json.dumps(out,indent=2)+'\n')
