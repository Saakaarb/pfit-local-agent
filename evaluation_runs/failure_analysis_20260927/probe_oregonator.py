import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import jax.numpy as jnp
import diffrax
from local_agent.agent.validators import parse_input_yaml,import_generated_script
from lib.utils.experiments import load_experiments,experiment_constants
from lib.utils.run_artifacts import parameter_axes
root=Path('evaluation_runs/model_comparison_20260927/cases/oregonator')
results=[]
for model in ['qwen32b','qwen14b','qwen3_coder_next']:
 session=root/model/'session';reader=parse_input_yaml(session/'inputs/user_input.yaml');module=import_generated_script(session/'generated/generated_script.py')
 lo,hi,logs=parameter_axes(reader);record=load_experiments(session,reader)[0];c=experiment_constants(record,reader);c.update(min_limits=lo,max_limits=hi,is_logscale=logs)
 for budget in ([10000,50000] if model=='qwen32b' else [10000]):
  start=time.monotonic();x=jnp.zeros(reader.n_search_axes)
  sol=diffrax.diffeqsolve(diffrax.ODETerm(module.user_defined_system),diffrax.Kvaerno5(),t0=c['init_time'],t1=c['t_eval'][-1],dt0=c['init_timestep'],y0=c['init_cond'],args={'constants':c,'trainable_variables':x},saveat=diffrax.SaveAt(ts=c['t_eval']),stepsize_controller=diffrax.PIDController(rtol=c['stepsize_rtol'],atol=c['stepsize_atol']),max_steps=budget,throw=False)
  row=dict(model=model,max_steps=budget,result=str(sol.result),stats={k:int(v) for k,v in sol.stats.items()},finite_rows=int(np.all(np.isfinite(np.asarray(sol.ys)),axis=1).sum()),rows=len(sol.ys),physical_midpoint=np.asarray(module.unscale_value(x,lo,hi,logs)).tolist(),seconds=time.monotonic()-start)
  if np.all(np.isfinite(sol.ys)):row['loss']=float(module._compute_loss_value(c,x,sol.ts,sol.ys))
  print(json.dumps(row),flush=True);results.append(row)
Path('evaluation_runs/failure_analysis_20260927/oregonator_probe.json').write_text(json.dumps(results,indent=2)+'\n')
