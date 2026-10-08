from pathlib import Path
import os,json,sys,re,types
import numpy as np
import jax,jax.numpy as jnp,optax
REPO=next(p for p in Path(__file__).resolve().parents if (p / '.git').exists());sys.path.insert(0,str(REPO))
from lib.utils.yamlread import YAMLReader
from lib.utils.experiments import load_experiments
from lib.utils.helper_functions import CreatedClass
from lib.utils.run_artifacts import parameter_axes,scale_parameters
ROOT=Path(__file__).resolve().parent
SOURCE=REPO/'evaluation_runs/qwen38_full_fits_20261008'
results={}
for name in ['oregonator','lotka_volterra','sneyd_ipr','ARC_fitting']:
 status=json.loads((SOURCE/'cases'/name/'status.json').read_text());session=SOURCE/'cases'/name/'session';run=Path(status['run_dir'])
 reader=YAMLReader.from_file(session/'inputs/user_input.yaml');records=load_experiments(session,reader)
 code=(session/'generated/generated_script.py').read_text()
 parameters=json.loads((run/'final_parameters.json').read_text())['parameters'];point=scale_parameters(np.clip([p['value'] for p in parameters],reader.min_axis_values,reader.max_axis_values),reader)
 def build(steps=None,profile='gradient'):
  module=types.ModuleType('probe_'+name+'_'+str(steps));script=code
  if steps is not None:script=re.sub(r'max_steps=\d+',f'max_steps={steps}',script)
  exec(compile(script,str(session/'generated/generated_script.py'),'exec'),module.__dict__)
  problem=CreatedClass(experiments=records,input_reader=reader,compute_loss_problem=module._compute_loss_problem,write_problem_result=module._write_problem_result)
  low,high,_=parameter_axes(reader);problem.set_min_limit(low);problem.set_max_limit(high);problem.set_is_logscale(reader.axis_logscale)
  if profile=='population':
   for c in problem.constants_list:
    c['stepsize_rtol']=np.asarray(reader.population_stepsize_rtol or reader.stepsize_rtol)
    c['stepsize_atol']=np.asarray(reader.population_stepsize_atol or reader.stepsize_atol)
  return module,problem
 if name=='oregonator':
  _,p=build(profile='population');v,g=jax.value_and_grad(p._compute_loss)(jnp.asarray(point));optimizer=optax.adam(1e-4);u,_=optimizer.update(g,optimizer.init(jnp.asarray(point)));next_point=np.asarray(jnp.clip(point+u,-1,1));rows=[]
  for steps in [2000,8000,16000]:
   for profile in ['gradient','population']:
    m,p=build(steps,profile)
    for label,x in [('winner',point),('first_adam_update',next_point)]:
     val=float(p._compute_loss(x));_,ys,res,stats=m._integrate_system_with_stats(p.constants_list[0],jnp.asarray(x))
     row=dict(max_steps=steps,profile=profile,point=label,loss=val,result=str(res),stats={k:int(z) for k,z in stats.items()});rows.append(row);print(name,row,flush=True)
  results[name]=dict(source=str(run),initial_gradient=np.asarray(g).tolist(),checks=rows)
 else:
  _,p=build();v,g=jax.value_and_grad(p._compute_loss)(jnp.asarray(point));g=np.asarray(g); projected=np.asarray(point)-np.clip(np.asarray(point)-g,-1,1)
  fd=[]
  for i in range(len(point)):
   a=point.copy();b=point.copy();a[i]=max(-1,a[i]-1e-4);b[i]=min(1,b[i]+1e-4)
   fd.append((float(p._compute_loss(b))-float(p._compute_loss(a)))/(b[i]-a[i]))
  results[name]=dict(source=str(run),loss=float(v),normalized_parameters=point.tolist(),gradient=g.tolist(),finite_difference_gradient=fd,projected_gradient_norm=float(np.linalg.norm(projected)),active_bounds=[reader.trainable_parameter_names[i] for i,x in enumerate(point) if abs(x)>.999])
  print(name,json.dumps(results[name]),flush=True)
 (ROOT/'diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
