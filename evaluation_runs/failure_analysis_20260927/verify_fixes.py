import sys,json,shutil,types
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import yaml,numpy as np,jax,jax.numpy as jnp
from local_agent.agent.validators import parse_input_yaml,import_generated_script,smoke_test_generated_script
from lib.utils.experiments import load_experiments,experiment_constants
from lib.utils.run_artifacts import parameter_axes,scale_parameters
root=Path('evaluation_runs/failure_analysis_20260927');out={}
for name in ['qwen32b','qwen14b','qwen3_coder_next']:
 src=Path('evaluation_runs/model_comparison_20260927/cases/oregonator')/name/'session';dst=root/'oregonator_50000'/name
 dst.mkdir(parents=True,exist_ok=True)
 shutil.copytree(src/'inputs',dst/'inputs',dirs_exist_ok=True);(dst/'generated').mkdir(exist_ok=True)
 shutil.copy2(src/'generated/user_model.py',dst/'generated/user_model.py')
 p=dst/'inputs/user_input.yaml';d=yaml.safe_load(p.read_text());d['gradient_opt']['max_steps']=50000;p.write_text(yaml.safe_dump(d,sort_keys=False))
 script=dst/'generated/generated_script.py';script.write_text((src/'generated/generated_script.py').read_text().replace('max_steps=10000,','max_steps=50000,'))
 try:smoke_test_generated_script(script,dst);out[name]={'smoke_and_fidelity':'passed'}
 except Exception as exc:out[name]={'error':str(exc)}
 print(name,out[name],flush=True)
src=Path('evaluation_runs/model_comparison_20260927/cases/cascaded_tanks/qwen32b/session')
reader=parse_input_yaml(src/'inputs/user_input.yaml');module=import_generated_script(src/'generated/generated_script.py');record=load_experiments(src,reader)[0];c=experiment_constants(record,reader);lo,hi,logs=parameter_axes(reader);c.update(min_limits=lo,max_limits=hi,is_logscale=logs)
point=scale_parameters(np.loadtxt(next((src/'outputs').glob('run_*/final_design_point.csv'))),reader)
v,g=jax.value_and_grad(lambda x:module._compute_loss_problem(c,x))(jnp.array(point));t,y,result=module._integrate_system(c,point)
tanks=dict(original_loss=float(v),original_gradient=np.asarray(g).tolist(),minimum_saved_states=np.min(np.asarray(y),axis=0).tolist(),solver_result=str(result))
code=(src/'generated/generated_script.py').read_text();code=code.replace('jnp.sqrt(jnp.maximum(x1, 0))','safe_sqrt_positive(x1)').replace('jnp.sqrt(jnp.maximum(x2, 0))','safe_sqrt_positive(x2)')
helper='\ndef safe_sqrt_positive(x):\n    positive = x > 0\n    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)\n'
code=code.replace('jax.config.update("jax_enable_x64", True)','jax.config.update("jax_enable_x64", True)'+helper)
ns=types.ModuleType('tank_probe');exec(compile(code,'tank_probe','exec'),ns.__dict__)
v2,g2=jax.value_and_grad(lambda x:ns._compute_loss_problem(c,x))(jnp.array(point));t2,y2,result2=ns._integrate_system(c,point)
tanks.update(safe_branch_loss=float(v2),safe_branch_gradient=np.asarray(g2).tolist(),max_trajectory_difference=float(np.max(np.abs(np.asarray(y)-np.asarray(y2)))),safe_branch_solver_result=str(result2))
out['cascaded_tanks']=tanks
print('TANKS',json.dumps(tanks),flush=True)
(root/'fix_verification.json').write_text(json.dumps(out,indent=2)+'\n')
