"""Evaluate archived acceleration expressions; no LLM calls or parameter fitting."""
import ast,json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
pars=dict(c2=10**5.5,Dk=.1,Dc=.55,m1=1000.,m2=10.,vf=.1,k=1.29e6)
results=[]
for case in ['sliding_basepoint','sliding_basepoint_headered']:
 for root,model in [('model_comparison_20261003','qwen32b'),('qwen38_comparison_20261005','qwen38_27b')]:
  path=REPO/'evaluation_runs'/root/'cases'/case/model/'session/generated/user_model.py'
  tree=ast.parse(path.read_text())
  expression=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dv2dt' for t in node.targets))
  code=compile(ast.Expression(expression),str(path),'eval')
  points=[]
  for force_fraction in [.5,.99,2.]:
   force=force_fraction*pars['c2']
   for velocity in [-1e-8,0.,1e-8,.2]:
    scope=dict(pars,x1=force/pars['k'],x2=0.,v2=velocity,np=np)
    actual=float(eval(code,{'__builtins__':{}},scope))
    expected=0. if abs(force)<pars['c2'] and abs(velocity)<pars['vf'] else (force-pars['c2']*np.sign(velocity))/pars['m2']
    points.append(dict(force_over_c2=force_fraction,v2=velocity,extracted_acceleration=actual,prompt_acceleration=float(expected)))
  results.append(dict(case=case,model=model,source=str(path.relative_to(REPO)),expression=ast.unparse(expression),points=points))
(out/'point_probes.json').write_text(json.dumps(results,indent=2)+'\n')
for result in results:
 print(result['case'],result['model'])
 for x in result['points']:
  if x['force_over_c2']==.99 or (x['force_over_c2']==2 and x['v2']==0):print(x)
