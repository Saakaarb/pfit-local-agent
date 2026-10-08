"""User-authorized direct ARC model edit; no LLM generation."""
from pathlib import Path
import ast, json, shutil, sys, yaml
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1];sys.path.insert(0,str(REPO))
SOURCE=REPO/'evaluation_runs/arc_extended_de_20261008/cases/ARC_fitting/session'
SESSION=ROOT/'cases/ARC_fitting/session'
from lib.utils.source_stamp import verify_stamp
assert verify_stamp(SOURCE)[0] is True
SESSION.mkdir(parents=True,exist_ok=False)
for folder in ['inputs','generated']:
 shutil.copytree(SOURCE/folder,SESSION/folder,ignore=shutil.ignore_patterns('__pycache__','agent_logs'))
archive=SESSION.parent/'original_validation';archive.mkdir()
for p in (SESSION/'generated').glob('*.json'):shutil.move(p,archive/p.name)
path=SESSION/'inputs/user_input.yaml';original=yaml.safe_load(path.read_text());config=yaml.safe_load(path.read_text())
config['model']['fixed_parameters'] += [dict(name='T_gate',value=470.0),dict(name='deltaT_gate',value=5.0)]
assert config['model']['trainable_parameters']==original['model']['trainable_parameters']
path.write_text(yaml.safe_dump(config,sort_keys=False))
for filename,prefix in [('user_model.py','np'),('generated_script.py','jnp')]:
 p=SESSION/'generated'/filename;before=p.read_text();after=before
 for line in ['    T = solution[:, 2]\n','    T = y[2]\n']:
  assert after.count(line)==1
  after=after.replace(line,line+f"    gate = {prefix}.exp(-{prefix}.logaddexp(0.0, -(T - fixed_parameters['T_gate']) / fixed_parameters['deltaT_gate']))\n")
 rate=f'A2 * {prefix}.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2'
 assert after.count(rate)==3
 after=after.replace(rate,'gate * '+rate)
 # Objective code, state layout, solver and optimizer logic are unchanged.
 a={n.name:ast.dump(n) for n in ast.parse(before).body if isinstance(n,ast.FunctionDef)}
 b={n.name:ast.dump(n) for n in ast.parse(after).body if isinstance(n,ast.FunctionDef)}
 assert {k for k in a if a[k]!=b[k]}=={'_observables','user_defined_system'}
 if filename=='generated_script.py':after='# pfit-sources: pending=true\n'+'\n'.join(l for l in after.splitlines() if not l.startswith('# pfit-sources:'))+'\n'
 p.write_text(after)
p=SESSION/'inputs/user_info.txt';s=p.read_text();s=s.replace('- kb = 1.38e-23','- kb = 1.38e-23\n- T_gate = 470.0 K (fixed)\n- deltaT_gate = 5.0 K (fixed)');s=s.replace('- r2(c2,T) = A2', '- g(T) = 1 / (1 + exp(-(T - T_gate) / deltaT_gate))\n- r2(c2,T) = g(T) * A2');p.write_text(s)
cpus=[5,18,21,26,29,44,47,62,69,82,85,90,93,108,111,126]
entry=dict(case='ARC_fitting',session=str(SESSION),source=str(SOURCE),status='prepared',parameters=8,population=config['population_opt'],gradient=config['gradient_opt'])
(SESSION.parent/'status.json').write_text(json.dumps(entry,indent=2)+'\n')
(ROOT/'plan.json').write_text(json.dumps(dict(cases=[entry],cpu_slots=[cpus],change='Only fixed temperature gate on reaction2 and its heat contribution. T_gate470K,deltaT_gate5K. Same DE128x300 seed29, Adam3000, original parameterization/bounds/objective. Fresh deterministic validation required.'),indent=2)+'\n')
print(ROOT)
