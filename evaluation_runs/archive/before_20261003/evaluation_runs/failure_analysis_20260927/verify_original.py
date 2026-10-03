import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from local_agent.agent.jax_fragments import parse_jax_fragments_response,render_generated_script_from_fragments
from local_agent.agent.session_spec import load_session_spec
from local_agent.agent.validators import smoke_test_generated_script
out={}
for name in ['qwen32b','qwen14b','qwen3_coder_next']:
 src=Path('evaluation_runs/model_comparison_20260927/cases/oregonator')/name/'session/generated/agent_logs/llm_calls.jsonl'
 records=[json.loads(s) for s in src.read_text().splitlines()]
 first=next(r for r in records if r['step']=='repair_jax_fragments')
 text=first['messages'][-1]['content'].split('Previous fragment response:',1)[1].lstrip()
 data,_=json.JSONDecoder().raw_decode(text)
 dst=Path('evaluation_runs/failure_analysis_20260927/oregonator_50000')/name
 spec=load_session_spec(dst/'inputs/user_input.yaml')
 fragments=parse_jax_fragments_response(json.dumps(data),spec)
 script=dst/'generated/generated_script.py'
 script.write_text(render_generated_script_from_fragments(fragments,spec))
 try:
  smoke_test_generated_script(script,dst);out[name]='passed'
 except Exception as e:out[name]=str(e)
 print(name,out[name],flush=True)
Path('evaluation_runs/failure_analysis_20260927/original_fragments_50000.json').write_text(json.dumps(out,indent=2)+'\n')
