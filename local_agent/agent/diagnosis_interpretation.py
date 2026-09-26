"""Ollama interpretation of computed diagnosis evidence; never applies edits."""
import json
from pathlib import Path
from local_agent.agent.config import WorkflowConfig
from local_agent.agent.prompts import PromptRenderer
from local_agent.agent.checks import _complete_with_log
from local_agent.agent.llm_json import parse_llm_json_object
from lib.utils.run_store import run_config


def interpret_diagnosis(session, output, llm_client, workflow_config=None):
    output=Path(output)
    config=workflow_config or WorkflowConfig()
    report=json.loads((output/'scientific_diagnosis.json').read_text())
    ids={f['id'] for f in report['findings']}
    for field in ('parameters','experiments','optimizer','optimizer_traces','limitations','loss','gradient_probe','sloppiness','seed_history'):
        if report.get(field) is not None and report.get(field) != [] and report.get(field) != {}:
            ids.add(field.upper())
    snapshot=output/'snapshot'
    yaml_path=run_config(output,session) if snapshot.is_dir() else None
    source=snapshot/'generated/user_model.py'
    context={'evidence_ids':', '.join(sorted(ids)), 'scientific_report':json.dumps(report,indent=2),
             'input_yaml':yaml_path.read_text() if yaml_path and yaml_path.exists() else 'No snapshot configuration available.',
             'user_model':source.read_text() if source.exists() else 'No snapshot source available.'}
    messages=PromptRenderer().render_messages('diagnose_run.system.md','diagnose_run.user.md',context)
    try:
        raw=_complete_with_log(llm_client,output,'scientific_diagnosis',messages,
                               temperature=config.temperature,max_tokens=min(config.max_tokens,4096))
        result=parse_llm_json_object(raw,'diagnosis')
        if not isinstance(result.get('verdict'),str) or not result['verdict'].strip():
            raise ValueError('Diagnosis must provide a verdict')
        recommendations=result.get('recommendations')
        limitations=result.get('limitations')
        if not isinstance(recommendations,list) or not isinstance(limitations,list) or any(not isinstance(x,str) for x in limitations):
            raise ValueError('Diagnosis recommendations/limitations must be lists')
        for item in recommendations:
            if not isinstance(item,dict) or any(not isinstance(item.get(k),str) or not item[k].strip() for k in ('action','reason')):
                raise ValueError('Every recommendation needs an action and reason')
            references=item.get('evidence_ids')
            if not isinstance(references,list) or not references or any(not isinstance(x,str) or x not in ids for x in references):
                raise ValueError('Recommendation cites missing or unknown evidence IDs')
        result['status']='ok'
        result['interpretation_only']=True
    except Exception as exc:
        result={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
    (output/'ollama_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
