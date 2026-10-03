"""Freeze human-writable specifications and CSVs, without generated answers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / 'benchmarks/model_comparison'
MODELS = ['qwen32b', 'qwen14b', 'qwen3_coder_next']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, models=None):
    models = MODELS if models is None else models
    if not models or any(model not in MODELS for model in models):
        raise ValueError("Select at least one known comparison model")
    root.mkdir(parents=True, exist_ok=False)
    cases = [(p.name, p, False) for p in sorted((REPO / 'sessions').iterdir()) if p.is_dir()]
    cases += [(p.name, p, True) for p in sorted((REPO / 'tests/fixtures').iterdir())
              if (p / 'inputs').is_dir()]
    inventory = []
    for name, source, fixture in cases:
        entry = dict(name=name, source=str(source.relative_to(REPO)), adaptations=[],
                     status='ready', expected_experiments=None)
        inventory.append(entry)
        for model in models:
            (root / 'cases' / name / model).mkdir(parents=True)
        prompt = SPEC / 'prompts' / (name + '.txt')
        source_prompt = source / 'inputs/user_info.txt'
        if not source_prompt.exists():
            entry.update(status='blocked', reason='No scientific problem specification; not invented for benchmark')
            continue
        if not prompt.exists():
            # Preserve the existing scientist-facing specification verbatim.
            prompt.write_text(source_prompt.read_text())
        inputs = root / 'frozen_inputs' / name
        inputs.mkdir(parents=True)
        shutil.copy2(prompt, inputs / 'user_info.txt')
        for path in (source / 'inputs').glob('*.csv'):
            shutil.copy2(path, inputs / path.name)
        config = yaml.safe_load((source / 'inputs/user_input.yaml').read_text())
        entry['expected_experiments'] = len(config['experiments'])
        entry['expected_data_files'] = sorted(e['data_file'] for e in config['experiments'])
        entry['prompt_sha256'] = digest(prompt)
        entry['original_prompt_sha256'] = digest(source_prompt)
        entry['prompt_audit'] = ('Expanded loss-only notes into scientific equations, bounds, initial '
            'conditions, observations, inputs and experiment definitions. No Python or YAML supplied.'
            if fixture else 'Existing prose/math specification retained verbatim; no framework schema required.')
        if name == 'nfkb_signaling':
            entry['prompt_audit'] += ' CSV has two columns matching the prompt; stale source YAML is not supplied.'
        if name == 'test_session':
            entry['duplicate_of'] = 'robertson_session'
        if name == 'sliding_basepoint_headered':
            entry['duplicate_of'] = 'sliding_basepoint'
        for experiment in config['experiments']:
            path = inputs / experiment['data_file']
            original = path.read_text()
            try:
                float(original.splitlines()[0].split(',')[0])
            except ValueError:
                continue
            columns = [c['name'] for c in experiment['columns']]
            if len(columns) != len(original.splitlines()[0].split(',')):
                raise ValueError(f'Header width mismatch: {path}')
            entry['adaptations'].append(dict(file=path.name, change='Add declared CSV header; numerical rows unchanged',
                                            original_sha256=digest(path)))
            path.write_text(','.join(columns) + '\n' + original)
        entry['input_hashes'] = {p.name: digest(p) for p in sorted(inputs.iterdir())}
    (root / 'input_audit.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (SPEC / 'input_audit.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.models)
