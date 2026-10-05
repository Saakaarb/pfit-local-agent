"""Freeze both existing benchmark cohorts for a new Qwen3.8 attempt per case."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    imports = []
    for cohort, relative in [('original', 'evaluation_runs/model_comparison_20261003'),
                              ('external', 'evaluation_runs/external_comparison_20261004')]:
        source = REPO / relative
        for original in json.loads((source / 'input_audit.json').read_text()):
            case = dict(original, cohort=cohort, source_benchmark=relative)
            case.setdefault('sloppiness', True)
            frozen = source / 'frozen_inputs' / case['name']
            for name, expected in case.get('input_hashes', {}).items():
                if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Original frozen input hash mismatch: {frozen / name}')
            if frozen.exists():
                shutil.copytree(frozen, root / 'frozen_inputs' / case['name'])
            for model in ['qwen32b', 'qwen14b']:
                old = source / 'cases' / case['name'] / model / 'metadata.json'
                entry = json.loads(old.read_text())
                if entry['status'] not in ('pass', 'fail', 'degraded', 'blocked'):
                    raise ValueError(f'Baseline is incomplete: {old}')
                dest = root / 'cases' / case['name'] / model
                dest.mkdir(parents=True)
                shutil.copyfile(old, dest / 'metadata.json')
                imports.append(dict(case=case['name'], model=model,
                    source=str(old.relative_to(REPO)),
                    sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                    note='Metadata reference only; full artifacts remain at source benchmark.'))
            cases.append(case)
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Case name collision across cohorts')
    for filename, value in [('input_audit.json', cases), ('baseline_imports.json', imports)]:
        (root / filename).write_text(json.dumps(value, indent=2) + '\n')
    (root / 'BENCHMARK_NOTES.md').write_text('''# Qwen3.8 27B benchmark protocol

Run all 16 runnable original cases and all six external cases, without revising
prompts or manually repairing earlier failures. Two original folders remain
blocked because they lack scientific specifications. Two original tasks are
duplicate variants; 22 runnable rows represent 20 independent scientific tasks.

Only Qwen3.8 is run in this batch. The 32B/14B columns reference existing single
attempts. `baseline_imports.json` gives their complete artifact locations and
metadata hashes; baseline subdirectories contain only copied metadata.

Frozen scientist-style prompts and CSVs are byte-identical to their respective
prior comparison. Population size 4, one DE iteration, five Adam iterations,
one population worker, CPU/float64 numerics. Sloppiness follows each original
cohort: enabled for the original cases, disabled for the six external cases.
Schwen and Borghans retain the explicit zero integration-start-time setup.

Qwen3.8 uses Ollama tag qwen3.8:27b, Q4_K_M, with explicit think=true and the
model's default reasoning effort. Context 32768, generation limit 12000, seed 7,
temperature 0.1, top_k 40 and top_p 0.9 match the earlier batches. The generation
limit includes reasoning; truncation remains a recorded failure, not an automatic
budget increase. Request timeout 600 seconds and existing repair limits remain.
Warm-up disables thinking solely to obtain a short response; task calls enable it.

Exact digest, actual size/parameter count, template, model defaults, GPU placement,
preparation time and Ollama version are captured in models/. Per-call metrics
include requested thinking mode, returned thinking-character count, token counts
and timing. Per-stage and total case times include failed attempts and repairs.
The 32B and 14B models are non-thinking: this compares deployed model configurations,
not parameter count alone. Hardware details are recorded for each batch; times
should not be treated as a controlled throughput comparison across VM allocations.

Production extraction and fitting code is unchanged for this batch. A smoke-test
pass establishes workflow completion, not scientific correctness or convergence.
''')
    print(f'Prepared {sum(c["status"] != "blocked" for c in cases)} runnable cases and '
          f'{sum(c["status"] == "blocked" for c in cases)} blocked folders in {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root.resolve())
