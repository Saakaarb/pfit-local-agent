# Ollama workflow comparison

Compare Qwen2.5-Coder 32B and 14B (Q4_K_M), and Qwen3-Coder-Next
80B total / approximately 3B active (Unsloth UD-Q3_K_XL). All three are
non-thinking models. Exact downloaded digests, byte sizes, GGUF parameter
counts, templates, capabilities and loaded GPU memory are captured at runtime.
Publisher descriptions and URLs are in `models.json`.

## Input audit

The criterion is **human writability**: a scientist can supply prose, equations,
parameter bounds, starting conditions, measurement definitions and a loss in a
text file. Equations need not be valid Python. No framework YAML, reference
Python, generated implementation or previous model responses are supplied.

The thirteen complete session specifications are retained verbatim. They use
ordinary scientific notation, including powers, derivatives and piecewise
equations. The three fixture loss-only notes were expanded into full scientific
specifications using their existing scientific definitions; those drafts were
audited against the fixture equations and declarations. They are benchmark
author inputs, not independently collected user submissions. No malformed-input
robustness or user-study claim is made.

Decay and Sneyd CSV copies receive their declared column headers without changing
numerical rows. NF-kB's two-column CSV agrees with its text prompt; its stale
source YAML is not provided. The HH folder lacks a scientific specification and
session1 has no inputs: both remain visibly blocked, with no invented task.

There are 16 eligible rows, including two duplicate variants (Robertson/test_session
and the sliding-basepoint pair), hence 14 distinct scientific tasks. Duplicates
remain visible and must not inflate claims about independent problem coverage.
`input_audit.json` records every case, preparation change and SHA-256 digest.

## Fixed protocol

- Freeze prompts and data before any model batch. Start from text and CSV only.
- One attempt per case/model, including up to five existing new/JAX repairs.
  No manual intervention or model-specific prompt tuning during scoring.
- Run `new -> check -> jax -> run -> diagnose`. Both code and YAML are generated
  fresh. After new, archive extracted YAML and explicitly override only fitting
  resource settings: DE population 4, one iteration, one CPU process, seed 7;
  Adam, five iterations. Preserve scientific declarations and solver choices.
- Common LLM options: temperature 0.1, seed 7, context 32,768, output cap 12,000,
  top-k 40, top-p 0.9; request timeout 600 seconds. These are controlled workflow
  settings, not individually tuned publisher recommendations. Fixed seeds do not
  guarantee bitwise reproducibility across GPU kernels or model architectures.
- Stage wall limits: new/JAX/fit 1,800 seconds, check/diagnosis 900 seconds.
  Numerical fitting uses CPU and JAX float64. Sloppiness remains enabled.
- Models are warmed once before their batches; downloads and warm-up time are
  separate from case timing. Capture Ollama load/evaluation durations as well as
  process wall times. Context caching may still affect timing. Require full GPU
  residency before scoring a model. No concurrent case/model jobs.
- Record partial timing for failures, stage/exit status, repair calls, token
  counts, errors, diagnosis acceptance, final loss and experiment output count.
  Success requires all stages, the correct experiment files, finite outputs and
  loss, completed gradient refinement and accepted diagnosis. Framework checks
  include semantic review and source/JAX fidelity; this is not an independent
  proof of all scientific equations. Five iterations do not establish fit quality.

The benchmark worker instruments the Ollama client at runtime for telemetry and
common options. Production framework and prompts remain unchanged. A single
trial measures observed outcomes, not a model's success probability. The larger
candidate changes architecture, generation and quantization as well as total
parameter count, so this is a practical comparison, not a scaling experiment.

## Artifacts and rotation

Persistent root: `/workspace/pfit-local-agent/evaluation_runs/model_comparison_20260927`.
Each `cases/<case>/` has `qwen32b/`, `qwen14b/`, and `qwen3_coder_next/` containing
its logs, `metadata.json`, request-level `llm_metrics.jsonl`, and a separate
`session/` containing only workflow inputs, generated code and output runs.
Benchmark records must remain outside `session/`: intake reads session files
recursively. Frozen inputs and model-level metadata live at the root.
`comparison.md` and `comparison.csv` update after each completed case.

Run 32B first, then 14B, then Coder-Next. Complete each model batch before removing
its weights. Only these known comparison model tags may be removed. Keep logs
and results, verify free disk before pulling, and restore the configured 32B
model after the experiment. The 50GB volume requires this rotation; the 36.3GB
larger-model quantization leaves approximately 4GB for other experiment growth.

```bash
source /workspace/pfit-env.sh
.venv/bin/python scripts/prepare_model_comparison.py --root evaluation_runs/new_comparison
.venv/bin/python scripts/run_model_comparison.py --root evaluation_runs/new_comparison \
  --rotate-models --restore-current-model
```

Preparation requires a new directory. The runner skips recorded case attempts
on restart and refuses a different framework commit or changed frozen input.
Interrupted attempts should be retained as interrupted, not silently retried.

## Preliminary batch excluded

The first harness revision put metadata/logs at the session root. Inspection of
the recorded model requests showed these benchmark files were included in the
LLM input context. The preliminary batch is retained under
`evaluation_runs/model_comparison_20260927_preliminary` and excluded from the
clean comparison. Every model starts afresh with the corrected session boundary;
the frozen scientific prompts and CSVs are unchanged. The observed extraction
errors remain useful diagnostic evidence, but are not clean benchmark scores.

## Prior RMSE correction rerun

Revision `68baaa8` was tested with fresh Boehm and Oregonator extraction in
`evaluation_runs/rmse_contract_live_20260927`. Boehm passed all five stages.
Oregonator passed new and check with the corrected uncertainty-weighted RMSE,
but failed JAX integration smoke tests with a sentinel loss after five repairs.
The loss mismatch is resolved in that observed run; integration readiness is a
separate unresolved outcome. No fix is inserted between model batches.
