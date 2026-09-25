# VM Handoff: Current Local-Agent pfit State

Date: 2026-09-25

This repo is the local-Ollama version of the pfit workflow. The intended user
workflow remains:

```text
pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose
```

The main experiment to run on the VM is whether a larger local model improves
the LLM-owned workflow steps, especially `pfit new` for large systems and
`pfit jax` for complex generated models.

## Current Implementation

- The workflow uses YAML only.
- Ollama is the only local LLM backend.
- `pfit new` is split into smaller extraction passes:
  dataset, parameters, states, equations, observables, and loss.
- `pfit check` combines deterministic validation with an LLM semantic review.
- `pfit jax` uses LLM translation and deterministic validation/smoke testing.
- Fitting uses the existing JAX/diffrax/optax backend under `lib/`.
- `INPUT_REQUIREMENTS.md` is the running contract for user prompt/data inputs.

Important guards currently implemented:

- CSV headers must exactly match observed states/observables.
- Fixed parameter values declared in the prompt are checked against generated
  YAML/model output.
- Explicit state initial values are checked against generated state initial
  values.
- Helper functions cannot close over undeclared constants.
- Loss checks catch missing RMSE/sqrt, missing log/log10 transforms, and missing
  normalization when the prompt explicitly asks for them.
- The loss renderer supports max-absolute-normalized RMSE.
- Large-model equation extraction has stricter prompt guidance, but this is not
  enough for NF-kB with the current 14B model.

## Current Model Baseline

Most recent local tests used:

```text
qwen2.5-coder:14b
```

This model can complete several examples, but it is not robust enough for the
largest biochemical case.

## Current Session Status

These cases have recently worked through `pfit new -> pfit check -> pfit jax`:

- `theophylline`
- `lotka_volterra`
- `robertson_session`
- `mapk_cascade`
- `vanderpol_session`
- `oregonator`
- `piezo_bouc_wen`
- `boehm_stat5`
- `ARC_fitting`
- `sliding_basepoint_headered`

`sliding_basepoint_headered` also reran end to end through `pfit run`, but fit
quality varied under the small default optimizer budget.

`ARC_fitting` runs reproducibly with a larger optimizer budget, but the fit is
still scientifically poor. It misses the rapid temperature/dTdt spike. This is
probably not just an optimizer-resource problem; it likely needs model/loss/bound
work.

`nfkb_signaling` currently fails at `pfit new`. The 14B model:

- changes a fixed parameter value,
- emits states as bare strings instead of objects with initial values,
- emits off-schema equations/observables/loss,
- and is rejected before generated files are written.

## Fit-Quality Snapshot

Recent latest losses from local runs:

| Session | Latest Outcome |
| --- | --- |
| `vanderpol_session` | good, loss around `2.43e-06` |
| `mapk_cascade` | good, loss around `6.56e-04` |
| `robertson_session` | good, loss around `1.746e-03` |
| `theophylline` | good, loss around `5.49e-03` |
| `piezo_bouc_wen` | acceptable-ish, loss around `4.07e-02` |
| `lotka_volterra` | moderate, loss around `7.07e-02` |
| `sliding_basepoint_headered` | runs, fit varies; recent losses `0.0775` and `0.1661` |
| `ARC_fitting` | reproducible but poor, loss around `0.1363` after larger budget |
| `oregonator` | poor, loss around `320`; likely model/loss/scaling issue |
| `nfkb_signaling` | fails at `pfit new`, no fit |

## VM Evaluation Script

Run:

```bash
bash scripts/run_vm_model_evaluation.sh qwen2.5-coder:32b
```

Useful environment variables:

```bash
export PFIT_LLM_BASE_URL=http://localhost:11434
export PFIT_EVAL_RUN_MODE=cheap   # none, cheap, or all
export PFIT_EVAL_DEBUG=0          # 1 streams LLM output into step logs
export PFIT_EVAL_MAX_TOKENS=12000
export PFIT_EVAL_TIMEOUT_SECONDS=600
```

The script copies sessions into a timestamped `evaluation_runs/` directory and
runs the workflow there, so the repo sessions are not mutated.

Default behavior:

- full suite: `pfit new -> pfit check -> pfit jax`
- selected run subset: `pfit run -> pfit diagnose`
- writes per-step logs and a Markdown/TSV summary

The main success criteria on the VM:

- Does `nfkb_signaling` pass `pfit new`?
- Does `boehm_stat5` remain stable through JAX translation?
- Does `sliding_basepoint_headered` produce stable generated dynamics and fit
  quality across reruns?
- Does a larger model reduce targeted repair frequency?
- Do generated losses preserve explicit user loss definitions without check
  failures?

## What To Inspect After Running

Look at:

```text
evaluation_runs/<timestamp>/summary.md
evaluation_runs/<timestamp>/summary.tsv
evaluation_runs/<timestamp>/<session>/*.log
evaluation_runs/<timestamp>/sessions/<session>/generated/agent_logs/
```

For successful fits, inspect:

```text
evaluation_runs/<timestamp>/sessions/<session>/outputs/<run-id>/fit_diagnosis.txt
evaluation_runs/<timestamp>/sessions/<session>/outputs/<run-id>/result_solution.csv
```

The script does not decide scientific correctness by itself; it reports workflow
success, repair counts, runtime, and final losses so the next agent can compare
the larger model against this 14B baseline.
