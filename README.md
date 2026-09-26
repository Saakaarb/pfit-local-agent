# pfit-local-agent

Local LLM orchestration for the pfit ODE parameter-estimation workflow.

The local workflow is intentionally aligned with the remote-agent workflow:

```text
pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose
```

## Session Layout

```text
sessions/<session>/
  inputs/
    user_info.txt
    user_input.yaml
    data.csv
  generated/
    user_model.py
    generated_script.py
  outputs/
```

The legacy fitting entry point remains supported when `generated/generated_script.py` already exists:

```bash
python fit_parameters.py <session_name>
```

## Install

```bash
pip install -e ".[test]"
```

## Workflow

Create or prepare a session using the configured local LLM:

```bash
pfit new sessions/my_session
```

Check the session and write `generated/user_input_check.txt`:

```bash
pfit check sessions/my_session
```

Translate the model to `generated/generated_script.py` using the configured local LLM:

```bash
pfit jax sessions/my_session
```

Run fitting. Outputs are written to `outputs/<run-id>/`:

```bash
pfit run sessions/my_session
```

Restart only gradient refinement from a saved run (using the current session settings):

```bash
pfit run sessions/my_session gradient-only --from-run run_20260926_120000_123456
```

Omit `--from-run` to select the latest completed run. The source run is preserved.
New runs save named parameters and data/model/configuration snapshots. For old unnamed parameter CSVs, add `--allow-legacy-seed` only when their
parameter order matches the current YAML.

Post-fit sloppiness analysis runs automatically and saves `sloppiness_report.txt`,
`sloppiness.json`, `sloppiness_spectrum.png`, and Hessian/eigenspectrum CSVs. Use `--no-sloppiness` to skip it
or `--sloppiness-method finite-difference` to select finite differences of autodiff
gradients directly. The default tries second-order autodiff first. Diagnostic
failure does not discard fitted parameters.

Diagnose a completed run:

```bash
pfit diagnose sessions/my_session <run-id>
```

## Local LLM Config

Create `pfit.yaml`:

```yaml
llm:
  model: qwen2.5-coder:7b
  base_url: http://localhost:11434

workflow:
  max_repair_attempts: 5
  temperature: 0.1
  max_tokens: 12000
```

Environment variables can override config:

```bash
export PFIT_LLM_MODEL=qwen2.5-coder:7b
export PFIT_LLM_BASE_URL=http://localhost:11434
```

## Logs

Agent logs are written under:

```text
sessions/<session>/generated/agent_logs/
```

Fitting outputs from `pfit run` are written under:

```text
sessions/<session>/outputs/<run-id>/
```

## More Detail

```text
docs/local_llm_orchestration.md
docs/user_workflow_compatibility.md
```

## Tests

```bash
pytest -q
```

Implementation and interpretation: [restart and sloppiness](docs/restart_and_sloppiness.md).

Current implemented features and remaining divergences: [feature parity status](docs/feature_parity_status.md).


Readiness checks can run without Ollama:

```bash
pfit check sessions/my_session --deterministic-only
pfit check sessions/my_session --ready
```

The first checks inputs and the user model before translation; `--ready` also
checks the generated script and whether its model/YAML sources have changed.
`pfit run` repeats deterministic readiness checks automatically. Retranslate with
`pfit jax` after editing either source. See
[readiness notes](docs/deterministic_readiness.md) for scope and legacy behavior.


Multi-experiment fitting

List each CSV under `experiments` in the session YAML, with optional per-record
`initial_conditions`. All records share the model and parameter vector and must
have the same ordered observation columns. Each record can have its own time grid.
The fitting objective is the equal-weight mean of per-experiment losses.

The existing `new`, `check`, `jax`, `run`, `run ... gradient-only`, and `diagnose`
commands now cover every record. Multiple experiments produce
`result_solution_exp1.csv`, `result_solution_exp2.csv`, etc. Run snapshots and
standalone sloppiness include every dataset. Single-experiment filenames remain
compatible. See the [input contract](INPUT_REQUIREMENTS.md) and
[implementation and validation notes](docs/multi_experiment_support.md).
