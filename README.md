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

For a repeatable development/test environment using the versions validated in
this repository, install the pip-native lock file instead:

```bash
pip install -r requirements-lock.txt
```

`pyproject.toml` remains the package and dependency definition;
`requirements-lock.txt` pins the complete resolved environment. Update the lock
deliberately whenever project dependencies change.

GitHub Actions installs this locked Python 3.12 environment, runs `pip check`,
and runs the default test suite on every push and pull request. Slow numerical
tests and live-Ollama tests remain excluded by the repository's pytest defaults.

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

Population evaluation uses JAX CPU devices. Before JAX is imported, the runner
detects CPUs available to the process (including Linux affinity limits) and
exposes `min(population_opt.processors, available CPUs)` devices. An explicit
`--xla_force_host_platform_device_count` in `XLA_FLAGS` overrides this choice.
The requested, detected, configured and actual JAX device counts are saved in
`run_manifest.json`.

Each run manifest also records best-effort reproducibility metadata: the Git
commit and dirty state, Python/platform and core package versions, optimizer
selection and random seed, the Ollama configuration visible at run time, and
SHA-256 hashes of the lock file and snapshotted configuration/model artifacts.
Missing metadata is reported in the manifest and never prevents a fit.

## Tests

```bash
pytest -q
```

Readiness checks can run without Ollama:

```bash
pfit check sessions/my_session --deterministic-only
pfit check sessions/my_session --ready
```

The first checks inputs and the user model before translation; `--ready` also
checks the generated script and whether its model/YAML sources have changed.
`pfit run` repeats deterministic readiness checks automatically. Retranslate with
`pfit jax` after editing either source.


Multi-experiment fitting

List each CSV under `experiments` in the session YAML, with optional per-record
`initial_conditions`. All records share the model and parameter vector and must
have the same ordered observation columns. Each record can have its own time grid.
The fitting objective is the equal-weight mean of per-experiment losses.

The existing `new`, `check`, `jax`, `run`, `run ... gradient-only`, and `diagnose`
commands now cover every record. Multiple experiments produce
`result_solution_exp1.csv`, `result_solution_exp2.csv`, etc. Run snapshots and
standalone sloppiness include every dataset. Single-experiment filenames remain
compatible. See the [input contract](INPUT_REQUIREMENTS.md).

Measured input histories are supported through `role: forcing` columns with
per-experiment linear interpolation.
`pfit diagnose SESSION [RUN]` computes snapshot-based evidence and asks the
configured Ollama model to interpret it. Use `--probe-gradients` for AD/FD checks
and `--deterministic-only` to work offline.
Supplied losses and output functions are preserved independently, with numerical
source/JAX checks before translation acceptance.
