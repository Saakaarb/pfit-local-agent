# pfit-local-agent

`pfit-local-agent` is a local version of the pfit workflow for fitting ordinary
differential equation models to data. It keeps the same command-line flow as the
remote-agent version, but uses an Ollama-served local coding model for the
model-assisted steps.

The workflow is:

```text
pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose
```

The repository is built around a simple contract: the local model may draft
configuration and code, but deterministic validation decides whether generated
artifacts are accepted. A session remains inspectable at each stage through the
generated YAML, readable Python model, JAX script, run snapshots, optimizer
logs, fitted trajectories, and diagnostic reports.

## What It Does

The workflow starts from a session directory containing a mathematical problem
statement and one or more CSV data files. It then:

- extracts a structured YAML fitting configuration;
- generates a readable NumPy model implementation;
- checks the session for input, data, loss, and mapping issues;
- translates the accepted model to differentiable JAX code;
- runs bounded global search followed by gradient-based refinement;
- writes preserved run directories with fitted parameters and trajectories;
- produces post-fit diagnostics, including sloppiness analysis when enabled.

The workflow supports observed states, derived observables, measured forcing
inputs, uncertainty columns, custom losses, multi-experiment fits, stiff
systems, and repeated fitting runs from saved artifacts.

## Install

Use a fresh virtual environment. The tested dependency set is pinned in
`requirements-lock.txt`; `pyproject.toml` remains the package definition.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

This installs the `pfit` command-line entry point and the numerical stack used by
the workflow, including JAX, Diffrax, Optax, and the Ollama Python client.

## Ollama Setup

Install Ollama separately, start the Ollama service, and pull the model you want
to use:

```bash
ollama pull qwen3:30b-a3b
```

The helper script can also prepare Ollama and pull a model:

```bash
scripts/setup_ollama.sh qwen3:30b-a3b
```

Create `pfit.yaml` in the repo root:

```yaml
llm:
  model: qwen3:30b-a3b
  base_url: http://localhost:11434

workflow:
  max_repair_attempts: 5
  temperature: 0.1
  max_tokens: 12000
```

Environment variables can override the config:

```bash
export PFIT_LLM_MODEL=qwen3:30b-a3b
export PFIT_LLM_BASE_URL=http://localhost:11434
```

The local model is needed for `pfit new`, semantic `pfit check`, `pfit jax`, and
optional semantic diagnosis. Once a session has an accepted YAML configuration
and generated JAX script, `pfit run` can be repeated without calling the local
model.

## Session Layout

A new session starts with user-supplied input files:

```text
sessions/<session>/
  inputs/
    user_info.txt
    data.csv
```

After running the workflow, the session contains generated artifacts and
preserved fitting runs:

```text
sessions/<session>/
  inputs/
    user_info.txt
    user_input.yaml
    data.csv
  generated/
    user_model.py
    generated_script.py
    agent_logs/
  outputs/
    run_<timestamp>/
```

For the input contract expected by `pfit new`, see
[INPUT_REQUIREMENTS.md](INPUT_REQUIREMENTS.md).

## Workflow

Create or regenerate a session from the user statement and CSV files:

```bash
pfit new sessions/my_session
```

Check the generated YAML, readable model, data mappings, and loss:

```bash
pfit check sessions/my_session
```

Translate the accepted model to JAX:

```bash
pfit jax sessions/my_session
```

Run parameter estimation:

```bash
pfit run sessions/my_session
```

Restart only gradient refinement from a saved run:

```bash
pfit run sessions/my_session gradient-only --from-run run_20260926_120000_123456
```

Diagnose a completed run:

```bash
pfit diagnose sessions/my_session <run-id>
```

Use deterministic-only checks when Ollama is unavailable:

```bash
pfit check sessions/my_session --deterministic-only
pfit check sessions/my_session --ready
pfit diagnose sessions/my_session <run-id> --deterministic-only
```

`pfit run` repeats readiness checks before fitting. If the YAML or source model
changes, rerun `pfit jax` before fitting again.

The legacy fitting entry point remains available when a session already has
`generated/generated_script.py`:

```bash
python fit_parameters.py <session_name>
```

## Outputs

Each run directory records the accepted configuration, generated script, fitted
parameters, optimizer logs, fitted trajectories, run manifest, and diagnostic
files. Post-fit sloppiness analysis writes `sloppiness_report.txt`,
`sloppiness.json`, `sloppiness_spectrum.png`, and Hessian/eigenspectrum CSVs
unless disabled with `--no-sloppiness`.

The run manifest records best-effort reproducibility metadata, including Git
state, Python/platform information, optimizer settings, visible Ollama config,
and hashes of the lock file and snapshotted model artifacts.

## Examples

Curated example sessions are stored under `sessions/`. Each retained example
keeps its input prompt and data, generated model files, and at least one
completed run. These sessions are meant to show the expected file layout and the
artifacts produced by the workflow.

## Tests

Run the default test suite with:

```bash
pytest -q
```

The default tests avoid live-Ollama and slow numerical runs unless explicitly
enabled.
