# Local LLM Orchestration

The local implementation keeps the same user-facing workflow as the remote-agent tool:

```bash
pfit new sessions/demo
pfit check sessions/demo
pfit jax sessions/demo
pfit run sessions/demo
pfit diagnose sessions/demo <run-id>
```

The deterministic harness owns validation, file writes, logs, fitting, and diagnosis. The local LLM is used inside `pfit new` to extract a structured study spec; deterministic renderers then write `inputs/user_input.xml` and `generated/user_model.py`. The local LLM is used inside `pfit jax` to translate `generated/user_model.py` into `generated/generated_script.py` and repair that script when validation fails.

## Commands

`pfit new`

Calls the configured local LLM, consumes user-supplied notes/data/files already present in the session directory, and asks for parameters, states, RHS expressions, data-column mapping, and notes. XML and Python are rendered deterministically and written only after validation accepts the draft.

When rerun with `--overwrite`, previous generated artifacts and logs are cleared before context is collected. The prompt context excludes `generated/`, `outputs/`, and prior `inputs/user_input.xml` so iterative runs use the user's source material rather than stale model output.

`pfit check`

Validates XML, data, declared variables, and numerical settings. Writes:

```text
generated/user_input_check.txt
```

`pfit jax`

Calls the configured local LLM, writes:

```text
generated/generated_script.py
```

Then runs static contract validation and a runtime smoke test. Repair attempts are made when validation fails.

`pfit run`

Runs the existing fitting engine using the generated script. Outputs are written to:

```text
outputs/<run-id>/
```

`pfit diagnose`

Writes:

```text
outputs/<run-id>/fit_diagnosis.txt
```

## Local LLM Config

Example `pfit.yaml`:

```yaml
llm:
  model: qwen2.5-coder:7b
  base_url: http://localhost:11434

workflow:
  max_repair_attempts: 5
  temperature: 0.1
  max_tokens: 12000
```

## Logs

Generation logs are written to:

```text
generated/agent_logs/
```

Key files:

```text
workflow_events.jsonl
llm_calls.jsonl
```
