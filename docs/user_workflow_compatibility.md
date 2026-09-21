# User Workflow Compatibility

The local agent must mirror the remote-agent workflow:

```text
pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose
```

The commands correspond to the remote skill sequence:

```text
/pfit-new -> /pfit-check -> /pfit-jax -> /pfit-run -> /pfit-diagnose
```

Compatibility rules:

- Do not require a different local-only sequence.
- Keep the session folder structure unchanged.
- Keep `python fit_parameters.py <session_name>` working when `generated/generated_script.py` already exists.
- Keep `pfit run` as fitting only.
- Keep local LLM artifacts under `generated/agent_logs/`.
- Write fitting outputs from `pfit run` to `outputs/<run-id>/` so previous runs are preserved.
