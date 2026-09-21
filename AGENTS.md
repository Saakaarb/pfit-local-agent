# Project Instructions

This repo is a local-agent version of the pfit workflow. Keep future work aligned
with these invariants.

- Preserve the user-facing workflow:
  `pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose`.
- Use YAML only. Do not reintroduce XML inputs, XML parsers, or XML workflow
  steps.
- Use Ollama only for local LLM orchestration.
- Keep implementation minimal and close to the corresponding `pfit-claude`
  functionality.
- Prefer existing `lib/` functionality over adding duplicate layers elsewhere.
- Avoid adding prompts, abstractions, or workflow steps unless they are required
  by the pfit writeup or existing `pfit-claude` behavior.
- `pfit jax` must include an LLM translation step. Do not treat Python-to-JAX
  conversion as fully deterministic.
- Use deterministic validation after LLM output. LLMs may draft fragments, but
  validation decides whether generated files are accepted.
- Support derived quantities and transforms as general workflow features, not
  example-specific patches.
- Keep tests focused on pfit workflow behavior and core fitting functionality.
