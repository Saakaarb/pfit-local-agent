# Migration Plan: Local LLM Orchestration

## Goal

Overhaul this repo so that a local LLM hosted through Ollama orchestrates the parameter-fitting workflow instead of relying on Claude/Codex skills or remote LLM execution.

The fitting engine should remain deterministic and testable. The local LLM should be used for narrow code-generation and repair tasks inside a validation-driven workflow.

## Target Architecture

Split the repo into three layers:

1. Core fitting engine
   - XML parsing
   - Dataset loading
   - Problem object construction
   - PSO optimization
   - NODE/L-BFGS refinement
   - Output writing

2. Workflow/orchestration layer
   - Session discovery
   - Input validation
   - Prompt rendering
   - Local LLM calls
   - Generated-code validation
   - Repair loops
   - Fitting execution
   - Logs and summaries

3. Ollama client layer
   - Ollama client
   - Fake/test client

## Proposed Repo Structure

```text
pfit-local-agent/
  fit_parameters.py
  local_agent/
    core/
      xmlread.py
      problem.py
      fitting.py
      generated_contract.py
    algorithms/
      pso.py
      node.py
    agent/
      orchestrator.py
      state.py
      workflow.py
      validators.py
      codegen.py
      repair.py
      prompts.py
    llm/
      base.py
      ollama.py
    cli/
      main.py
  local_agent/
    agent/
      prompt_templates/
        generated_script.system.md
        generated_script.user.md
        repair_generated_script.system.md
        repair_generated_script.user.md
  sessions/
  tests/
```

Keep `fit_parameters.py` as a compatibility wrapper during migration.

## Desired CLI

The local workflow must match the remote-agent workflow:

```bash
pfit new sessions/my_session
pfit check sessions/my_session
pfit jax sessions/my_session
pfit run sessions/my_session
pfit diagnose sessions/my_session <run-id>
```

The JAX translation step is the only step that calls a local LLM:

```bash
pfit jax sessions/my_session --model qwen2.5-coder:7b
```

Useful `pfit jax` flags:

```text
--model MODEL_NAME
--base-url URL
--max-repair-attempts 5
```

## Workflow State Machine

The orchestrator should own the workflow, not the LLM.

1. Discover session
   - Check `inputs/user_input.xml`.
   - Check data file exists.
   - Check optional user pseudocode/model file exists.
   - Create `generated/` and `outputs/` if needed.

2. Validate XML
   - Parse XML with `XMLReader`.
   - Check required sections.
   - Check parameter counts and names.
   - Check dataset columns match model/data declarations.
   - Fail early with actionable errors before calling the LLM.

3. Generate or repair user model skeleton
   - Use the local LLM only if needed.
   - Convert XML into a user-editable model skeleton.
   - Save to `generated/user_model.py`.

4. Generate JAX-compatible script
   - Prompt local LLM with:
     - XML-derived schema
     - User model or pseudocode
     - Generated-script contract
     - Strict required function names
   - Save to `generated/generated_script.py`.

5. Static validation
   - Parse with `ast`.
   - Verify required imports/functions exist:
     - `user_defined_system`
     - `_integrate_system`
     - `_compute_loss_problem`
     - `_write_problem_result`
   - Verify no markdown fences or prose leaked into the file.
   - Verify expected function signatures.

6. Runtime smoke test
   - Import `generated_script.py`.
   - Run a small integration/loss call using session inputs.
   - Check the result is finite or a controlled `error_loss`.
   - If failure occurs, send traceback plus source context to the repair prompt.

7. Repair loop
   - Local LLM receives:
     - Generated script
     - Error traceback
     - Validation failure
     - Relevant XML/data schema
   - It returns a full replacement script or patch.
   - Repeat up to `max_repair_attempts`.

8. Fit
   - Call existing fitting workflow.
   - Write logs and outputs.

9. Summarize
   - Report final parameters.
   - Report fitting loss if available.
   - Report generated/modified files.
   - Preserve intermediate logs.

## LLM Provider Interface

Define one internal interface:

```python
class LLMClient:
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...
```

Client implementations:

- `OllamaClient`
  - Calls `http://localhost:11434/api/chat`.
  - Supports models such as `qwen2.5-coder:7b`, `deepseek-coder-v2`, and `llama3.1`.

- `FakeLLMClient`
  - Used in tests.
  - Returns deterministic canned outputs.

## Prompt Migration

Keep local LLM prompt templates inside `local_agent/agent/prompt_templates/`.

Initial prompt files:

```text
local_agent/agent/prompt_templates/
  generated_script.system.md
  generated_script.user.md
  repair_generated_script.system.md
  repair_generated_script.user.md
```

Prompts should enforce:

- Output only Python code where needed.
- No Markdown fences.
- Required function names.
- JAX compatibility.
- Use `diffrax.Kvaerno5`.
- Use `jax.config.update("jax_enable_x64", True)`.
- Treat trainable parameters as vectors.
- Treat fixed parameters as a dict.
- Preserve XML ordering.
- Match the generated-script function contract exactly.

Local models will need shorter, stricter prompts than remote frontier models. Validators and repair loops should carry most of the reliability burden.

## Validation Strategy

The repo should be validation-driven. Do not trust the local LLM output directly.

Add validators for:

- XML structure.
- Parameter name uniqueness.
- Declared parameter count consistency.
- Dataset file existence.
- Dataset shape and numeric loadability.
- Required generated Python functions.
- Generated Python syntax.
- Generated function signatures.
- Importability of `generated_script.py`.
- One-step or small-run loss computation.
- Output array shape from `_write_problem_result`.
- No Markdown/prose in generated code.

## Config File

Add repo-level and optional session-level config:

```yaml
llm:
  model: qwen2.5-coder:7b
  base_url: http://localhost:11434

workflow:
  max_repair_attempts: 5
  temperature: 0.1
  max_tokens: 12000
```

Potential paths:

```text
pfit.yaml
sessions/<session>/pfit.yaml
```

Session config should override repo config.

## Testing Strategy

The current tests are useful as smoke tests for the existing repo, but the overhaul needs a substantially revamped test suite. The new system will have more moving parts: deterministic fitting code, session validation, prompt rendering, local LLM adapters, generated-code validation, repair loops, and CLI orchestration. Tests should be rebuilt around those boundaries instead of only checking that sample sessions run.

Add three categories of tests:

1. Unit tests
   - XML parsing.
   - Dataset validation.
   - Generated-script validators.
   - Provider request formatting.

2. Prompt rendering tests
   - Given XML and user model input, rendered prompt contains required sections.
   - Avoid brittle full-string prompt tests.

3. Workflow tests
   - Use `FakeLLMClient`.
   - Return known generated code.
   - Validate full orchestration without requiring Ollama.

Local LLM integration tests should be optional and skipped unless explicitly enabled.

## Test Suite Revamp

The revamped test suite should be organized by system boundary.

### Core Fitting Tests

- Test XML parsing independently from fitting.
- Test parameter scaling and unscaling.
- Test log-scale parameter behavior.
- Test PSO initialization without requiring long optimization runs.
- Test NODE initialization and bounds clipping.
- Test fitting output file creation with small deterministic sessions.
- Keep slow numerical optimization tests marked separately.

### Session Validation Tests

- Missing `inputs/user_input.xml`.
- Missing dataset file.
- Malformed XML.
- Duplicate trainable/fixed/state variable names.
- Mismatched `N_TRAINABLE_PARAMETERS`.
- Invalid logscale values.
- Non-numeric dataset values.
- Dataset with too few columns.
- Dataset shape inconsistent with expected model output.
- Missing required optimizer settings.

### Generated Script Contract Tests

- Valid generated script passes.
- Missing required function fails.
- Wrong function signature fails.
- Markdown-fenced output fails.
- Syntax error fails.
- Import error fails with useful diagnostics.
- `_compute_loss_problem` smoke test succeeds on a known fixture.
- `_write_problem_result` returns the expected shape.

### Prompt Rendering Tests

- Generated-script prompt includes XML-derived parameter ordering.
- Generated-script prompt includes fixed parameter names.
- Repair prompt includes traceback and previous generated code.
- Prompts do not accidentally omit required function names.
- Tests should check critical prompt sections, not exact full prompt text.

### LLM Client Tests

- Ollama request payload formatting.
- Ollama error handling.
- Timeout handling.
- Malformed Ollama response handling.
- Fake client behavior for deterministic orchestration tests.

### Orchestrator Tests

- Check workflow.
- JAX translation workflow.
- Full run using `FakeLLMClient`.
- Failed generation followed by successful repair.
- Repair attempts exhausted.
- Existing generated script reused when requested.
- Logs written to expected locations.
- Outputs are not silently overwritten except where intended.

### CLI Tests

- `pfit new`.
- `pfit check`.
- `pfit jax`.
- `pfit run`.
- `pfit diagnose`.
- Config loading from `pfit.yaml`.
- Session config overriding repo config.
- Helpful error messages for missing model config.

### Optional Local LLM Integration Tests

These should not run by default in CI.

- Ollama availability test.
- One small generation task against a configured local model.
- One repair task against a configured local model.

Enable with an environment variable such as:

```bash
PFIT_RUN_LOCAL_LLM_TESTS=1
```

### Test Markers

Use pytest markers to keep the suite usable:

```text
unit
contract
workflow
cli
slow
local_llm
```

Default test runs should avoid `slow` and `local_llm` tests.

## Migration Phases

## Progress Checkpoint

Implemented so far:

- Added `local_agent/` package structure for local workflow orchestration.
- Added `local_agent.core.fitting` and converted `fit_parameters.py` into a compatibility wrapper.
- Added `local_agent.core.generated_contract` for static generated-script contract checks.
- Added `local_agent.agent.validators` for session, XML, dataset, generated-script, and runtime smoke validation.
- Added `local_agent.agent.session_spec` for deterministic XML-derived session summaries.
- Added deterministic `generated/user_model.py` skeleton rendering.
- Added `local_agent.agent.workflow` with validate, generate, repair, smoke-test, and event logging.
- Added local-agent prompt templates under `local_agent/agent/prompt_templates/`.
- Added local LLM clients for Ollama and fake tests.
- Added `local_agent.agent.config` for repo/session `pfit.yaml` loading.
- Added environment-variable overrides for local LLM and workflow settings.
- Added remote-workflow-aligned `pfit` CLI commands: `new`, `check`, `jax`, `run`, and `diagnose`.
- Aligned `pfit run` with the writeup by writing fitting artifacts to `outputs/<run-id>/`.
- Added simple fenced-code cleanup for local model outputs.
- Added `pyproject.toml` with a `pfit` console entry point.
- Trimmed previously added helper commands and aliases so the CLI matches the writeup workflow.
- Added `README.md` with the five-step workflow, config, logs, and test instructions.
- Added `docs/local_llm_orchestration.md` with critical architecture notes and Ollama setup guidance.
- Added optional local LLM integration test gated by `PFIT_RUN_LOCAL_LLM_TESTS=1`.
- Revamped tests around unit, contract, workflow, CLI, and slow markers.
- Restored the Differential Evolution population optimizer and XML selection via `ALGORITHM = DE`.
- Added package metadata for the top-level numerical framework modules used by installed `pfit`.
- Kept the pfit-claude-style boundary: `lib/` is the canonical fitting engine, while `local_agent/` owns the local-agent CLI, validation, LLM orchestration, diagnostics, and config.

Current default test status:

```text
72 passed, 4 deselected
```

## Running List: Still Missing

This list should be maintained as implementation continues. It tracks gaps that remain after the local Ollama-only migration and the workflow-parity trim.

### End-to-End Workflow Gaps

- Verify the local workflow against the PDF/writeup step-by-step, including artifact names, required directories, and user-visible command behavior.
- Confirm installed-package behavior from a clean virtual environment, not only in-place repo execution.

Completed end-to-end workflow items:

- Added a default-suite end-to-end CLI test for `pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose` with a fake LLM and a tiny generated script.
- Added slow fitting-engine regression coverage for DE global search handing off to NODE and writing expected artifacts.

### pfit-new Gaps

- Keep exactly one `pfit new <session>` entry point. If the user starts with a rough draft or detailed fileset inside the session directory, the default workflow should consume and preserve it instead of requiring a separate intake flag.
- Add targeted follow-up handling for missing or ambiguous symbols, columns, bounds, forcing inputs, and loss choices.

Completed pfit-new items:

- `pfit new` now writes `inputs/user_input.xml` and `generated/user_model.py` together so parameter/state ordering starts aligned.
- `pfit new` now asks the local LLM for a structured study spec and renders XML/Python deterministically from that spec, so the model no longer writes full boilerplate files.
- `pfit new` now preserves existing rough drafts and detailed filesets in the session directory, filling only missing standard files/directories.
- `pfit new` now calls the configured local LLM and rejects drafts that report or fail validation for missing equations, loss, parameter ranges, dataset, or initial conditions.
- `pfit new --overwrite` now clears prior generated XML/model/check/log artifacts before collecting context, and context collection excludes `generated/`, `outputs/`, and prior `inputs/user_input.xml`.

### LLM Orchestration Gaps

- Tune `pfit jax` prompts against the target Ollama models that will actually be recommended for users.
- Add optional Ollama integration tests for a real generation task and a real repair task, gated behind `PFIT_RUN_LOCAL_LLM_TESTS=1`.
- Add clearer model setup guidance: recommended Ollama models, minimum RAM/VRAM expectations, and model-specific caveats.
- Improve repair prompts with tighter traceback/source/context windows so local models do not drift.

### Diagnostics Gaps

- Add richer model/data-specific diagnosis rules from the writeup, beyond optimizer trend summaries.
- Include generated-script validation failure messages from `generated/agent_logs/workflow_events.jsonl` in diagnosis reports.

Completed diagnostics items:

- `pfit diagnose` now recognizes both `pso_fitting.log` and `de_fitting.log`.
- `pfit diagnose` summarizes optimizer iteration count, first loss, best loss, and final loss for global search and NODE.
- `pfit diagnose` includes workflow event counts from `generated/agent_logs/workflow_events.jsonl`.

### Fitting Engine Gaps

- Keep future fitting-engine functionality in `lib/` unless there is a clear workflow/orchestration reason for it to live under `local_agent/`.

Completed fitting-engine items:

- Added `POPULATION_STEPSIZE_RTOL` and `POPULATION_STEPSIZE_ATOL` aliases while preserving legacy `PSO_STEPSIZE_RTOL` and `PSO_STEPSIZE_ATOL`.
- Removed the recognized-but-unimplemented `gradient-only` `pfit run` mode to keep the CLI aligned with the PDF/writeup workflow.
- Added focused DE unit tests that run in the required JAX environment.
- Replaced the SciPy-backed DE path with a minimal NumPy implementation matching the writeup algorithm: LHS initialization, best-based mutation, binomial crossover, and greedy selection.

### Validation And Safety Gaps

- Add stronger XML validation for unknown entries with clearer field-specific errors.
- Add validation for non-numeric dataset cells and more precise dataset/model column mismatches.
- Add stricter generated-code safety checks around file/network/subprocess access if generated scripts are expected to be untrusted.
- Add tests for config precedence at the CLI layer: repo `pfit.yaml`, session `pfit.yaml`, environment variables, and flags.

### Documentation Gaps

- Add a walkthrough for one complete local session using Ollama.
- Document how to select PSO vs DE with `ALGORITHM = PSO|DE`.
- Document expected generated artifacts after each step.
- Add troubleshooting notes for common Ollama failures, invalid generated code, JAX/diffrax import errors, and slow/noisy fits.

### Phase 1: Stabilize Core

- Move core code into clearer modules.
- Keep `fit_parameters.py` working as a compatibility wrapper.
- Add explicit generated-script contract validation.
- Add stronger session validation.
- Start the test suite revamp with core fitting, XML, and generated-script contract tests.

### Phase 2: Extract Prompts

- Keep the prompt templates in `local_agent/agent/prompt_templates/`.
- Add prompt rendering helpers.
- Add prompt rendering tests.

### Phase 3: Add Local LLM Adapters

- Implement base `LLMClient`.
- Implement `OllamaClient`.
- Implement `FakeLLMClient` for tests.

### Phase 4: Build Orchestrator

- Implement state machine:
  - validate
  - generate
  - static check
  - runtime smoke test
  - repair
  - fit
- Store workflow logs in:

```text
sessions/<session>/generated/agent_logs/
```

### Phase 5: Add CLI

- Add `pfit new`.
- Add `pfit check`.
- Add `pfit jax`.
- Add `pfit run`.
- Add `pfit diagnose`.
- Add config loading from repo and session config files.

### Phase 6: Harden Local-Model Behavior

- Tune prompts against target local models.
- Improve repair prompts.
- Improve runtime smoke tests.
- Add structured failure reports.
- Add optional integration tests for Ollama.

### Phase 7: Documentation And Examples

- Document Ollama setup.
- Add a walkthrough for one existing session.
- Add troubleshooting notes.
- Add "bring your own local model" instructions.

## Design Principle

Treat the local LLM as an unreliable code generator inside a deterministic harness, not as the workflow owner.

The orchestrator should own:

- State
- Validation
- Retries
- File writes
- Fitting execution
- Logs
- User-facing summaries

The LLM should only perform narrow transformations:

- XML plus pseudocode to user model skeleton.
- User model to JAX generated script.
- Traceback plus generated script to repaired generated script.

This keeps the system robust and makes it possible to swap local models without rewriting the parameter-fitting workflow.
