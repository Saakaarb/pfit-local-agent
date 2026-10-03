# Model comparison and follow-up evidence, 2026-09-27

The completed clean comparison is in `model_comparison_20260927`.
`model_comparison_20260927_preliminary` is preserved for provenance and excluded
from the reported results because its session context included benchmark metadata.
Reports and comparative tables are in `docs/model_comparison_20260927*`.

These records include frozen prompts/data, per-model inputs, generated code,
LLM request/response logs, timings, model metadata, fit outputs and failure logs.
Model weights and Python bytecode are excluded.

Follow-ups are separate from the original comparison: failure analysis, RMSE
contract checks, solver recovery, and initial-condition extraction trials.
Sneyd initialization passed in `initial_conditions_live_20260927_v4` with all
eight defaults and nine experiment clamp pairs verified. Earlier trials failed.
The optional tanks restart in v3 was interrupted before completion for device
shutdown; its first evaluated gradient was finite, so it is not evidence of a
live NaN-warning reproduction. The NaN warning is covered by a regression test.

`manifest_20260927.json` records SHA-256 hashes and sizes of the saved artifacts.
