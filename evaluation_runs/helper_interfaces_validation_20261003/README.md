# Helper interface validation, 2026-10-03

All six isolated Qwen2.5-Coder 32B JAX reruns passed. Each uses the failed
comparison case's extracted YAML and user_model.py, with fresh helper/RHS
translation. No initial extraction or fitting was rerun. Logs and source stamps
record deterministic and numerical acceptance. Every case used two LLM calls;
no repairs were needed. The regression tests separately exercise bounded
helper-only repair with deterministic loss preservation.

The original comparison remains stopped and its failed attempts are preserved
in `evaluation_runs/model_comparison_20261003`; these validation runs must not
be substituted into its scores. NF-kB and Sneyd extraction are not addressed.
