# Ollama model comparison — 2026-09-27

All 48 eligible case/model attempts completed. Observed complete workflow passes: **32B 12/16, 14B 10/16, Coder-Next 11/16**. Two incomplete folders are excluded for every model. These are bounded workflow tests, not converged scientific fits.

The clean run used commit `dd823e6a80f20c812c0e698e3c9c0fba84b0407f`. The production framework was unchanged across batches. The preliminary run with benchmark metadata in model context is excluded. A final audit of 437 recorded successful LLM calls found no benchmark metadata/log file blocks in their session context. Failed-call partial outputs are retained separately.

## Model metadata and timing

| Model | Parameters total / active | Quantization | Download GB | GPU memory MiB after warm-up | Pass / fail / degraded | Case time total (min) | Median time on common distinct passing tasks (s) |
|---|---|---|---:|---:|---|---:|---:|
| Qwen2.5-Coder 32B | 32.76B / 32.76B | Q4_K_M | 19.85 | 23917 MiB | 12 / 3 / 1 | 29.99 | 82.17 |
| Qwen2.5-Coder 14B | 14.77B / 14.77B | Q4_K_M | 8.99 | 12307 MiB | 10 / 6 / 0 | 21.17 | 63.09 |
| Qwen3-Coder-Next 80B/A3B | 79.67B / ~3B | UD-Q3_K_XL | 36.28 | 35771 MiB | 11 / 4 / 1 | 21.38 | 77.34 |

All models are **non-thinking**. The dense models use their total count as the nominal active count; Coder-Next is a mixture-of-experts model. Actual GGUF parameter counts and download bytes are reported above, rather than rounded model names. GPU figures are total device memory used immediately after warm-up, not per-case peak measurements. All models loaded fully on the H100 80GB GPU.

Case totals include failed attempts and exclude downloading/warm-up. Early failure can make a model look faster. The timing comparison uses the same 7 distinct tasks passed by all models: boehm_stat5, decay_multiexp, lotka_volterra, mapk_cascade, robertson_session, theophylline, vanderpol_session. It includes fitting and validation time, not just inference. Single trials provide no uncertainty estimate.

## Case outcomes

| Case | 32B | 14B | Coder-Next |
|---|---|---|---|
| ARC_fitting | pass (148.1s) | pass (92.9s) | fail at new (4.2s) |
| boehm_stat5 | pass (220.7s) | pass (125.3s) | pass (110.0s) |
| hodgkin_huxley | blocked | blocked | blocked |
| lotka_volterra | pass (66.5s) | pass (55.8s) | pass (60.9s) |
| mapk_cascade | pass (127.2s) | pass (98.5s) | pass (115.3s) |
| nfkb_signaling | fail at new (235.8s) | fail at new (93.1s) | fail at new (74.0s) |
| oregonator | fail at jax (137.9s) | fail at jax (75.9s) | fail at jax (109.2s) |
| piezo_bouc_wen | pass (109.2s) | fail at new (30.6s) | pass (99.4s) |
| robertson_session | pass (82.2s) | pass (76.0s) | pass (77.3s) |
| session1 | blocked | blocked | blocked |
| sliding_basepoint | pass (122.2s) | fail at new (39.0s) | pass (86.0s) |
| sliding_basepoint_headered | pass (109.0s) | fail at new (37.2s) | pass (79.5s) |
| test_session | pass (82.5s) | pass (69.5s) | pass (76.7s) |
| theophylline | pass (83.6s) | pass (57.7s) | pass (107.8s) |
| vanderpol_session | pass (74.2s) | pass (54.7s) | pass (63.1s) |
| cascaded_tanks | degraded (91.0s) | fail at new (41.4s) | degraded (80.8s) |
| decay_multiexp | pass (76.1s) | pass (63.1s) | pass (67.3s) |
| sneyd_ipr | fail at new (33.3s) | pass (259.3s) | fail at check (71.4s) |

The two duplicate variants are test_session/Robertson and sliding_basepoint_headered/sliding_basepoint. On the 14 distinct eligible scientific tasks, passes are Qwen2.5-Coder 32B: 10/14, Qwen2.5-Coder 14B: 9/14, Qwen3-Coder-Next 80B/A3B: 9/14.

## Findings and limitations

- 32B completed the most cases in this single run. The larger model did not improve completion overall; generation, architecture and quantization differ, so this does not isolate parameter count.
- 14B passed Sneyd. 32B incorrectly reported supplied initial conditions as missing. Coder-Next reached check but produced invalid JSON there. This shows different failure modes across extraction and review, rather than a universal inability to process nine experiments.
- Oregonator failed JAX validation for every model. NF-kB failed initial generation for every model. These shared blockers warrant investigation independently of model size.
- Cascaded tanks was degraded for 32B and Coder-Next: fitting reported invalid_loss_or_gradient rather than completing refinement. 14B failed initial generation. An accepted diagnosis alone does not count as a workflow pass.
- Coder-Next failed ARC intake by returning inputs/NMC_SOC80_M2_normalized_converted_units.csv where the intake expected the filename relative to inputs/. This was scored as an observed workflow failure, without a model-specific patch.
- The human-writable scientific prompts and CSV hashes were frozen before the clean run. No source YAML or reference implementation was supplied. Exact prompts, preparation notes and protocol are in [the benchmark directory](../benchmarks/model_comparison/README.md).
- The benchmark enforces expected experiment files and output counts, finite outputs/loss, refinement completion and accepted diagnosis, in addition to framework semantic and source/JAX checks. These checks are not an independent proof that all generated equations match the scientific specification.

## Retained artifacts

- [Per-case CSV with all five stage timings and model metadata](model_comparison_20260927.csv).
- [Model identities, digests and preparation metadata](model_comparison_20260927_models.json).
- Raw session code, outputs, request telemetry and logs: `/workspace/pfit-local-agent/evaluation_runs/model_comparison_20260927/`.
- Preliminary context-contaminated runs: `evaluation_runs/model_comparison_20260927_preliminary/`, excluded from scoring.
- The first Coder-Next preflight stopped on insufficient reported free space immediately after model deletion. An infrastructure-only retry waited for RunPod volume accounting; it did not repeat or replace any scored case. That event is retained under `infrastructure_attempts/`, with the recovery script/logs at the run root.
- The 32B default model is restored separately after scoring; its download is not charged to any case.
