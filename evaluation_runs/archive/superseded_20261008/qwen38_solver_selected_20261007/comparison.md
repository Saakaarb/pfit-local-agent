# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen38_27b (status; seconds) |
|---|---|
| ARC_fitting | pass; 344.795 |
| boehm_stat5 | fail at jax; 393.571 |
| hodgkin_huxley | blocked; 0 |
| lotka_volterra | pass; 230.452 |
| mapk_cascade | pass; 341.294 |
| nfkb_signaling | fail at new; 160.039 |
| oregonator | pass; 264.461 |
| piezo_bouc_wen | pass; 262.854 |
| robertson_session | pass; 347.634 |
| session1 | blocked; 0 |
| sliding_basepoint_headered | fail at jax; 416.733 |
| test_session | pending |
| theophylline | pass; 239.868 |
| vanderpol_session | pending |
| cascaded_tanks | degraded; 269.57 |
| decay_multiexp | pass; 244.6 |
| sneyd_ipr | pending |
| beer_indigoidine | pass; 265.319 |
| raia_il13 | fail at jax; 412.553 |
| schwen_insulin | pass; 548.467 |
| armistead_sphingolipid | pass; 326.191 |
| borghans_calcium | pass; 325.36 |
| fujita_egf | fail at jax; 528.862 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen38_27b | 12 | 6 | 2 | 3 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.
