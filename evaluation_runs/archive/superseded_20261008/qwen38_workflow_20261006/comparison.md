# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen32b (status; seconds) | qwen14b (status; seconds) | qwen38_27b (status; seconds) |
|---|---|---|---|
| ARC_fitting | pending | pass; 201.702 | pass; 243.997 |
| boehm_stat5 | pending | pass; 203.138 | pass; 268.163 |
| hodgkin_huxley | pending | blocked; 0 | blocked; 0 |
| lotka_volterra | pending | pass; 99.687 | pass; 150.076 |
| mapk_cascade | pending | pass; 198.1 | pass; 256.116 |
| nfkb_signaling | pending | fail at new; 123.398 | fail at new; 126.482 |
| oregonator | pending | pass; 156.068 | pass; 206.122 |
| piezo_bouc_wen | pending | fail at new; 43.492 | pass; 213.513 |
| robertson_session | pending | pending | pass; 200.062 |
| session1 | pending | pending | blocked; 0 |
| sliding_basepoint_headered | pending | pending | fail at jax; 197.57 |
| test_session | pending | pending | pass; 183.656 |
| theophylline | pending | pass; 118.242 | pass; 175.556 |
| vanderpol_session | pending | pending | pass; 195.781 |
| cascaded_tanks | pending | fail at new; 55.595 | degraded; 180.704 |
| decay_multiexp | pending | pass; 106.49 | pass; 179.17 |
| sneyd_ipr | pending | pending | pass; 365.182 |
| beer_indigoidine | pending | pass; 125.6 | pass; 205.118 |
| raia_il13 | pending | fail at new; 89.578 | pass; 290.515 |
| schwen_insulin | pending | pending | pass; 349.857 |
| armistead_sphingolipid | pending | fail at new; 49.65 | pass; 259.009 |
| borghans_calcium | pending | fail at new; 48.987 | pass; 230.798 |
| fujita_egf | pending | fail at new; 142.228 | pass; 296.587 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen32b | 0 | 0 | 0 | 23 |
| qwen14b | 8 | 7 | 1 | 7 |
| qwen38_27b | 18 | 3 | 2 | 0 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.
