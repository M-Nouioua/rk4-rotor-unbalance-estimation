# Generated results — do not edit by hand

Regenerate with `python -m scripts.benchmark`. Source of truth: `analysis/benchmark.json`.

- Labeled acquisitions: **510** from **85** physical conditions (the independent unit).
- Influence-matrix conditioning: cond(A)@N1=16.99, cond(A)@N2=24.81
- Detection threshold used for fixed-threshold metrics: 6.0 g·mm; phase reported for U ≥ 24.0 g·mm.

## 1. Condition-level performance (cite these)

| Method | R²pred | R²line | Slope | RMSE g·mm | MAE | Phase U≥24 | Cross-talk | Localization | ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| physics | 0.623 | 0.716 | 0.89 | 8.81 | 6.46 | 22.9° | 11.82 | 51.2% [41–62] | 0.812 |
| ml | 0.531 | 0.774 | 0.71 | 9.84 | 7.66 | 5.7° | 8.31 | 47.6% [37–58] | 0.688 |
| hybrid | 0.659 | 0.808 | 0.79 | 8.38 | 6.25 | 6.1° | 5.65 | 56.1% [45–66] | 0.792 |

Always-loaded baseline accuracy: **70.7%** — compare every fixed-threshold accuracy against it.

## 2. Paired condition-level differences (cluster bootstrap, 95% CI)

Negative = the first method is better. An interval containing 0 means this campaign cannot separate them.

| Comparison | Δ magnitude abs-err | 95% CI | Δ phase (U≥24) | 95% CI |
|---|--:|---|--:|---|
| hybrid − physics | -0.21 | [-1.86, +1.36] | -16.8° | [-22.3, -11.5] * |
| ml − physics | +1.20 | [-0.63, +3.04] | -17.2° | [-23.0, -11.8] * |
| hybrid − ml | -1.41 | [-1.96, -0.87] * | +0.4° | [-0.3, +1.3] |

`*` = interval excludes zero.

## 3. Factor-aware validation matrix

condition-level scores. Protocols are distinct deployment questions, not an ordered difficulty ladder.

| Protocol | Hybrid R²pred | Hybrid phase | ML R²pred | ML phase |
|---|--:|--:|--:|--:|
| random_acquisition | 0.978 | 0.3° | 0.961 | 0.3° |
| condition | 0.659 | 6.1° | 0.531 | 5.7° |
| magnitude_interpolating | 0.643 | 6.0° | 0.550 | 5.7° |
| magnitude_extrapolating | -4.460 | 15.0° | -4.820 | 15.8° |
| configuration | -0.393 | 10.1° | -0.625 | 19.9° |
| angle_sector | -1.120 | 67.4° | -1.222 | 93.0° |
| physics / random_acquisition | 0.624 | 22.9° | — | — |
| physics / condition | 0.624 | 22.9° | — | — |
| physics / magnitude_interpolating | 0.502 | 23.6° | — | — |
| physics / magnitude_extrapolating | 0.408 | 24.0° | — | — |
| physics / configuration | 0.624 | 22.9° | — | — |
| physics / angle_sector | 0.624 | 22.9° | — | — |

Protocol definitions:

- **random_acquisition** — LEAKY NEGATIVE CONTROL -- repeats of one condition straddle folds. Included only to quantify what the field's default split buys.
- **condition** — Interpolation to unseen mass/angle COMBINATIONS on the sampled grid.
- **magnitude_interpolating** — Leave one interior total-unbalance level out at a time. Endpoint levels are excluded, so every scored level is bracketed by lower and higher levels in training.
- **magnitude_extrapolating** — Train on total U<36 g.mm and test on total U>=36 g.mm. This is a high-total-unbalance holdout; loading composition is not matched.
- **configuration** — Leave-one-loaded-configuration-out (D1 / D2 / in-phase / anti-phase); the baseline-only calibration group is not scored.
- **angle_sector** — Leave-one-angle-sector-out. Only 4 distinct angles exist at U>=24 g.mm, so on-grid phase scores cannot be read as generalization.

## 4. Blind validation (labels revealed after prediction)

5 loaded + 5 balanced conditions, 20 disk-points. condition-disk (complex-averaged repeats).

| Method | MAE | Loaded MAE | Detect acc [95% CI] | Spec | Sens | ROC-AUC | Phase U≥24 |
|---|--:|--:|---|--:|--:|--:|--:|
| physics | 6.3 | 6.5 | 65% [40–90] | 45% | 89% | 0.849 | 7° |
| ml | 9.8 | 17.5 | 85% [70–100] | 91% | 78% | 0.919 | 20° |
| hybrid | 9.2 | 17.1 | 85% [70–100] | 91% | 78% | 0.879 | 16° |

Detection intervals resample the 10 whole conditions, preserving the two disk outcomes within each condition.

## 5. Exploratory ablation — direct |U| target

**EXPLORATORY -- post-hoc target change, selection-biased, needs nested validation or a new blind set before any claim**

| Protocol | R²pred (magnitude) | Slope |
|---|--:|--:|
| condition | 0.736 | 0.75 |
| angle_sector | -0.049 | 0.25 |

## Legacy per-acquisition block

`methods.<m>.overall` in the JSON is the old per-acquisition scoring, kept so the existing figures/dashboard keep running. Its n counts records, not independent conditions, so it must not be cited.
