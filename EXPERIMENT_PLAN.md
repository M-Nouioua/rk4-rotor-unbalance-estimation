# RK4 Digital Twin — Experimental Plan (Imbalance Only)

Quantitative unbalance estimation: magnitude (g·mm), keyphasor-referenced phase (deg),
and balancing-plane localization, each with propagated uncertainty. Misalignment is out
of scope (AC-coupled proximity front end → no DC gap/centerline; only the AC-observable
1X/2X unbalance response is recoverable).

> This plan was independently checked for trial-mass physics, run-count arithmetic,
> and statistical adequacy. Changes made in response to that check are marked
> **[FIX]**. Two values MUST be measured before running — see §0.

---

## 0. Measure these FIRST (everything scales from them)

| Quantity | Why | How |
|---|---|---|
| **Balance-hole radius r_h** (PCD/2) | every g·mm value is linear in r_h | caliper; rescale all masses below |
| **First critical speed N_c** | sets the two dwell speeds | run-up (Day 1) |
| **Bearing clearance** | sets the safe-orbit trip limit | measure on your unit (~5–10 mil typical) |
| **Disk mass m_disk** | trial-mass sizing | scale (assumed 0.80 kg) |
| **Max continuous speed** | you stated 0–15,000 rpm; RK4 spec is often 10,000 rpm (15,000 may be the ramp *rate*) | confirm on controller |

Assumed nominal values used below (confirm & rescale): **r_h = 30 mm** (so 1 g = 30 g·mm),
**m_disk = 0.80 kg** (W_plane = 7.85 N), **N_c ≈ 3000 rpm**, ζ ≈ 0.03–0.05.

---

## 1. Trial masses

Sizing rule — trial centrifugal force = 5–15 % of per-plane static weight:

```
m_t = f · m_disk · g / (r_h · ω²),   ω = 2π·N/60
```

Worked (N₁ = 0.5·N_c ≈ 1500 rpm, f = 10 %): ω² = 24,674 → m_t = 0.10·0.80·9.81/(0.030·24,674)
= **1.06 g ≈ 1.0 g → 30 g·mm** (force 0.74 N = 9.4 % W_plane). ✔ verified.

**A single trial mass is NOT valid across speeds — resize per speed:**

| Balancing speed | ω² (s⁻²) | Recommended trial mass |
|---|---|---|
| N₁ = 0.5 N_c ≈ 1500 rpm | 24,674 | **1.0 g** (30 g·mm) ← primary |
| N₂ = 0.8 N_c ≈ 2400 rpm | 63,167 | **0.4 g** (12 g·mm) |

Two independent mass populations (calibration never trains on the scored test set):

| | Purpose | Range |
|---|---|---|
| Calibration trial | build the ICM influence matrix A | 1.0 g @ N₁, 0.4 g @ N₂ |
| Test / unknown | score the estimator (regressions) | 0.05 – 1.5 g |

**Practice:** weigh every grub screw on a 1 mg balance; log mass + hole #; fixed calibrated
torque; **never remove the disks** between runs (largest error source).

---

## 2. Test matrix

**Severity ladder (8 levels incl. baseline), denser at the low end:**

Uses the **actual RK4 calibration weight set** (0.1 / 0.2 / 0.4 / 0.8 / 1.0 / 1.2 / 1.6 / 2.0 g):

| Level | m (g) | U (g·mm @30mm) | Purpose |
|--:|--:|--:|---|
| 0 | 0.00 | 0 | baseline / noise floor |
| 1 | 0.10 | 3 | detection limit (smallest in kit) |
| 2 | 0.20 | 6 | low |
| 3 | 0.40 | 12 | low-mid |
| 4 | 0.80 | 24 | mid |
| 5 | 1.00 | 30 | mid-high (= calibration mass) |
| 6 | 1.20 | 36 | high |
| 7 | 1.60 | 48 | high |
| 8 | 2.00 | 60 | max *(safety-gate at N₂ — see §5)* |

**Detection limit:** 0.1 g (3 g·mm) is the smallest weight in the kit; against the measured
~0.07 mils in-band noise floor that is easily resolved. To probe *below* 0.1 g (a finer LoD
ladder) you'd need custom masses (calibrated putty / drilled screws) — optional.

**Angles:** 0 / 90 / 180 / 270° primary; fine sub-sweep +45 / 135 / 225 / 315°
(**16 tapped holes @ 22.5°**, of which 8 are engraved — confirmed from
`rig_balance_disk.jpg`; angles off the 22.5° grid need two-hole vector synthesis).

**Plane configurations (4):** D1-only · D2-only · in-phase/static (m@θ both) ·
anti-phase/couple (m@θ and m@θ+180°).

**Distinct labeled conditions — CORE = 57:**

| Block | Configs | Mag | Angle | Distinct |
|---|--:|--:|--:|--:|
| A Severity backbone (θ=0°) | 4 | 7 | 1 | 28 |
| B Phase characterization | 4 | 2→**4** | 4 | 24 new |
| C Fine phase sweep (D1) | 1 | 1 | +4 | 4 |
| Baseline | — | 0 | — | 1 |
| **CORE** | | | | **57** |

**[FIX] Extend Block B masses** to include **0.2 g and 0.3 g** (not only 0.5 & 1.0 g), so
phase error is characterized where SNR is low — otherwise phase claims must be scoped to
U ≥ 15 g·mm.

**Blind / hold-out — [FIX] raise 12 → ~24–30**, colleague-prepared, off-grid, frozen
pipeline: single-plane off-grid, two-plane unequal, near-detection, and **≥8–10
balanced/residual "is it balanced?" cases** for a meaningful specificity CI (n=2 is useless).

---

## 3. Replication — "how many trials"

| Layer | Count | Captures |
|---|--:|---|
| Consecutive records per mount·speed (`n_acq`) | **3** | measurement/estimator noise (nearly free — nothing touched) |
| Independent screw re-mounts per condition (`n_mount`) | **3** | setup variance |
| → repeats per condition·speed | **9** | mean ± std everywhere |
| Anchor conditions at `n_mount` = **10** | 4 conds | pin σ_setup / σ_meas |
| Repeatability floor (same screw/hole, re-torqued) | ≥5× | reported uncertainty floor |

**[FIX]** Place anchors at a **low (~6 g·mm) and an off-axis (90°)** point too — not only at
30 g·mm / 0° — because variance scales with magnitude (heteroscedastic); one pooled σ can't
be extrapolated to the low-mass regime that matters most.

---

## 4. Speeds

- **Run-up first** → fix N_c, ζ/Q, mode shape before any dwell.
- **Core dwells:** N₁ = 0.5 N_c (primary) and N₂ = 0.8 N_c, both sub-critical, ±15 %
  exclusion band around N_c.
- **[FIX] For any "speed-dependent influence coefficient / α(ω)" claim, use S = 3**
  sub-critical speeds (e.g. 0.4 / 0.6 / 0.8 N_c) — two points can't establish a trend.
- **[FIX] Promote a reduced near-critical dwell set to CORE** (a few conditions at ~0.9 and
  ~1.1 N_c, low mass) if you want the flagship "cond(A) worst at resonance" result on
  dwell data; otherwise downgrade that claim to "consistent with run-up evidence."
- **ICM calibration** at each dwell speed = baseline + 1 trial run per plane (**3 runs/speed**).

---

## 5. Safety gates [FIX — from physics verification]

1. **1.5 g test mass at N₂ = 0.8 N_c is NOT automatically safe.** At r = 0.8, A ≈ 1.7×;
   with single-disk modal mass the orbit can reach ~7.7 mil pk-pk (> 5 mil half-clearance).
   **Gate:** do not run level 7 (1.5 g) at N₂ until the Day-1 orbit measurement confirms
   margin; if effective mass ≈ one disk, **cap N₂ test mass at ~0.8–1.0 g.**
2. **Through-resonance run-ups: use residual mass ≤ ~0.1 g**, not 0.3 g — at Q ≈ 10 even
   0.3 g can reach ~8–9 mil and trip. Keep the live-amplitude trip armed at ~half clearance.
3. Rub screw set as a deliberate mechanical limit stop; keep steady pk-pk orbit < ~half
   bearing clearance.

---

## 6. Run count & bench time

**Master formula (acquisition records):**
```
N = C_sev·S·(n_mount·n_acq) + C_blind·S·(n_mount·n_acq) + anchor_upgrade + ICM_cal + N_sweeps
```
with C_sev = 57, C_blind = 12, S = 2, n_mount = 3, n_acq = 3 (anchors n_mount = 10).

| Block | Cond | ×Speed | ×Rep | Runs |
|---|--:|--:|--:|--:|
| A Severity | 28 | 2 | 9 | 504 |
| B Phase | 24 | 2 | 9 | 432 |
| C Fine phase | 4 | 2 | 9 | 72 |
| Baseline | 1 | 2 | 9 | 18 |
| *core subtotal* | | | | *1026* |
| Blind | 12 | 2 | 9 | 216 |
| Anchor upgrade | 4 | 2 | 21 | 168 |
| ICM calibration | 3 | 2 | 9 | 54 |
| Run-up/coast sweeps | 14 | — | — | 14 |
| **GRAND TOTAL** | | | | **1478 acquisitions** |

Physical mass changes (the real cost): **~255 mounts.**

**[FIX] Bench time is mount-driven and 14 min/mount is optimistic** (each mount runs at BOTH
speeds → 2 spin-ups + ~6 min recording + the mass change ≈ 17–20 min):
**≈ 85–100 h → ~12–13 bench-days** (not 72 h / 9–10 days). Raising the blind set and adding
the low-mass ladder / S=3 pushes this up further — budget ~3 weeks incl. analysis.

---

## 7. Minimum-viable subset (first submission)

| | MVP | Full |
|---|---|---|
| Speeds | 1 (0.5 N_c) | 2–3 |
| Angles | 2 (0°, 90°) | 4 (+fine 8) |
| C_sev | ~41 | 57 |
| Blind | ~10 (incl. balanced) | 24–30 |
| Records | ~540 | 1478+ |
| Time | ~5 days | ~12–13 days |
| Supports | magnitude R²/RMSE + localization + basic UQ | + full phase coverage, speed-dependence, tight blind stats |

---

## 8. Schedule (full plan)

- **Day 0** Weigh screws (1 mg); measure r_h, clearance, disk mass; map hole #1 → keyphasor
  (±1–2°); set probe gaps to −10 V DC and lock; set rub screw; warm-up checks.
- **Day 1** Baseline run-ups → N_c, ζ/Q, mode shape → first ROSS update; fix N₁/N₂;
  empirically confirm orbit limits & the 1.5 g / N₂ gate.
- **Day 2** ICM 3-run set at each speed; 2× linearity check; 4 anchors at n_mount = 10;
  ≥5-repeat repeatability floor.
- **Days 3–7** Labeled grid (Blocks A/B/C) — **[FIX] interleave A/B/C and randomize speed
  order** (don't run them as sequential campaigns; else angle/config aliases with day/thermal
  drift). RCBD half-day blocks; baseline + check-standard each block (control chart).
- **Day 8** Blind hold-out (analyst blind); calibrated-known-mass run-ups (predict-not-fitted
  twin validation).
- **Day 9** Near-critical / generalization block; re-run control-chart flags.
- **Day 10** QC, final ROSS update, regressions (magnitude slope/R²/RMSE + Bland–Altman;
  circular phase-error stratified by magnitude; localization accuracy with Wilson CI;
  GUM Type A+B budget → U₉₅), archive with seeds/labels.

---

## 9. Reporting / novelty

1. **Uncertainty-quantified, physics-cross-validated unbalance metrology** — traceable g·mm
   labels, error budget, magnitude/phase regressions, localization confusion matrix,
   cond(A) as the localization figure of merit (**[FIX] bootstrap cond(A) over calibration
   replicates → report with a CI**).
2. **Update-then-predict digital twin** — run-up updates ROSS support stiffness/damping; the
   updated model predicts the unbalance FRF / influence coefficients it was not fit to.
3. Physics thread: A ill-conditioned near critical → localization worst at resonance (ROSS
   predicts, experiment confirms) — needs the near-critical dwell set (§4).

**[FIX]** Tighten the angular datum (optical/index fixture) or report phase error against the
±1–2° label floor; drop any 4.0 g extrapolation claim (test masses reach only 1.5 g).

---

## 10. Follow-up experiments (added 2026-07-26, after the blind validation)

The blind validation (BLND01–10 unsealed) exposed three concrete gaps. All three are
now wired into the acquisition app (`python -m scripts.add_experiments` generated the
conditions/speeds; run `python -m gui.campaign`). Acquire, then re-run
`python -m scripts.pipeline` — the analysis picks them up automatically.

| Block / speed | What | Why (the finding it fixes) | Cost |
|---|---|---|---|
| **`T_twoplane`** (12) | general two-plane, per-disk **24–60 g·mm** at **arbitrary relative angles** (45/90/135°), not just in/anti-phase | the trees underestimated the blind two-plane loads (loaded-MAE ~17 vs physics 6.3) because training two-plane was thin (15 % ≥36 g·mm) and only Δ0/180° | N1,N2 · ~1 mount ea |
| **`U_anchor`** (4) | low/mid/high + off-axis, **`n_mount=6` independent re-mounts** each | **0 of 85 conditions currently have a repeated mount** → σ_setup is unmeasured; needed for the GUM U₉₅ budget | N1,N2 · 6 mounts ea |
| **`N3` = 660 rpm** | 0.40 N_c third sub-critical dwell for a 12-condition subset (baseline + trials + D1 ladder + 2 two-plane + 1 anchor) | multi-speed ICM / α(ω) trend; a wider speed lever arm than N1(0.61)+N2(0.73) | subset only |

**App behaviour added:** `run_at_speeds` is now enforced (a condition shows *n/a* at a
speed it isn't scheduled for, so N3 only lists its subset); each **Run captures one
physical mount** (≤ `n_acq`), and multi-mount anchors prompt you to re-mount between
repeats so the mount labels — and therefore σ_mount — are real.

**Held-out design:** keep **BLND01–05** unseen when you re-train; the money figure is the
before/after drop in blind two-plane MAE once `T_twoplane` is in the training set.

**Safety:** N3 (660 rpm) is far below the 1400–1900 resonance band — safe with the
standard trial/test masses. A *near-critical* dwell (≈1350–1380 rpm) would sharpen the
cond(A) story but must be run manually with ≤0.4 g and the orbit trip armed; it is
deliberately **not** in `speeds.csv`.
