"""
Generate the pre-filled experiment matrix for the RK4 imbalance campaign:
  datasets/conditions.csv  - every planned labeled condition (ground truth)
  datasets/speeds.csv      - dwell-speed template (fill rpm after N_c is measured)
  datasets/index.csv       - empty acquisition log (acquisition appends one row/run)

Reproduces EXPERIMENT_PLAN.md. Run from the repo root:
    python -m scripts.gen_matrix                              # CORE (57 conditions)
    python -m scripts.gen_matrix --phase-fix --lod --blind 24 # recommended (verified) plan

All unbalance = mass (g) x radius (mm) = U (g.mm). Everything scales with --radius,
so re-run with your MEASURED balance-hole radius before the campaign.
"""
from __future__ import annotations

import argparse
import csv
import os

CONFIGS = ["D1", "D2", "inphase", "antiphase"]
# Actual RK4 calibration weight-set masses (g) — from the labelled kit.
SEV_MASSES = [0.10, 0.20, 0.40, 0.80, 1.00, 1.20, 1.60, 2.00]
CARDINAL = [0, 90, 180, 270]
FINE = [45, 135, 225, 315]
PHASE_MASSES_CORE = [0.40, 1.00]                           # Block B (core)
PHASE_MASSES_FIX = [0.20, 0.40, 0.80, 1.00]                # Block B ([FIX] add low SNR)
LOD_MASSES = [0.10, 0.20]      # smallest in the kit; sub-0.1 g needs custom putty/weights

# Anchor conditions (n_mount = 10) — pin setup vs measurement variance.
# [FIX] include a low (0.2 g) and an off-axis (90 deg) point, not only 30 g.mm/0 deg.
ANCHORS = {("D1", 1.00, 0), ("D2", 1.00, 0), ("D1", 0.20, 0), ("D1", 0.40, 90)}

FIELDS = ["condition_id", "block", "config", "disk1_mass_g", "disk1_angle_deg",
          "disk2_mass_g", "disk2_angle_deg", "radius_mm", "U1_gmm", "U2_gmm",
          "n_mount", "is_anchor", "run_at_speeds", "notes"]


def layout(config: str, m: float, angle: int):
    """(disk1_mass, disk1_angle, disk2_mass, disk2_angle) for a config."""
    a2 = (angle + 180) % 360
    return {
        "baseline":  (0.0, 0, 0.0, 0),          # balanced reference
        "D1":        (m, angle, 0.0, 0),
        "D2":        (0.0, 0, m, angle),
        "inphase":   (m, angle, m, angle),      # static unbalance
        "antiphase": (m, angle, m, a2),         # couple unbalance
    }[config]


def row(cid, block, config, m, angle, r, n_mount=3, note=""):
    d1m, d1a, d2m, d2a = layout(config, m, angle)
    anchor = (config, m, angle) in ANCHORS
    return {
        "condition_id": cid, "block": block, "config": config,
        "disk1_mass_g": round(d1m, 3), "disk1_angle_deg": d1a,
        "disk2_mass_g": round(d2m, 3), "disk2_angle_deg": d2a,
        "radius_mm": r, "U1_gmm": round(d1m * r, 2), "U2_gmm": round(d2m * r, 2),
        "n_mount": 10 if anchor else n_mount, "is_anchor": int(anchor),
        "run_at_speeds": "all", "notes": note,
    }


def build(r, phase_masses, add_lod, n_blind):
    rows = []
    # Baseline (also serves as ICM baseline)
    rows.append(row("BASE", "baseline", "baseline", 0.0, 0, r, note="balanced reference"))

    # Block A - severity backbone at theta = 0
    i = 1
    for cfg in CONFIGS:
        for m in SEV_MASSES:
            rows.append(row(f"A{i:03d}", "A_severity", cfg, m, 0, r)); i += 1

    # Block B - phase characterization (skip theta=0, already in A)
    i = 1
    for cfg in CONFIGS:
        for m in phase_masses:
            for ang in CARDINAL:
                if ang == 0:
                    continue
                rows.append(row(f"B{i:03d}", "B_phase", cfg, m, ang, r)); i += 1

    # Block C - fine phase sweep (D1, 0.5 g)
    for j, ang in enumerate(FINE, 1):
        rows.append(row(f"C{j:03d}", "C_fine_phase", "D1", 0.40, ang, r))

    # Optional low-mass Limit-of-Detection ladder (D1, extra replication).
    # Skip masses already in Block A's D1@0 set (0.10/0.20 g) to avoid duplicates.
    if add_lod:
        j = 1
        for m in LOD_MASSES:
            if m in SEV_MASSES:
                continue
            rows.append(row(f"LOD{j:02d}", "LoD", "D1", m, 0, r, n_mount=5,
                            note="low-mass detection-limit ladder (extra replication)"))
            j += 1

    # ICM calibration trials (mass is speed-specific -> see speeds.csv)
    for cfg, cid in (("D1", "ICM_D1"), ("D2", "ICM_D2")):
        rr = row(cid, "ICM", cfg, 0.0, 0, r, note="trial mass per speeds.csv (resize per speed)")
        rr["disk1_mass_g"] = rr["disk2_mass_g"] = ""      # filled per speed at run time
        rr["U1_gmm"] = rr["U2_gmm"] = ""
        rows.append(rr)

    # Blind / hold-out placeholders (colleague fills, kept sealed from the analyst)
    for k in range(1, n_blind + 1):
        need_balanced = k > n_blind - max(8, n_blind // 3)   # >=8 balanced for specificity
        rows.append({
            "condition_id": f"BLND{k:02d}", "block": "blind", "config": "sealed",
            "disk1_mass_g": "", "disk1_angle_deg": "", "disk2_mass_g": "",
            "disk2_angle_deg": "", "radius_mm": r, "U1_gmm": "", "U2_gmm": "",
            "n_mount": 3, "is_anchor": 0, "run_at_speeds": "all",
            "notes": "SEALED - colleague fills off-grid; " +
                     ("balanced/residual (specificity)" if need_balanced else "unknown fault"),
        })
    return rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=30.0, help="balance-hole radius (mm) - MEASURE THIS")
    ap.add_argument("--phase-fix", action="store_true", help="add 0.2/0.3 g to phase block")
    ap.add_argument("--lod", action="store_true", help="add low-mass detection-limit ladder")
    ap.add_argument("--blind", type=int, default=12, help="number of blind hold-out conditions")
    ap.add_argument("--outdir", default="datasets")
    a = ap.parse_args()

    phase_masses = PHASE_MASSES_FIX if a.phase_fix else PHASE_MASSES_CORE
    rows = build(a.radius, phase_masses, a.lod, a.blind)

    write_csv(os.path.join(a.outdir, "conditions.csv"), FIELDS, rows)

    # speeds template (fill rpm after Day-1 run-up fixes N_c)
    write_csv(os.path.join(a.outdir, "speeds.csv"),
              ["speed_id", "rpm", "fraction_Nc", "trial_mass_g", "notes"],
              [{"speed_id": "N1", "rpm": "", "fraction_Nc": 0.5, "trial_mass_g": 1.0,
                "notes": "primary; resize trial mass after N_c measured"},
               {"speed_id": "N2", "rpm": "", "fraction_Nc": 0.8, "trial_mass_g": 0.4,
                "notes": "gate 1.5 g test mass here on Day-1 orbit check"}])

    # empty acquisition log
    write_csv(os.path.join(a.outdir, "index.csv"),
              ["run_id", "timestamp", "condition_id", "speed_id", "rpm_commanded",
               "rpm_measured", "mount_idx", "acq_idx", "file", "T_ambient_C", "notes"], [])

    # summary
    from collections import Counter
    c = Counter(r["block"] for r in rows)
    labeled = sum(1 for r in rows if r["block"] not in ("blind", "ICM"))
    anchors = sum(r["is_anchor"] for r in rows)
    print(f"conditions.csv written: {len(rows)} rows")
    for b in ["baseline", "A_severity", "B_phase", "C_fine_phase", "LoD", "ICM", "blind"]:
        if c.get(b):
            print(f"  {b:14s}: {c[b]}")
    print(f"  anchors (n_mount=10 subset): {anchors}")
    print(f"labeled (non-blind, non-ICM) conditions: {labeled}")
    print("speeds.csv, index.csv (empty log) written.")


if __name__ == "__main__":
    main()
