"""
Build the cached feature table for the digital twin.

Processes every logged acquisition once (keyphasor order tracking is the slow
step) and writes a flat, ML-ready table to `analysis/features.csv`:

    metadata + labels + physics-informed features   (one row per .npz)

Labels are the *added* unbalance per disk (g.mm complex, from the mounted screw),
so both the physics ICM and the ML models regress against the same ground truth.
Runout is subtracted with `datasets/runout_slowroll.npz`; the per-speed baseline
(mean of the BASE acquisitions) enables baseline-subtracted "response" features.

    python -m scripts.build_features
"""
from __future__ import annotations

import csv
import numpy as np

from config import load_config, ROOT
from processing.features import (SENSORS, acquisition_vectors, runout_vectors,
                                 feature_row)

DATASETS = ROOT / "datasets"
ANALYSIS = ROOT / "analysis"
LOAD_THRESH = 6.0   # g.mm (=0.2 g @30 mm) to call a disk "loaded"


def load_csv(name):
    with open(DATASETS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def true_added(c):
    """Added unbalance per disk (complex g.mm) from the mounted screw.

    Returns (None, None) for sealed/blind conditions (no ground truth) or any row
    with empty mass fields, so they are excluded from supervised training/scoring.
    """
    if c.get("block") == "blind" or not (c.get("disk1_mass_g", "").strip()
                                         or c.get("disk2_mass_g", "").strip()):
        return None, None
    try:
        r = float(c["radius_mm"])
        u1 = float(c["disk1_mass_g"] or 0) * r * np.exp(1j * np.radians(float(c["disk1_angle_deg"] or 0)))
        u2 = float(c["disk2_mass_g"] or 0) * r * np.exp(1j * np.radians(float(c["disk2_angle_deg"] or 0)))
        return u1, u2
    except (ValueError, KeyError):
        return None, None


def mean_baseline(cfg, runout, index, speed):
    """Mean runout-compensated vectors over the BASE acquisitions at `speed`."""
    rows = [r for r in index if r["condition_id"] == "BASE" and r["speed_id"] == speed]
    if not rows:
        return None
    acc = {p: {1: [], 2: [], 3: []} for p in SENSORS}
    for r in rows:
        d = np.load(DATASETS / r["file"], allow_pickle=True)
        v = acquisition_vectors(d, cfg, runout)
        for p in SENSORS:
            for k in (1, 2, 3):
                acc[p][k].append(v[p].get(k, 0j))
    return {p: {k: np.mean(acc[p][k]) for k in (1, 2, 3)} for p in SENSORS}


def main():
    cfg = load_config()
    index = load_csv("index.csv")
    conds = {c["condition_id"]: c for c in load_csv("conditions.csv")}

    runout = runout_vectors(cfg, np.load(DATASETS / "runout_slowroll.npz", allow_pickle=True))
    baselines = {s: mean_baseline(cfg, runout, index, s) for s in ("N1", "N2")}
    print("baselines:", {s: "ok" if b else "MISSING" for s, b in baselines.items()})

    rows = []
    n = len(index)
    for i, r in enumerate(index, 1):
        cid = r["condition_id"]
        c = conds.get(cid, {})
        speed = r["speed_id"]
        path = DATASETS / r["file"]
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        base = baselines.get(speed)
        feat = feature_row(d, cfg, runout, base)

        u1, u2 = true_added(c)
        labeled = u1 is not None
        block = c.get("block", "")
        rec = {
            "file": r["file"], "condition_id": cid, "block": block,
            "config": c.get("config", ""), "speed_id": speed,
            "rpm_measured": r.get("rpm_measured", ""),
            "mount_idx": r.get("mount_idx", ""), "acq_idx": r.get("acq_idx", ""),
            "labeled": int(labeled),
            "U1_re": u1.real if labeled else "", "U1_im": u1.imag if labeled else "",
            "U2_re": u2.real if labeled else "", "U2_im": u2.imag if labeled else "",
            "U1_mag": abs(u1) if labeled else "", "U2_mag": abs(u2) if labeled else "",
            "U1_ang": np.degrees(np.angle(u1)) if labeled and abs(u1) > 1e-9 else "",
            "U2_ang": np.degrees(np.angle(u2)) if labeled and abs(u2) > 1e-9 else "",
            "disk1_loaded": int(labeled and abs(u1) >= LOAD_THRESH),
            "disk2_loaded": int(labeled and abs(u2) >= LOAD_THRESH),
        }
        rec.update(feat)
        rows.append(rec)
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n}  {cid} {speed}")

    ANALYSIS.mkdir(exist_ok=True)
    out = ANALYSIS / "features.csv"
    cols = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    nlab = sum(r["labeled"] for r in rows)
    print(f"\nwrote {out}: {len(rows)} rows ({nlab} labeled, {len(rows)-nlab} blind), "
          f"{len(cols)} columns")


if __name__ == "__main__":
    main()
