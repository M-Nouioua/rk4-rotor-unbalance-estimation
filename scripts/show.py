"""
Summarise the logged acquisitions for a condition + speed: per-run 1X/2X per
probe (amplitude in mils + keyphasor phase) and the mean +/- std across runs.

    python -m scripts.show --condition BASE --speed N1
"""
from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from config import load_config, ROOT
from processing.order_tracking import harmonic_vectors, detect_pulses, instantaneous_rpm

DATASETS = ROOT / "datasets"
PROBES = ["plane1_x", "plane1_y", "plane2_x", "plane2_y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--speed", default="N1")
    args = ap.parse_args()

    cfg = load_config()
    sens = cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"]
    to_mils = 1000.0 / (sens * 25.4)
    kc = cfg["daq"]["keyphasor"]

    with open(DATASETS / "index.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["condition_id"] == args.condition and r["speed_id"] == args.speed]
    if not rows:
        print(f"no logged runs for {args.condition} @ {args.speed}"); return

    print(f"{args.condition} @ {args.speed}  —  {len(rows)} run(s)\n")
    acc = {p: [] for p in PROBES}
    for r in rows:
        path = DATASETS / r["file"]
        if not path.exists():
            print(f"  (missing {r['file']})"); continue
        d = np.load(path)
        fs = float(d["fs"]); kp = d["keyphasor"]
        pulses = detect_pulses(kp, fs, kc["threshold_v"], kc["edge"])
        rpm = float(np.median(instantaneous_rpm(pulses))) if len(pulses) > 2 else float("nan")
        parts = []
        for p in PROBES:
            v1 = harmonic_vectors(d[p], fs, kp, cfg)[1]
            amp = abs(v1) * to_mils
            acc[p].append(amp)
            parts.append(f"{p}={amp:5.3f}mils∠{np.degrees(np.angle(v1)):+6.1f}")
        print(f"  m{r['mount_idx']}a{r['acq_idx']}  {rpm:6.1f}rpm  " + "  ".join(parts))

    print("\n1X amplitude mean ± std (mils):")
    for p in PROBES:
        a = np.array(acc[p])
        print(f"  {p:9s}  {a.mean():.4f} ± {a.std():.4f}")


if __name__ == "__main__":
    main()
