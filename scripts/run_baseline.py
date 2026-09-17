"""
Acquire a run, extract 1X/2X vectors at both planes, print a health summary.

Usage:
    python -m scripts.run_baseline --duration 10 --rpm 6000 --out datasets/baseline.npz

Runs against live NI hardware if nidaqmx is available; otherwise use --sim to
generate a synthetic imbalance+misalignment signal so the pipeline is testable
offline.
"""
from __future__ import annotations

import argparse
import numpy as np

from config import load_config
from processing.order_tracking import harmonic_vectors, amp_phase, detect_pulses, instantaneous_rpm
from processing.spectra import diagnostic_features


def synthetic_run(cfg, rpm=6000.0, duration=5.0, seed=0):
    """Fake 5-channel record: 1X (imbalance) + 2X (misalignment) + noise."""
    rng = np.random.default_rng(seed)
    fs = cfg["daq"]["sample_rate_hz"]
    t = np.arange(int(duration * fs)) / fs
    f = rpm / 60.0
    kp = -5.0 * (np.mod(f * t, 1.0) < 0.02).astype(float)         # 1 neg pulse/rev
    def chan(a1, p1, a2, p2):
        return (a1 * np.sin(2 * np.pi * f * t + p1)
                + a2 * np.sin(2 * np.pi * 2 * f * t + p2)
                + 0.02 * rng.standard_normal(t.size))
    return {
        "keyphasor": kp,
        "plane1_x": chan(0.8, 0.3, 0.3, 1.1),
        "plane1_y": chan(0.7, 0.3 - np.pi / 2, 0.28, 1.1 - np.pi / 2),
        "plane2_x": chan(0.5, 1.7, 0.35, 0.4),
        "plane2_y": chan(0.45, 1.7 - np.pi / 2, 0.33, 0.4 - np.pi / 2),
        "fs": fs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--rpm", type=float, default=6000.0)
    ap.add_argument("--sim", action="store_true", help="synthetic data (no NI HW)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.sim:
        rec = synthetic_run(cfg, rpm=args.rpm, duration=args.duration)
    else:
        from acquisition.daq import acquire_record
        rec = acquire_record(cfg, args.duration)

    fs = rec["fs"]
    kp = rec["keyphasor"]
    pulses = detect_pulses(kp, fs, cfg["daq"]["keyphasor"]["threshold_v"],
                           cfg["daq"]["keyphasor"]["edge"])
    rpm_meas = float(np.median(instantaneous_rpm(pulses))) if len(pulses) > 2 else float("nan")
    print(f"keyphasor pulses: {len(pulses)}   measured speed: {rpm_meas:.1f} rpm")

    o_p1x = harmonic_vectors(rec["plane1_x"], fs, kp, cfg)
    o_p2x = harmonic_vectors(rec["plane2_x"], fs, kp, cfg)

    for tag, o in (("plane1_x", o_p1x), ("plane2_x", o_p2x)):
        for k in cfg["order_tracking"]["orders"]:
            a, p = amp_phase(o[k])
            print(f"  {tag}  {k}X: {a:.4f} ∠ {p:7.1f}°")

    feats = diagnostic_features(o_p1x, o_p2x)
    print("features:", {k: round(v, 4) for k, v in feats.items()})

    if args.out:
        np.savez(args.out, **{k: v for k, v in rec.items() if k != "fs"}, fs=fs)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
