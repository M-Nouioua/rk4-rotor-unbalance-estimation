"""
Live channel / wiring check: read a short block from the real DAQ and report
per-channel RMS + peak, plus keyphasor pulse count and speed.

Works whether or not the rotor is spinning — use it to confirm all 5 channels
are wired and responding before a run.

    python -m scripts.test_channels --duration 1
"""
from __future__ import annotations

import argparse
import numpy as np

from config import load_config
from acquisition.daq import acquire_record, CHANNEL_ORDER
from processing.order_tracking import detect_pulses, instantaneous_rpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    rec = acquire_record(cfg, args.duration)
    fs = rec["fs"]
    print(f"acquired {args.duration:.1f} s @ {fs} S/s\n")
    print(f"{'channel':10s} {'device':16s} {'RMS (V)':>10s} {'pk (V)':>10s}")
    for name in CHANNEL_ORDER:
        x = np.asarray(rec[name], dtype=float)
        dev = cfg["daq"]["channels"][name]["device"]
        print(f"{name:10s} {dev:16s} {np.sqrt(np.mean(x**2)):10.4f} {np.max(np.abs(x)):10.4f}")

    kc = cfg["daq"]["keyphasor"]
    pulses = detect_pulses(rec["keyphasor"], fs, kc["threshold_v"], kc["edge"])
    print(f"\nkeyphasor pulses: {len(pulses)}")
    if len(pulses) > 2:
        print(f"measured speed  : {np.median(instantaneous_rpm(pulses)):.1f} rpm")
    else:
        print("measured speed  : (rotor not spinning or threshold needs tuning)")


if __name__ == "__main__":
    main()
