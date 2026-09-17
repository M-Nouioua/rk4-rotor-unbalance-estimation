"""
Acquire a raw record and save it to .npz — for run-ups / coast-downs or any
manual capture (no processing, just the raw channels).

    python -m scripts.record --duration 60 --out datasets/runup1.npz

For a RUN-UP: start this, then ramp the motor from slow-roll up through the first
critical speed (and back down for a coast-down) within the acquisition window.
Then analyse it with:  python -m scripts.runup_critical --file datasets/runup1.npz --plot
"""
from __future__ import annotations

import argparse
import numpy as np

from config import load_config
from acquisition.daq import acquire_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config()
    print(f"acquiring {args.duration:.0f} s … ramp the motor through the first critical now")
    rec = acquire_record(cfg, args.duration)
    np.savez(args.out, **{k: v for k, v in rec.items() if k != "fs"}, fs=rec["fs"])
    print(f"saved raw record -> {args.out}  ({args.duration:.0f} s @ {rec['fs']} S/s)")


if __name__ == "__main__":
    main()
