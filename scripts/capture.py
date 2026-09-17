"""
Capture ONE acquisition for a matrix condition + speed and log it to
datasets/index.csv (raw .npz saved alongside). Prints the 1X/2X vectors and the
peak orbit, and refuses to save if the orbit exceeds the safety trip (unless --force).

    python -m scripts.capture --condition BASE --speed N1
    python -m scripts.capture --condition ICM_D1 --speed N1 --notes "1.6 g trial on disk1 @0"

Commanded rpm is read from datasets/speeds.csv; the measured rpm comes from the
keyphasor. mount/acq auto-advance unless given explicitly.
"""
from __future__ import annotations

import argparse
from gui.session import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--speed", default="N1")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--mount", type=int, default=None)
    ap.add_argument("--acq", type=int, default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--force", action="store_true", help="save even if orbit trips")
    a = ap.parse_args()

    s = Session()
    sens = s.cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"]
    to_mils = 1000.0 / (sens * 25.4)          # V -> mils

    print(f"condition {a.condition}:  {s.label_text(a.condition)}")
    rpm_cmd = s.speed_rpm(a.speed) or 0.0
    m, acq = s.next_indices(a.condition, a.speed)
    if a.mount:
        m = a.mount
    if a.acq:
        acq = a.acq
    print(f"speed {a.speed} = {rpm_cmd:.0f} rpm cmd   |   mount {m}, acq {acq}")

    rec = s.acquire(a.duration, simulate=a.sim, rpm=rpm_cmd or 1000)
    feats = s.features(rec)
    print(f"measured speed: {feats['rpm']:.1f} rpm")
    for p, d in feats["planes"].items():
        o = d["orders"]
        print(f"  {p}: 1X {o[1][0]*to_mils:6.3f} mils ∠{o[1][1]:+6.1f}°   "
              f"2X {o[2][0]*to_mils:6.3f} mils ∠{o[2][1]:+6.1f}°   peak {d['peak_mils']:.2f} mils")
    trip = s.orbit_trip(feats)
    print(f"max orbit: {feats['max_orbit_mils']:.2f} mils  ->  {'** TRIP **' if trip else 'OK'}")

    if trip and not a.force:
        print("NOT saved — orbit over safety limit. Reduce mass/speed, or re-run with --force.")
        return
    res = s.save(rec, a.condition, a.speed, rpm_cmd, m, acq, feats["rpm"], notes=a.notes)
    pg = s.progress()
    print(f"saved {res['row']['file']}   |   progress {pg['done']}/{pg['planned']}")


if __name__ == "__main__":
    main()
