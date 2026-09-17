"""
Influence Coefficient Method on the LOGGED data — quantitative unbalance estimate.

Uses a baseline + one trial run per balancing disk to build the influence matrix,
then inverts the baseline response to the residual unbalance at each disk (g.mm and
angle), with the equivalent correction. Sensors = all four probes (over-determined,
least-squares).

    python -m scripts.icm --speed N1 --baseline BASE --trial-d1 A007 --trial-d2 A015

Trial mass/angle/radius are read from datasets/conditions.csv for the trial conditions.
"""
from __future__ import annotations

import argparse
import csv
import numpy as np

from config import load_config, ROOT
from processing.order_tracking import harmonic_vectors, detect_pulses, instantaneous_rpm
from estimation.icm import (influence_matrix, estimate_unbalance, to_mass_at_radius,
                            correction_weights, condition_report)

DATASETS = ROOT / "datasets"
SENSORS = ["plane1_x", "plane1_y", "plane2_x", "plane2_y"]


def _index():
    with open(DATASETS / "index.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _conditions():
    with open(DATASETS / "conditions.csv", newline="", encoding="utf-8") as f:
        return {r["condition_id"]: r for r in csv.DictReader(f)}


def mean_1x(cid, speed, cfg, to_mils):
    """Mean complex 1X (mils) per sensor over all logged runs of cid@speed."""
    kc = cfg["daq"]["keyphasor"]
    rows = [r for r in _index() if r["condition_id"] == cid and r["speed_id"] == speed]
    if not rows:
        raise SystemExit(f"no logged runs for {cid} @ {speed}")
    per = {p: [] for p in SENSORS}
    rpms = []
    for r in rows:
        d = np.load(DATASETS / r["file"])
        fs = float(d["fs"]); kp = d["keyphasor"]
        pulses = detect_pulses(kp, fs, kc["threshold_v"], kc["edge"])
        rpms.append(np.median(instantaneous_rpm(pulses)) if len(pulses) > 2 else np.nan)
        for p in SENSORS:
            per[p].append(harmonic_vectors(d[p], fs, kp, cfg)[1] * to_mils)
    vec = np.array([np.mean(per[p]) for p in SENSORS])
    return vec, len(rows), float(np.nanmean(rpms))


def mean_1x_file(path, cfg, to_mils):
    """Per-sensor complex 1X (mils) from a single raw npz (e.g. a slow-roll run)."""
    kc = cfg["daq"]["keyphasor"]
    d = np.load(path)
    fs = float(d["fs"]); kp = d["keyphasor"]
    return np.array([harmonic_vectors(d[p], fs, kp, cfg)[1] * to_mils for p in SENSORS])


def trial_W(cid, conds):
    """Complex trial unbalance (g.mm) from a trial condition's disk mass/angle."""
    c = conds[cid]
    for disk in ("disk1", "disk2"):
        m = float(c[f"{disk}_mass_g"] or 0)
        if m > 0:
            a = float(c[f"{disk}_angle_deg"] or 0)
            r = float(c["radius_mm"])
            return m * r * np.exp(1j * np.radians(a)), disk, m, a, r
    raise SystemExit(f"{cid} has no trial mass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", default="N1")
    ap.add_argument("--baseline", default="BASE")
    ap.add_argument("--trial-d1", default="A007")
    ap.add_argument("--trial-d2", default="A015")
    ap.add_argument("--runout", default=None,
                    help="npz of a slow-roll run; its 1X is subtracted from the baseline")
    a = ap.parse_args()

    cfg = load_config()
    conds = _conditions()
    sens = cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"]
    to_mils = 1000.0 / (sens * 25.4)

    V0, n0, rpm0 = mean_1x(a.baseline, a.speed, cfg, to_mils)
    Vt1, n1, _ = mean_1x(a.trial_d1, a.speed, cfg, to_mils)
    Vt2, n2, _ = mean_1x(a.trial_d2, a.speed, cfg, to_mils)
    W1, dk1, m1, ang1, r = trial_W(a.trial_d1, conds)
    W2, dk2, m2, ang2, _ = trial_W(a.trial_d2, conds)

    print(f"ICM @ {a.speed}  (~{rpm0:.0f} rpm)")
    print(f"  baseline {a.baseline} ({n0} runs);  trial-1 {a.trial_d1}={m1} g@{ang1:.0f}° on {dk1} "
          f"({n1} runs);  trial-2 {a.trial_d2}={m2} g@{ang2:.0f}° on {dk2} ({n2} runs)\n")

    # runout cancels inside the (trial - baseline) differences, so A is unaffected;
    # it only needs removing from the baseline that we invert.
    V0u = V0.copy()
    if a.runout:
        R = mean_1x_file(a.runout, cfg, to_mils)
        V0u = V0 - R
        print(f"  runout-compensated: subtracted slow-roll 1X ({a.runout})\n")
    else:
        print("  WARNING: no --runout given; estimate includes slow-roll runout (inflated)\n")

    A = influence_matrix(V0, [{"mass": W1, "response": Vt1}, {"mass": W2, "response": Vt2}])
    cond = condition_report(A)["condition_number"]
    U0 = estimate_unbalance(A, V0u)         # residual unbalance per disk (g.mm ∠°)
    W = correction_weights(U0)              # correction to add
    fit = A @ U0 - V0u                      # least-squares fit residual (mils)

    print(f"influence-matrix condition number: {cond:.1f}\n")
    print("RESIDUAL UNBALANCE (present on the rotor):")
    for i, disk in enumerate(("disk 1", "disk 2")):
        u = U0[i]; c = to_mass_at_radius(W[i], r)
        print(f"  {disk}: {abs(u):6.2f} g·mm ∠{np.degrees(np.angle(u)):+6.1f}°   "
              f"(= {abs(u)/r:.3f} g @ {r:.0f} mm)")
        print(f"         -> correction: add {c['mass_g']:.3f} g @ {c['angle_deg']:+.1f}°")
    print(f"\nmodel fit residual per probe (mils): {np.round(np.abs(fit), 4)}  "
          f"(small = the 2-plane model explains the runout-free baseline)")


if __name__ == "__main__":
    main()
