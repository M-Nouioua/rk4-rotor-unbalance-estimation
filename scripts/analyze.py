"""
Full validation analysis of the completed severity/phase campaign.

Builds the ICM influence matrix per speed from BASE + A007 (1.6 g D1 trial) +
A015 (1.6 g D2 trial), then for every OTHER loaded condition estimates the ADDED
unbalance and compares to the known screw:

    added_U = A^-1 (V_test - V_baseline)      (runout & residual cancel in the diff)

Outputs: estimated-vs-true magnitude regression (R^2, RMSE, slope), phase error,
plane-localization accuracy, cross-talk; per speed and combined. Saves figures to
analysis/ and a per-condition table to analysis/results.csv.

    python -m scripts.analyze
"""
from __future__ import annotations

import csv
import numpy as np

from config import load_config, ROOT
from processing.order_tracking import harmonic_vectors

DATASETS = ROOT / "datasets"
ANALYSIS = ROOT / "analysis"
SENSORS = ["plane1_x", "plane1_y", "plane2_x", "plane2_y"]
BASE, TRIAL_D1, TRIAL_D2 = "BASE", "A007", "A015"
CAL = {BASE, TRIAL_D1, TRIAL_D2}
# The measured 1X phase runs opposite to the mounting-angle sense (whirl/probe
# handedness). Conjugating the measured vectors aligns them (fixes the +/-90 flips).
WHIRL_CONJ = True
TEST_BLOCKS = {"A_severity", "B_phase", "C_fine_phase"}
LOAD_THRESH = 6.0        # g.mm (=0.2 g) to call a disk "loaded" in localization


def load(name):
    with open(DATASETS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_1x(cid, speed, cfg, to_mils, index):
    kc = cfg["daq"]["keyphasor"]
    rows = [r for r in index if r["condition_id"] == cid and r["speed_id"] == speed]
    if not rows:
        return None
    per = {p: [] for p in SENSORS}
    for r in rows:
        d = np.load(DATASETS / r["file"]); fs = float(d["fs"]); kp = d["keyphasor"]
        for p in SENSORS:
            per[p].append(harmonic_vectors(d[p], fs, kp, cfg)[1] * to_mils)
    vec = np.array([np.mean(per[p]) for p in SENSORS])
    return np.conj(vec) if WHIRL_CONJ else vec


def true_added(c):
    r = float(c["radius_mm"])
    u1 = float(c["disk1_mass_g"] or 0) * r * np.exp(1j * np.radians(float(c["disk1_angle_deg"] or 0)))
    u2 = float(c["disk2_mass_g"] or 0) * r * np.exp(1j * np.radians(float(c["disk2_angle_deg"] or 0)))
    return np.array([u1, u2])


def analyse_speed(speed, cfg, to_mils, index, conds):
    V0 = mean_1x(BASE, speed, cfg, to_mils, index)
    Vt1 = mean_1x(TRIAL_D1, speed, cfg, to_mils, index)
    Vt2 = mean_1x(TRIAL_D2, speed, cfg, to_mils, index)
    W1 = true_added(conds[TRIAL_D1])[0]; W2 = true_added(conds[TRIAL_D2])[1]
    A = np.column_stack([(Vt1 - V0) / W1, (Vt2 - V0) / W2])   # 4x2 complex

    rows = []
    for cid, c in conds.items():
        if c["block"] not in TEST_BLOCKS or cid in CAL:
            continue
        Vt = mean_1x(cid, speed, cfg, to_mils, index)
        if Vt is None:
            continue
        est = np.linalg.lstsq(A, Vt - V0, rcond=None)[0]      # [disk1, disk2] g.mm
        tru = true_added(c)
        for k in (0, 1):
            rows.append({"cid": cid, "config": c["config"], "disk": k + 1,
                         "true_mag": abs(tru[k]), "true_ang": np.degrees(np.angle(tru[k])),
                         "est_mag": abs(est[k]), "est_ang": np.degrees(np.angle(est[k])),
                         "loaded": abs(tru[k]) > 1e-6})
    return A, rows


def regression(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return {}
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))              # est vs true (1:1)
    return {"slope": slope, "intercept": intercept, "r2": r2, "rmse": rmse, "n": len(x)}


def ang_err(a, b):
    return (a - b + 180) % 360 - 180


def main():
    cfg = load_config()
    to_mils = 1000.0 / (cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"] * 25.4)
    index = load("index.csv")
    conds = {c["condition_id"]: c for c in load("conditions.csv")}
    speeds = sorted({r["speed_id"] for r in index if r["condition_id"] not in ("",)}) or ["N1"]
    speeds = [s for s in ("N1", "N2") if s in {r["speed_id"] for r in index}]

    ANALYSIS.mkdir(exist_ok=True)
    all_rows = {}
    for s in speeds:
        A, rows = analyse_speed(s, cfg, to_mils, index, conds)
        all_rows[s] = rows
        loaded = [r for r in rows if r["loaded"]]
        unloaded = [r for r in rows if not r["loaded"]]
        mag = regression([r["true_mag"] for r in loaded], [r["est_mag"] for r in loaded])
        ph = [abs(ang_err(r["est_ang"], r["true_ang"])) for r in loaded]
        crosstalk = np.mean([r["est_mag"] for r in unloaded]) if unloaded else 0.0
        # localization: loaded-disk set per condition
        by_cid = {}
        for r in rows:
            by_cid.setdefault(r["cid"], {}).setdefault("t", set()); by_cid[r["cid"]].setdefault("e", set())
            if r["loaded"]:
                by_cid[r["cid"]]["t"].add(r["disk"])
            if r["est_mag"] > LOAD_THRESH:
                by_cid[r["cid"]]["e"].add(r["disk"])
        loc_ok = sum(1 for v in by_cid.values() if v["t"] == v["e"])
        print(f"\n===== {s} =====   ({len(by_cid)} test conditions, {len(loaded)} loaded points)")
        print(f"  MAGNITUDE  slope={mag['slope']:.3f}  R²={mag['r2']:.4f}  "
              f"RMSE={mag['rmse']:.2f} g·mm  (n={mag['n']})")
        print(f"  PHASE      mean|err|={np.mean(ph):.1f}°  max={np.max(ph):.1f}°")
        print(f"  CROSS-TALK est on unloaded disks: {crosstalk:.2f} g·mm (want ~0)")
        print(f"  LOCALIZE   correct loaded-disk set: {loc_ok}/{len(by_cid)}")

    _save_figs(all_rows, speeds)
    _save_csv(all_rows)
    print(f"\nfigures -> {ANALYSIS}/regression.png, phase.png ; table -> {ANALYSIS}/results.csv")


def _save_figs(all_rows, speeds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipped figures"); return
    colors = {"N1": "#2980b9", "N2": "#8e44ad"}

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = 0
    for s in speeds:
        L = [r for r in all_rows[s] if r["loaded"]]
        t = [r["true_mag"] for r in L]; e = [r["est_mag"] for r in L]
        ax.scatter(t, e, s=22, alpha=.7, color=colors.get(s, "#333"), label=f"{s}")
        lim = max(lim, max(t + e) if t else 0)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    ax.set_xlabel("true added unbalance (g·mm)"); ax.set_ylabel("estimated (g·mm)")
    ax.set_title("Unbalance: estimated vs true"); ax.legend(); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(ANALYSIS / "regression.png", dpi=140)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for s in speeds:
        L = [r for r in all_rows[s] if r["loaded"]]
        t = [r["true_mag"] for r in L]
        er = [ang_err(r["est_ang"], r["true_ang"]) for r in L]
        ax2.scatter(t, er, s=22, alpha=.7, color=colors.get(s, "#333"), label=s)
    ax2.axhline(0, color="k", lw=.6)
    ax2.set_xlabel("true added unbalance (g·mm)"); ax2.set_ylabel("phase error (deg)")
    ax2.set_title("Phase error vs severity"); ax2.legend()
    fig2.tight_layout(); fig2.savefig(ANALYSIS / "phase.png", dpi=140)


def _save_csv(all_rows):
    with open(ANALYSIS / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speed", "condition", "config", "disk", "true_gmm", "true_deg",
                    "est_gmm", "est_deg"])
        for s, rows in all_rows.items():
            for r in rows:
                w.writerow([s, r["cid"], r["config"], r["disk"],
                            f"{r['true_mag']:.2f}", f"{r['true_ang']:.1f}",
                            f"{r['est_mag']:.2f}", f"{r['est_ang']:.1f}"])


if __name__ == "__main__":
    main()
