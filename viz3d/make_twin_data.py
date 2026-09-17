"""
Build the shared state payload for the 3D digital-twin viewers (three.js + PyVista).

Emits viz3d/twin_state.json:
  * rig GEOMETRY (shaft / disks / bearings / probe planes) from config/rig.yaml,
  * for a set of representative conditions x speeds: the measured, runout-compensated
    complex 1X/2X/3X vectors at each probe plane (x & y), the true mounted unbalance
    (for the heavy-spot marker), and the measured rpm.

Both viewers reconstruct the shaft motion the same way: at rotation phase θ the lateral
displacement at a plane is Σ_k Re(V_k · e^{ikθ}); between/beyond the planes it is
interpolated along the shaft (zero at the bearings) — a data-driven, physically faithful
whirl, not a scripted animation.

    python viz3d/make_twin_data.py
"""
from __future__ import annotations

import csv
import json
import sys
import pathlib

ROOTDIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOTDIR))

import numpy as np
from config import load_config, ROOT
from processing.order_tracking import harmonic_vectors, detect_pulses, instantaneous_rpm

D = ROOT / "datasets"
SENS = ["plane1_x", "plane1_y", "plane2_x", "plane2_y"]
# representative spread: balanced, single-plane ladder, D2, static, couple, two-plane
CONDITIONS = ["BASE", "A002", "A004", "A006", "A008", "A012", "A020", "A028",
              "BLND01", "BLND05"]
SPEEDS = ["N1", "N2"]


def main():
    cfg = load_config()
    to_mils = 1000.0 / (cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"] * 25.4)
    kc = cfg["daq"]["keyphasor"]

    # raw runout reference (complex per sensor per harmonic, mils)
    r = np.load(D / "runout_slowroll.npz", allow_pickle=True)
    fr, kpr = float(r["fs"]), r["keyphasor"]
    runout = {p: {k: harmonic_vectors(r[p], fr, kpr, cfg)[k] * to_mils for k in (1, 2, 3)}
              for p in SENS}

    index = list(csv.DictReader(open(D / "index.csv", encoding="utf-8")))
    conds = {c["condition_id"]: c for c in csv.DictReader(open(D / "conditions.csv", encoding="utf-8"))}

    def phasors(cid, speed):
        rows = [x for x in index if x["condition_id"] == cid and x["speed_id"] == speed]
        if not rows:
            return None
        acc = {p: {1: [], 2: [], 3: []} for p in SENS}
        rpm = []
        for x in rows:
            d = np.load(D / x["file"], allow_pickle=True)
            fs, kp = float(d["fs"]), d["keyphasor"]
            pu = detect_pulses(kp, fs, kc["threshold_v"], kc["edge"])
            if len(pu) > 2:
                rpm.append(float(np.median(instantaneous_rpm(pu))))
            for p in SENS:
                h = harmonic_vectors(d[p], fs, kp, cfg)
                for k in (1, 2, 3):
                    acc[p][k].append(h[k] * to_mils - runout[p][k])
        mean = {p: {k: np.mean(acc[p][k]) for k in (1, 2, 3)} for p in SENS}
        return mean, (float(np.mean(rpm)) if rpm else float("nan"))

    def cx(v):
        return [round(float(v.real), 5), round(float(v.imag), 5)]

    def true_U(c, disk):
        m = float(c.get(f"disk{disk}_mass_g") or 0)
        a = float(c.get(f"disk{disk}_angle_deg") or 0)
        return {"mag_gmm": round(m * 30, 1), "mass_g": m, "ang_deg": a}

    # geometry
    rr = cfg["rotor"]
    L = float(rr["shaft"]["length_mm"]); nel = int(rr["shaft"]["n_elements"]); step = L / nel
    geom = {
        "shaft": {"length": L, "od": float(rr["shaft"]["outer_diameter_mm"])},
        "disks": [{"name": d["name"], "z": d["node"] * step,
                   "od": float(d["outer_diameter_mm"]), "thick": float(d["thickness_mm"])}
                  for d in rr["disks"]],
        "bearings": [b["node"] * step for b in rr["bearings"]],
        "planes": [{"name": "plane1", "z": rr["measurement_planes"]["plane1_node"] * step},
                   {"name": "plane2", "z": rr["measurement_planes"]["plane2_node"] * step}],
        "balance_radius": 30.0, "n_holes": 8,
    }

    out = {"geom": geom, "conditions": {}, "order": []}
    for cid in CONDITIONS:
        if cid not in conds:
            continue
        c = conds[cid]
        for sp in SPEEDS:
            res = phasors(cid, sp)
            if res is None:
                continue
            ph, rpm = res
            key = f"{cid}|{sp}"
            out["conditions"][key] = {
                "id": cid, "speed": sp, "config": c.get("config", ""), "rpm": round(rpm, 1),
                "U1": true_U(c, 1), "U2": true_U(c, 2),
                "planes": {
                    "plane1": {"x": [cx(ph["plane1_x"][k]) for k in (1, 2, 3)],
                               "y": [cx(ph["plane1_y"][k]) for k in (1, 2, 3)]},
                    "plane2": {"x": [cx(ph["plane2_x"][k]) for k in (1, 2, 3)],
                               "y": [cx(ph["plane2_y"][k]) for k in (1, 2, 3)]},
                },
            }
            out["order"].append(key)

    dst = ROOTDIR / "viz3d" / "twin_state.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}: {len(out['conditions'])} condition-states, "
          f"{len(out['order'])} keys")
    print("geometry:", {k: v for k, v in geom.items() if k != "disks"})


if __name__ == "__main__":
    main()
