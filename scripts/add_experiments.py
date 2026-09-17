"""
Add the post-blind follow-up experiments to the campaign (idempotent).

The blind validation exposed three concrete gaps; this wires the fix into the
acquisition app by extending datasets/conditions.csv and datasets/speeds.csv:

  T_twoplane (12)  general two-plane conditions — high per-disk magnitude (24-60 g.mm)
                   at ARBITRARY relative angles (45/90/135deg), not just in/anti-phase.
                   Fills the out-of-distribution hole where the ML underestimated the
                   blind two-plane loads. Runs at N1|N2.
  U_anchor  (4)    uncertainty anchors — low/mid/high + off-axis, n_mount=6 independent
                   re-mounts each, to measure setup variance (sigma_mount) for the
                   GUM budget. Currently ZERO conditions have a repeated mount. Runs at N1|N2.
  N3 = 660 rpm     a third (0.40 N_c) sub-critical dwell for a subset (baseline + trials +
                   D1 severity ladder + a couple two-plane) -> multi-speed ICM / alpha(omega).

Existing conditions are relabeled run_at_speeds "all" -> "N1|N2" (they were only ever
run at N1/N2); the N3 subset gets "N1|N2|N3". Re-runnable: existing rows are left in
place and only missing conditions/speeds are appended.

    python -m scripts.add_experiments            # apply
    python -m scripts.add_experiments --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import csv

from config import ROOT

DATASETS = ROOT / "datasets"
FIELDS = ["condition_id", "block", "config", "disk1_mass_g", "disk1_angle_deg",
          "disk2_mass_g", "disk2_angle_deg", "radius_mm", "U1_gmm", "U2_gmm",
          "n_mount", "is_anchor", "run_at_speeds", "notes"]
R = 30.0

# General two-plane: (id, d1_mass, d1_ang, d2_mass, d2_ang) — arbitrary relative angle.
TWOPLANE = [
    ("T01", 0.8,   0, 0.8,  45), ("T02", 0.8,   0, 0.8,  90),
    ("T03", 0.8,   0, 0.8, 135), ("T04", 1.6,   0, 1.6,  45),
    ("T05", 1.6,   0, 1.6,  90), ("T06", 1.6,   0, 0.8,  90),
    ("T07", 1.2,   0, 0.4, 135), ("T08", 2.0,   0, 1.2,  90),
    ("T09", 1.2,  45, 0.8, 180), ("T10", 1.6,  90, 1.6, 270),
    ("T11", 1.0,   0, 1.6, 225), ("T12", 2.0,   0, 2.0,  45),
]
# Uncertainty anchors: single-disk D1 (one screw to re-mount), n_mount=6.
ANCHORS = [
    ("U01", 0.2,   0, "low  (6 g.mm)"),  ("U02", 0.8,  0, "mid  (24 g.mm)"),
    ("U03", 1.6,   0, "high (48 g.mm)"), ("U04", 0.8, 90, "off-axis (24 g.mm @90)"),
]
# Conditions to also acquire at N3 (multi-speed subset).
N3_SUBSET = {"BASE", "A007", "A015", "A002", "A004", "A005", "A006", "A008",
             "T02", "T05", "T08", "U03"}


def two_plane_row(cid, d1m, d1a, d2m, d2a):
    return {"condition_id": cid, "block": "T_twoplane", "config": "twoplane",
            "disk1_mass_g": d1m, "disk1_angle_deg": d1a,
            "disk2_mass_g": d2m, "disk2_angle_deg": d2a, "radius_mm": R,
            "U1_gmm": round(d1m * R, 2), "U2_gmm": round(d2m * R, 2),
            "n_mount": 3, "is_anchor": 0, "run_at_speeds": "N1|N2",
            "notes": f"two-plane general (D1 {d1m}g@{d1a} / D2 {d2m}g@{d2a}) - OOD training fill"}


def anchor_row(cid, m, a, tag):
    return {"condition_id": cid, "block": "U_anchor", "config": "D1",
            "disk1_mass_g": m, "disk1_angle_deg": a, "disk2_mass_g": 0.0,
            "disk2_angle_deg": 0, "radius_mm": R, "U1_gmm": round(m * R, 2),
            "U2_gmm": 0.0, "n_mount": 6, "is_anchor": 1, "run_at_speeds": "N1|N2",
            "notes": f"uncertainty anchor {tag} - 6 independent re-mounts for sigma_mount"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:   # utf-8, no BOM
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conds = read_csv(DATASETS / "conditions.csv")
    have = {c["condition_id"] for c in conds}
    # ensure the run_at_speeds column exists on every row
    for c in conds:
        c.setdefault("run_at_speeds", "all")

    new = [two_plane_row(*t) for t in TWOPLANE if t[0] not in have]
    new += [anchor_row(*a) for a in ANCHORS if a[0] not in have]
    conds += new

    # scope speeds: existing "all" -> "N1|N2"; N3 subset -> "N1|N2|N3"
    relabeled = 0
    for c in conds:
        cur = (c.get("run_at_speeds") or "all").strip()
        base = "N1|N2" if cur in ("", "all") else cur
        speeds = set(base.split("|"))
        if c["condition_id"] in N3_SUBSET:
            speeds.add("N3")
        newval = "|".join(s for s in ("N1", "N2", "N3") if s in speeds)
        if newval != cur:
            relabeled += 1
        c["run_at_speeds"] = newval

    speeds = read_csv(DATASETS / "speeds.csv")
    sids = {s["speed_id"] for s in speeds}
    add_n3 = "N3" not in sids
    if add_n3:
        speeds.append({"speed_id": "N3", "rpm": 660, "fraction_Nc": 0.40,
                       "trial_mass_g": 1.6,
                       "notes": "0.40 Nc sub-critical 3rd dwell; multi-speed ICM / alpha(omega); "
                                "safe (far below 1400-1900 band); resize motor to 660 rpm"})

    print(f"two-plane added : {sum(1 for t in TWOPLANE if t[0] not in have)}")
    print(f"anchors added   : {sum(1 for a in ANCHORS if a[0] not in have)}")
    print(f"run_at_speeds relabeled rows: {relabeled}")
    print(f"N3 speed added  : {add_n3}")
    print(f"N3 subset ({len(N3_SUBSET)}): {', '.join(sorted(N3_SUBSET))}")
    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    write_csv(DATASETS / "conditions.csv", FIELDS, conds)
    write_csv(DATASETS / "speeds.csv",
              ["speed_id", "rpm", "fraction_Nc", "trial_mass_g", "notes"], speeds)
    print(f"\nwrote conditions.csv ({len(conds)} rows) and speeds.csv ({len(speeds)} rows)")


if __name__ == "__main__":
    main()
