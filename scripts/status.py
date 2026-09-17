"""
Campaign progress snapshot: how many conditions are complete vs pending, per
speed and per block, from datasets/index.csv + conditions.csv.

    python -m scripts.status            # print
    python -m scripts.status --pending  # also list pending condition ids
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from config import ROOT

DATASETS = ROOT / "datasets"
TARGET = 3          # acquisitions per condition per speed to count as complete


def load(name):
    with open(DATASETS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending", action="store_true")
    a = ap.parse_args()

    conds = load("conditions.csv")
    idx = load("index.csv")
    speeds = sorted({r["speed_id"] for r in idx}) or ["N1", "N2"]

    done = defaultdict(int)
    for r in idx:
        done[(r["condition_id"], r["speed_id"])] += 1

    print(f"acquisitions logged: {len(idx)}   speeds: {', '.join(speeds)}   "
          f"(complete = >= {TARGET} acqs)\n")

    def applies(c, s):
        ras = (c.get("run_at_speeds") or "all").strip()
        return ras in ("", "all") or s in ras.replace(",", "|").split("|")

    def target(c):  # uncertainty anchors need n_mount independent re-mounts
        return int(c.get("n_mount") or 1) * TARGET if c.get("block") == "U_anchor" else TARGET

    blocks = ["baseline", "A_severity", "B_phase", "C_fine_phase", "LoD",
              "T_twoplane", "U_anchor", "ICM", "blind"]
    header = f"{'block':14s}" + "".join(f"{s:>12s}" for s in speeds)
    print(header); print("-" * len(header))
    pend = defaultdict(list)
    for b in blocks:
        bconds = [c for c in conds if c["block"] == b]
        if not bconds:
            continue
        cells = []
        for s in speeds:
            applic = [c for c in bconds if applies(c, s)]
            nd = sum(1 for c in applic if done[(c["condition_id"], s)] >= target(c))
            cells.append(f"{nd}/{len(applic)}" if applic else "—")
            for c in applic:
                if done[(c["condition_id"], s)] < target(c):
                    pend[s].append(c["condition_id"])
        print(f"{b:14s}" + "".join(f"{c:>12s}" for c in cells))

    tot = [c for c in conds if c["block"] not in ("blind", "ICM")]
    print("-" * len(header))
    for s in speeds:
        nd = sum(1 for c in tot if done[(c["condition_id"], s)] >= TARGET)
        print(f"CORE (labeled) complete @ {s}: {nd}/{len(tot)}")

    if a.pending:
        for s in speeds:
            print(f"\npending @ {s} ({len(pend[s])}): " + ", ".join(pend[s][:60]))


if __name__ == "__main__":
    main()
