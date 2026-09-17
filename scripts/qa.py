"""
Data-quality QA of the logged campaign — flags condition/speeds worth redoing.

Checks (from index.csv + speeds.csv, no .npz load):
  - acquisition count per condition/speed (expect 3),
  - measured rpm vs the speed's target (off-target),
  - rpm stability within a condition (did the speed drift during capture).

    python -m scripts.qa
"""
from __future__ import annotations

import csv
from collections import defaultdict

from config import ROOT

DATASETS = ROOT / "datasets"
TARGET_ACQ = 3
OFF_TARGET_PCT = 4.0     # mean rpm vs speed target
SPREAD_PCT = 2.0         # max-min rpm within a condition


def load(name):
    with open(DATASETS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    idx = load("index.csv")
    target = {r["speed_id"]: float(r["rpm"]) for r in load("speeds.csv") if r["rpm"].strip()}

    groups = defaultdict(list)
    for r in idx:
        try:
            groups[(r["condition_id"], r["speed_id"])].append(float(r["rpm_measured"]))
        except ValueError:
            pass

    flags = []
    for (cid, speed), rpms in sorted(groups.items()):
        n = len(rpms); mean = sum(rpms) / n; mn, mx = min(rpms), max(rpms)
        issues = []
        if n != TARGET_ACQ:
            issues.append(f"{n} acqs (expected {TARGET_ACQ})")
        tgt = target.get(speed)
        if tgt and abs(mean - tgt) / tgt * 100 > OFF_TARGET_PCT:
            issues.append(f"rpm {mean:.0f} vs target {tgt:.0f} ({abs(mean-tgt)/tgt*100:.1f}% off)")
        if mean and (mx - mn) / mean * 100 > SPREAD_PCT:
            issues.append(f"rpm drifted {mn:.0f}-{mx:.0f} ({(mx-mn)/mean*100:.1f}%) within condition")
        if issues:
            flags.append((cid, speed, issues))

    print(f"QA over {len(groups)} condition/speed groups ({len(idx)} acquisitions):\n")
    if not flags:
        print("  ✅ all clean — counts correct, rpm on-target and stable. Nothing to redo.")
    else:
        print(f"  ⚠ {len(flags)} to review (select the row in the GUI and press 'Redo selected'):")
        for cid, speed, issues in flags:
            print(f"    {cid} @ {speed}: " + "; ".join(issues))


if __name__ == "__main__":
    main()
