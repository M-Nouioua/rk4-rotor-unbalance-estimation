"""
Rebuild datasets/index.csv from the saved .npz files (each embeds its own labels).
Use it to recover the log if index.csv is lost/corrupted, or after deleting bad runs.

    python -m scripts.rebuild_index
"""
from __future__ import annotations

import csv
import glob
import os
import time
import numpy as np

from config import ROOT
from gui.session import INDEX_FIELDS

DATASETS = ROOT / "datasets"


def val(d, k, default=""):
    return d[k].item() if k in d.files else default


def main():
    rows = []
    for path in sorted(glob.glob(str(DATASETS / "*.npz"))):
        try:
            d = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}"); continue
        if "condition_id" not in d.files:
            continue                                   # not an acquisition record
        name = os.path.basename(path)
        rows.append({
            "run_id": name[:-4],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path))),
            "condition_id": str(val(d, "condition_id")),
            "speed_id": str(val(d, "speed_id")),
            "rpm_commanded": val(d, "rpm_commanded"),
            "rpm_measured": round(float(val(d, "rpm_measured", 0)), 1),
            "mount_idx": int(val(d, "mount_idx", 0)),
            "acq_idx": int(val(d, "acq_idx", 0)),
            "file": name, "T_ambient_C": "", "notes": "",
        })
    rows.sort(key=lambda r: (r["condition_id"], r["speed_id"], r["mount_idx"], r["acq_idx"]))
    with open(DATASETS / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"rebuilt index.csv from {len(rows)} .npz record(s)")
    for r in rows:
        print(f"  {r['condition_id']:8s} {r['speed_id']:3s} m{r['mount_idx']}a{r['acq_idx']} "
              f"{r['rpm_measured']:.0f}rpm  {r['file']}")


if __name__ == "__main__":
    main()
