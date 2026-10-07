"""
Export the train/test membership of every condition under each partition.

The six partitions are the central claim of the paper and are currently carried
by prose alone. This writes the data for a figure that shows them: one row per
condition, with its position in the design space, and for each protocol whether
that condition is trained on or tested in one representative fold.

Showing a single fold per protocol, rather than averaging over folds, keeps the
picture unambiguous: a reader sees exactly which part of the design is withheld.
The fold chosen is the one that withholds the first group, which is the natural
illustration for a leave-one-out protocol.

    python -m scripts.export_partition_map

Writes analysis/fig_data/partition_map.csv.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from config import ROOT
from estimation.twin import CALIB
from scripts.benchmark import _protocols, _protocol_splits

OUT = ROOT / "analysis" / "fig_data" / "partition_map.csv"

ORDER = ["random_acquisition", "condition", "magnitude_interpolating",
         "magnitude_extrapolating", "configuration", "angle_sector"]

LABEL = {"random_acquisition": "Random records",
         "condition": "Condition",
         "magnitude_interpolating": "Magnitude, interpolating",
         "magnitude_extrapolating": "Magnitude, extrapolating",
         "configuration": "Configuration",
         "angle_sector": "Angle sector"}


def main() -> int:
    lab = pd.read_csv(ROOT / "analysis" / "features.csv")
    lab = lab[lab.labeled == 1].reset_index(drop=True)

    total = (lab.U1_mag + lab.U2_mag).to_numpy(float)
    ang = np.where(lab.U1_mag.to_numpy() >= lab.U2_mag.to_numpy(),
                   lab.U1_ang.fillna(0).to_numpy(), lab.U2_ang.fillna(0).to_numpy())
    sector = np.round(ang / 45).astype(int) % 8
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()

    specs = _protocols(lab)
    rows = []
    for name in ORDER:
        spec = specs[name]
        folds = _protocol_splits(len(lab), spec)
        if not folds:
            continue
        # The illustrative fold: the one withholding the first group, which for a
        # leave-one-out protocol is the cleanest single picture of the split.
        tr, te = folds[0]
        role = np.full(len(lab), "train", dtype=object)
        role[te] = "test"
        role[~scored] = "calibration"

        # Collapse to one row per condition. Under a grouped protocol every
        # record of a condition shares its role. Under the record-level control
        # they do not, and a condition whose repeats fall on both sides is
        # marked "split": that is the leakage the control exists to expose, and
        # it should be visible rather than averaged away.
        def collapse(s: pd.Series) -> str:
            roles = set(s)
            if roles == {"calibration"}:
                return "calibration"
            roles.discard("calibration")
            return roles.pop() if len(roles) == 1 else "split"

        d = pd.DataFrame({"condition_id": lab.condition_id, "role": role,
                          "sector": sector, "total_gmm": np.round(total, 1),
                          "config": lab.config})
        g = (d.groupby("condition_id")
               .agg(role=("role", collapse),
                    sector=("sector", "first"),
                    total_gmm=("total_gmm", "first"),
                    config=("config", "first"))
               .reset_index())
        g["protocol"] = name
        g["protocol_label"] = LABEL[name]
        g["n_folds"] = len(folds)
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"  {out.condition_id.nunique()} conditions x {out.protocol.nunique()} protocols")
    for name in ORDER:
        s = out[out.protocol == name]
        if s.empty:
            continue
        n_te = int((s.role == "test").sum())
        n_tr = int((s.role == "train").sum())
        n_ca = int((s.role == "calibration").sum())
        print(f"    {LABEL[name]:26s} fold 1 of {s.n_folds.iloc[0]:2d}: "
              f"{n_tr:3d} train, {n_te:3d} test, {n_ca:2d} calibration")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
