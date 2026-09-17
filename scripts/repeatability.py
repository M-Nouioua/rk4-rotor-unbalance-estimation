"""
Back-to-back measurement repeatability of the order-tracked first-order response.

Section 2.4 of the manuscript benchmarks the forward model error of Section 6.7
against a measurement repeatability expressed in the same relative-norm measure.
Those figures were previously quoted without a script behind them, so they could
not be regenerated from the archive. This script produces both of them.

Two measures are reported, matching the two sentences in Section 2.4.

1. Absolute spread. At each probe, the standard deviation of the first-order
   amplitude across the three repeats of one condition and speed. Reported as the
   median and the 95th percentile over all probe-group pairs, in micrometres.

2. Relative spread, the measure used for forward prediction. For a record with
   baseline-subtracted first-order response vector V over the four probes,

       ||V - Vbar|| / ||Vbar||,

   with Vbar the complex mean over the three repeats. Complex averaging is used
   because averaging magnitude and phase separately is wrong across the
   +-180 degree boundary. Balanced conditions are excluded, since their |Vbar| is
   the small residual response and the ratio would be dominated by its
   denominator.

The feature table stores displacement in mils; both measures are converted to
micrometres here.

    python -m scripts.repeatability

Writes analysis/repeatability.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES = ROOT / "analysis" / "features.csv"
OUT = ROOT / "analysis" / "repeatability.json"

PROBES = ("plane1_x", "plane1_y", "plane2_x", "plane2_y")
UM_PER_MIL = 25.4


def main() -> int:
    d = pd.read_csv(FEATURES)
    lab = d[d.labeled == 1].copy() if "labeled" in d.columns else d.copy()

    V = np.column_stack([
        lab[f"{p}_resp_re"].to_numpy(float) + 1j * lab[f"{p}_resp_im"].to_numpy(float)
        for p in PROBES])
    applied = lab.U1_mag.to_numpy(float) + lab.U2_mag.to_numpy(float)

    amp_sd_um: list[float] = []      # measure 1, one entry per probe and group
    rel: list[float] = []            # measure 2, one entry per record
    n_groups = n_excluded = 0

    for _, idx in lab.groupby(["condition_id", "speed_id"]).groups.items():
        rows = lab.index.get_indexer(idx)
        if len(rows) < 2:
            continue
        Vg = V[rows]
        amp_sd_um.extend(np.abs(Vg).std(axis=0, ddof=1) * UM_PER_MIL)

        if applied[rows][0] <= 0:
            n_excluded += len(rows)
            continue
        mean = Vg.mean(axis=0)
        nm = np.linalg.norm(mean)
        if nm == 0:
            n_excluded += len(rows)
            continue
        rel.extend(np.linalg.norm(Vg - mean, axis=1) / nm)
        n_groups += 1

    a = np.asarray(amp_sd_um)
    r = np.asarray(rel)

    res = {
        "absolute": {
            "measure": "per-probe standard deviation of the first-order amplitude "
                       "across the three repeats of one condition and speed",
            "unit": "micrometre",
            "n_probe_group_pairs": int(a.size),
            "median": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
        },
        "relative": {
            "measure": "||V - Vbar|| / ||Vbar|| per record, Vbar the complex mean "
                       "of the repeats; balanced conditions excluded",
            "n_records": int(r.size),
            "n_condition_speed_groups": int(n_groups),
            "n_records_excluded_balanced": int(n_excluded),
            "median": round(float(np.median(r)), 4),
            "p90": round(float(np.percentile(r, 90)), 4),
        },
        "note": "Back-to-back only. The three repeats share one mounting of the "
                "added mass, so remounting variance is not included. Section 2.6 "
                "of the manuscript states this limitation.",
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print(f"  absolute spread  median {res['absolute']['median']:.4f} um, "
          f"p95 {res['absolute']['p95']:.4f} um "
          f"({res['absolute']['n_probe_group_pairs']} probe-group pairs)")
    print(f"  relative spread  median {res['relative']['median']:.4f} "
          f"over {res['relative']['n_records']} records "
          f"({res['relative']['n_records_excluded_balanced']} balanced excluded)")
    print(f"\nwritten to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
