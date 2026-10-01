"""
Export plot-ready data for the MATLAB figure scripts in `matlab/`.

Manuscript figures are rendered in MATLAB (Times New Roman, no titles, vector PDF),
so the Python side does no plotting for the paper. `scripts/figures.py` still writes
matplotlib PNGs, but those are working diagnostics and the interactive dashboard
only; they are not manuscript figures.

Every file written here is a flat CSV with explicit column names, derived only from
the generated analysis artifacts. Nothing is smoothed, clipped, or rescaled, so the
figures represent the underlying results exactly.

    python -m scripts.export_fig_data     ->  analysis/fig_data/*.csv
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from config import ROOT
from estimation.stats import angdiff

ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "fig_data"


def _load(name):
    p = ANALYSIS / name
    return json.loads(p.read_text()) if p.exists() else None


def _write(df: pd.DataFrame, name: str) -> int:
    """Write a CSV that MATLAB's readtable parses cleanly.

    Booleans must go out as 0/1: pandas writes True/False, which readtable reads as
    character data, and any numeric comparison on the column then errors.
    """
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == bool:
            d[c] = d[c].astype(int)
        elif d[c].dtype == object:
            u = set(map(str, pd.unique(d[c].dropna())))
            if u and u <= {"True", "False"}:
                d[c] = (d[c].astype(str) == "True").astype(int)
    d.to_csv(OUT / name, index=False)
    return len(d)


def design_coverage():
    """Figure 1: what the campaign actually covers in (magnitude, angle, config)."""
    f = pd.read_csv(ANALYSIS / "features.csv")
    c = f[f.labeled == 1].drop_duplicates("condition_id")
    rows = []
    for _, r in c.iterrows():
        for k in (1, 2):
            rows.append(dict(condition_id=r.condition_id, block=r.block,
                             config=r.config, disk=k,
                             U_gmm=float(r[f"U{k}_mag"]),
                             angle_deg=(np.nan if pd.isna(r[f"U{k}_ang"])
                                        else float(r[f"U{k}_ang"])),
                             loaded=int(float(r[f"U{k}_mag"]) > 1e-6)))
    d = pd.DataFrame(rows)
    _write(d, "design_coverage.csv")
    # angular histogram at the phase-relevant magnitudes, which is the design gap
    hi = d[(d.loaded == 1) & (d.U_gmm >= 24)]
    _write(hi.groupby(hi.angle_deg.round(1)).size()
             .rename("n_conditions").reset_index(), "design_angles_highU.csv")
    return len(d)


def factor_matrix():
    """Figure 4: method performance by evaluation protocol."""
    b = _load("benchmark.json")
    if not b or "factor_matrix" not in b:
        return 0
    fm = b["factor_matrix"]["methods"]
    rows = []
    for m in ("hybrid", "ml"):
        for proto, s in (fm.get(m) or {}).items():
            if s:
                rows.append(dict(method=m, protocol=proto, r2_pred=s["r2_pred"],
                                 phase_deg=s["phase_hi"], slope=s["slope"],
                                 crosstalk=s["crosstalk"],
                                 localization=s["localization"]))
    for proto, s in (fm.get("physics") or {}).items():
        if s:
            rows.append(dict(method="physics", protocol=proto,
                             r2_pred=s["r2_pred"], phase_deg=s["phase_hi"],
                             slope=s["slope"], crosstalk=s["crosstalk"],
                             localization=s["localization"]))
    g = _load("pinn_gate.json")
    if g:
        for v, protos in (g.get("variants") or {}).items():
            for proto, s in protos.items():
                if s:
                    rows.append(dict(method=f"pinn_{v}", protocol=proto,
                                     r2_pred=s["r2_pred"], phase_deg=s["phase_hi"],
                                     slope=s["slope"], crosstalk=s["crosstalk"],
                                     localization=s["localization"]))
    _write(pd.DataFrame(rows), "factor_matrix.csv")
    return len(rows)


def representation():
    """Figure 5: Cartesian vs equivariant vs ICM, plus the size-matched control."""
    r = _load("representation_test.json")
    n = 0
    if r:
        rows = [dict(protocol=p, variant=v, r2_pred=s["r2_pred"],
                     phase_deg=s["phase_hi"], slope=s["slope"],
                     crosstalk=s["crosstalk"])
                for p, block in r["protocols"].items() for v, s in block.items()]
        _write(pd.DataFrame(rows), "representation_test.csv")
        n += len(rows)
    a = _load("angle_novelty_control.json")
    if a:
        rows = ([dict(arm="unseen_angle", **{k: x[k] for k in
                      ("tag", "n_train_conditions", "r2_pred", "phase_hi")})
                 for x in a["unseen_angle"]]
                + [dict(arm="size_matched_random", **{k: x[k] for k in
                        ("tag", "n_train_conditions", "r2_pred", "phase_hi")})
                   for x in a["size_matched_random"]])
        _write(pd.DataFrame(rows), "angle_novelty_control.csv")
        n += len(rows)
    return n


def threshold_sweep():
    """Figure 7: localization and detection versus the decision threshold."""
    b = _load("benchmark.json")
    if not b:
        return 0
    rows = []
    for m, blk in b["methods"].items():
        det = blk.get("condition", {}).get("detection")
        if not det:
            continue
        for s in det["sweep"]:
            rows.append(dict(method=m, thresh_gmm=s["thresh"], acc=s["acc"],
                             sens=s["sens"], spec=s["spec"],
                             localization=s["localization"]))
        rows.append(dict(method=m, thresh_gmm=np.nan, acc=det["trivial_acc"],
                         sens=np.nan, spec=np.nan, localization=np.nan))
    _write(pd.DataFrame(rows), "threshold_sweep.csv")
    auc = [dict(method=m, roc_auc=b["methods"][m]["condition"]["detection"]["roc_auc"],
                pr_auc=b["methods"][m]["condition"]["detection"]["pr_auc"],
                trivial_acc=b["methods"][m]["condition"]["detection"]["trivial_acc"])
           for m in b["methods"]]
    _write(pd.DataFrame(auc), "detection_auc.csv")
    return len(rows)


def estimates():
    """Figure 3: condition-level estimated versus true, per method."""
    p = ANALYSIS / "estimates_condition.csv"
    if not p.exists():
        return 0
    d = pd.read_csv(p)
    keep = ["method", "cid", "disk", "n_acq", "true_mag", "est_mag",
            "true_ang", "est_ang", "ang_err", "loaded"]
    _write(d[[c for c in keep if c in d.columns]], "estimates_condition.csv")
    return len(d)


def blind():
    """Figure 8: blind validation at condition level, magnitude and phase."""
    b = _load("benchmark.json")
    if not b or "blind_validation" not in b:
        return 0
    bv = b["blind_validation"]
    rows = []
    for m, pts in bv["points"].items():
        for x in pts:
            phase_valid = bool(x.get("phase_label_valid", True))
            rows.append(dict(method=m, condition_id=x["cid"], disk=x["disk"],
                             true_gmm=x["true"], est_gmm=x["est"],
                             true_ang=x["true_ang"], est_ang=x["est_ang"],
                             ang_err=(np.nan if x["est_ang"] is None or not phase_valid
                                      else abs(float(angdiff(x["est_ang"], x["true_ang"])))),
                             phase_label_valid=int(phase_valid),
                             balanced=int(x["true"] < 1e-6)))
    _write(pd.DataFrame(rows), "blind_points.csv")
    summ = [dict(method=m, **{k: v for k, v in s.items()
                              if not isinstance(v, (dict, list, str))})
            for m, s in bv["methods"].items()]
    _write(pd.DataFrame(summ), "blind_summary.csv")
    return len(rows)


def runup_and_conditioning():
    """Figure 2: run-up Bode, and influence-matrix conditioning at the dwell speeds."""
    n = 0
    f = ROOT / "datasets" / "runup_1x.csv"
    if f.exists():
        d = pd.read_csv(f).sort_values("rpm")
        # The probes are calibrated 200 mV/mil and the stored amplitudes are in
        # mils. The manuscript reports displacement in micrometres, so the figure
        # data is converted here and the axis labels match the text.
        d["amp_1x_um"] = d["amp_1x"] * 25.4
        _write(d, "runup_bode.csv")
        n += len(d)
    b = _load("benchmark.json")
    if b:
        _write(pd.DataFrame([dict(speed=k, cond_A=v)
                             for k, v in b["cond_A"].items()]), "conditioning.csv")
    return n


def pinn_forward_and_modes():
    """Digital-twin figures: held-out response prediction, and accuracy versus modes."""
    n = 0
    v = _load("pinn_validate.json")
    if v and "forward" in v:
        methods = v["forward"].get("methods", {})
        rows = [dict(method=name, **metrics) for name, metrics in methods.items()]
        if not rows:
            rows = [dict(method="operator", **{k: x for k, x in v["forward"].items()
                                                if not isinstance(x, (dict, list))})]
        _write(pd.DataFrame(rows), "pinn_forward_summary.csv")
        n += len(rows)
    if v and v.get("blind", {}).get("methods"):
        rows = []
        for m, s in v["blind"]["methods"].items():
            rows.append(dict(method=m, **{k: x for k, x in s.items()
                                          if not isinstance(x, (dict, list))}))
        _write(pd.DataFrame(rows), "pinn_blind_summary.csv")
        n += len(rows)
    if v and "nested_selection" in v:
        rows = []
        for proto, s in v["nested_selection"].items():
            rows.append(dict(protocol=proto,
                             r2_pred=s.get("r2_pred"), phase_deg=s.get("phase_hi"),
                             slope=s.get("slope"), crosstalk=s.get("crosstalk"),
                             localization=s.get("localization"),
                             chosen=",".join(s.get("chosen_per_fold", [])),
                             stable=int(bool(s.get("selection_stable")))))
        _write(pd.DataFrame(rows), "pinn_nested.csv")
        n += len(rows)
    return n


def main():
    OUT.mkdir(exist_ok=True)
    for fn in (design_coverage, factor_matrix, representation, threshold_sweep,
               estimates, blind, runup_and_conditioning,
               pinn_forward_and_modes):
        n = fn()
        print(f"  {fn.__name__:18s} {n:6d} rows")
    print(f"\nwrote {OUT}")
    print("render with the scripts in matlab/ (Times New Roman, no titles, vector PDF)")


if __name__ == "__main__":
    main()
