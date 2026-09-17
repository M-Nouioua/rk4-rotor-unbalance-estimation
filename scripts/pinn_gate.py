"""
Go / no-go gate for the physics-informed inverse network.

Scores the PINN under the SAME six factor-aware protocols as the baselines, so the
comparison is apples to apples and the paper does not repeat the evaluation error it
diagnoses. Variants:

    iso            isotropic operator (Hb == 0), exactly equivariant
    aniso          + backward-whirl residues (physical anisotropy)
    aniso+res      + small equivariant residual head on invariant features
    aniso+unlab    + physics residual on the 51 unlabeled blind responses
                   (no blind labels are used; reported separately because it uses
                   more data than the baselines do)

Gate criterion, from RESEARCH_PAPER_PLAN.md section 6.7, judged on the angle-sector
protocol, which is where every learned baseline collapses.

    python -m scripts.pinn_gate                  # full gate
    python -m scripts.pinn_gate --quick          # condition + angle only
    python -m scripts.pinn_gate --identifiability  # + parameter identifiability

Outputs
    analysis/pinn_gate.json          all protocol scores, per variant
    analysis/fig_data/pinn_gate.csv  plot-ready long table for MATLAB figures

CALIBRATION PARITY, WHICH THE MANUSCRIPT MUST STATE PLAINLY. The PINN identifies its
operator from the labeled training fold, roughly 68 conditions. The ICM identifies
its influence matrix from three runs (one balanced baseline plus one trial per disk).
They are therefore not interchangeable in a field-balancing workflow, and the PINN's
advantage over the ICM partly reflects a much larger calibration budget. The like for
like comparison is against the Extra-Trees baselines, which consume the same labeled
campaign.
"""
from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd

from config import ROOT
from estimation.ml import TARGETS
from estimation.twin import augment_with_physics, CALIB
from estimation.stats import aggregate_conditions, r2_pred, r2_line, angdiff
from estimation.pinn import PINNRegressor, pinn_design_matrix
from scripts.benchmark import _protocols, _cv_predict
from scripts.representation_test import INVARIANT_SUFFIXES, INVARIANT_EXACT

ANALYSIS = ROOT / "analysis"
FIGDATA = ANALYSIS / "fig_data"
HIGH_U = 24.0
QUICK = ("condition", "angle_sector")
# Mode count selected by scripts/pinn_modes.py. Override with --modes N.
MODES = 3
for _a in sys.argv:
    if _a.startswith("--modes"):
        MODES = int(_a.split("=")[1]) if "=" in _a else MODES


def invariant_columns(lab: pd.DataFrame) -> list[str]:
    cols = [c for c in lab.columns
            if c.endswith(INVARIANT_SUFFIXES) and pd.api.types.is_numeric_dtype(lab[c])]
    return cols + [c for c in INVARIANT_EXACT if c in lab.columns]


def score(lab, scored, pred, tag) -> dict:
    """Condition-level scores for a (n, 4) component prediction."""
    keep = scored & np.isfinite(pred).all(axis=1)
    if keep.sum() < 10:
        return {}
    rows = []
    for k in (1, 2):
        u = pred[:, 2 * (k - 1)] + 1j * pred[:, 2 * (k - 1) + 1]
        ut = (lab[f"U{k}_mag"].to_numpy()
              * np.exp(1j * np.radians(lab[f"U{k}_ang"].fillna(0).to_numpy())))
        rows.append(pd.DataFrame(dict(
            cid=lab.condition_id.to_numpy()[keep], disk=k,
            true_mag=np.abs(ut[keep]), est_mag=np.abs(u[keep]),
            true_ang=np.degrees(np.angle(ut[keep])),
            est_ang=np.degrees(np.angle(u[keep])))))
    P = pd.concat(rows, ignore_index=True)
    P["ang_err"] = np.abs(angdiff(P.est_ang, P.true_ang))
    P["loaded"] = P.true_mag > 1e-6
    C = aggregate_conditions(P)
    L = C[C.loaded]
    hi = L[L.true_mag >= HIGH_U]
    est = (C.est_mag > 6.0).astype(int).to_numpy()
    tru = (C.true_mag > 6.0).astype(int).to_numpy()
    ok = pd.DataFrame({"cid": C.cid, "hit": est == tru}).groupby("cid").hit.all()
    return {"r2_pred": round(float(r2_pred(L.true_mag, L.est_mag)), 4),
            "r2_line": round(float(r2_line(L.true_mag, L.est_mag)), 4),
            "slope": round(float(np.polyfit(L.true_mag, L.est_mag, 1)[0]), 4),
            "rmse": round(float(np.sqrt(np.mean((L.est_mag - L.true_mag) ** 2))), 3),
            "mae": round(float(np.mean(np.abs(L.est_mag - L.true_mag))), 3),
            "phase_hi": round(float(hi.ang_err.mean()), 2) if len(hi) else None,
            "phase_hi_n": int(len(hi)),
            "crosstalk": round(float(C[~C.loaded].est_mag.mean()), 3),
            "localization": round(float(ok.mean()), 4),
            "n_points": int(len(C))}


def main():
    quick = "--quick" in sys.argv
    do_ident = "--identifiability" in sys.argv
    df = pd.read_csv(ANALYSIS / "features.csv")
    aug = augment_with_physics(df)
    lab = aug[aug.labeled == 1].reset_index(drop=True)
    unlab = aug[(aug.labeled == 0) & (aug.block == "blind")]
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    Y = lab[TARGETS].to_numpy(float)
    inv_cols = invariant_columns(lab)

    X_plain = pinn_design_matrix(lab)
    X_inv = pinn_design_matrix(lab, inv_cols)
    Xu_plain = pinn_design_matrix(unlab) if len(unlab) else None

    variants = {
        "iso":         dict(X=X_plain, kw=dict(anisotropic=False, n_invariant=0)),
        "aniso":       dict(X=X_plain, kw=dict(anisotropic=True, n_invariant=0)),
        "aniso+res":   dict(X=X_inv,   kw=dict(anisotropic=True, residual=True,
                                               n_invariant=len(inv_cols))),
        "aniso+unlab": dict(X=X_plain, kw=dict(anisotropic=True, n_invariant=0,
                                               extra_unlabeled=Xu_plain)),
    }

    specs = _protocols(lab)
    if quick:
        specs = {k: v for k, v in specs.items() if k in QUICK}

    out = {"gate": "RESEARCH_PAPER_PLAN.md section 6.7, judged on angle_sector",
           "calibration_parity_note":
               "The PINN identifies its operator from the labeled training fold "
               "(~68 conditions); the ICM uses three runs. The like-for-like "
               "comparison is against the Extra-Trees baselines, which consume the "
               "same campaign.",
           "n_invariant_features": len(inv_cols),
           "protocols": {k: v["note"] for k, v in specs.items()},
           "variants": {}, "identifiability": {}}

    rows = []
    for vname, v in variants.items():
        out["variants"][vname] = {}
        print(f"\n=== PINN variant: {vname} ===")
        for pname, spec in specs.items():
            def make(kind="et", _kw=v["kw"], _M=MODES):
                return PINNRegressor(n_modes=_M, epochs=1500, lr=0.03,
                                     n_restarts=3, **_kw)
            pred = _cv_predict(v["X"], Y, spec, kind_model="et", factory=make)
            s = score(lab, scored, pred, f"{vname}/{pname}")
            out["variants"][vname][pname] = s
            if s:
                print(f"  {pname:26s} R2pred={s['r2_pred']:7.3f} slope={s['slope']:5.2f} "
                      f"phase={s['phase_hi']:6.1f}deg xtalk={s['crosstalk']:5.2f} "
                      f"loc={s['localization']*100:5.1f}%")
                rows.append(dict(method=f"pinn_{vname}", protocol=pname, **s))
        if do_ident:
            m = make().fit(v["X"][scored], Y[scored])
            out["identifiability"][vname] = {
                "parameters": m.n_parameters(),
                "poles": m.identified_poles(),
                "anisotropy_strength": round(m.anisotropy_strength(), 4),
                "final_losses": {k: round(x, 6) for k, x in m.final_.items()}}

    # ---- baselines from benchmark.json, for the same table ----
    bench = json.loads((ANALYSIS / "benchmark.json").read_text())
    fm = bench.get("factor_matrix", {}).get("methods", {})
    for mname in ("hybrid", "ml"):
        for pname, s in (fm.get(mname) or {}).items():
            if s:
                rows.append(dict(method=mname, protocol=pname, **s))
    pi = (fm.get("physics") or {}).get("protocol_invariant")
    if pi:
        for pname in specs:
            rows.append(dict(method="physics", protocol=pname,
                             **{k: v for k, v in pi.items() if k != "note"}))

    FIGDATA.mkdir(exist_ok=True)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(FIGDATA / "pinn_gate.csv", index=False)

    # ---- verdict ----
    icm = pi or {}
    best = max(("iso", "aniso", "aniso+res"),
               key=lambda v: (out["variants"][v].get("angle_sector") or {}).get("r2_pred", -9))
    a = out["variants"][best].get("angle_sector") or {}
    c = out["variants"][best].get("condition") or {}
    hyb_c = (fm.get("hybrid") or {}).get("condition") or {}
    out["verdict"] = {
        "best_variant_on_angle": best,
        "angle_sector": a, "condition": c,
        "icm_reference": {k: icm.get(k) for k in ("r2_pred", "phase_hi", "slope")},
        "beats_icm_off_grid": bool(
            a.get("r2_pred", -9) > icm.get("r2_pred", 9)
            and (a.get("phase_hi") or 1e9) < (icm.get("phase_hi") or 0)),
        "beats_hybrid_in_grid": bool(
            c.get("r2_pred", -9) > hyb_c.get("r2_pred", 9)),
    }
    v = out["verdict"]
    print("\n=== GATE ===")
    print(f"  best variant on angle_sector: {best}")
    print(f"    angle : R2pred={a.get('r2_pred')}  phase={a.get('phase_hi')}")
    print(f"    ICM   : R2pred={icm.get('r2_pred')}  phase={icm.get('phase_hi')}")
    print(f"  beats ICM off-grid    : {v['beats_icm_off_grid']}")
    print(f"  beats hybrid in-grid  : {v['beats_hybrid_in_grid']}")
    (ANALYSIS / "pinn_gate.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANALYSIS/'pinn_gate.json'} and {FIGDATA/'pinn_gate.csv'}")


if __name__ == "__main__":
    main()
