"""
How many modes should the operator carry? Inverse and forward accuracy versus M.

This justifies a structural choice that would otherwise look arbitrary, and it
produces one result that is worth reporting in its own right.

WITH ONE MODE THE INVERSE PROBLEM IS RANK DEFICIENT. A single mode contributes a
rank-one residue, H_sj = phi_s * chi_j, so the columns for the two disks are
proportional and the operator cannot distinguish which plane carries the unbalance.
Forward prediction still works, because the summed response is well represented,
but the inverse collapses. Two modes are the minimum for two-plane separation.
That asymmetry between the forward and inverse directions is a clean illustration
of why a twin must be validated in both.

Modes beyond the measured first bending pair are residual-flexibility terms placed
at geometrically increasing frequencies and only weakly anchored, because two
sub-critical dwell speeds cannot identify them (see `estimation.pinn.pole_priors`).
An earlier version tiled the measured pair, which produced duplicate poles and a
degenerate basis; that was a defect of the setup, not evidence about mode count.

    python -m scripts.pinn_modes   ->  analysis/pinn_modes.json
                                       analysis/fig_data/pinn_modes.csv
Restarts: three random initializations per fit, selected on the TRAINING loss.
An earlier single-initialization run had M=4 collapse while M=3 and M=6 succeeded,
which was a bad local minimum rather than anything structural. Selection uses the
training objective only, so it removes the artifact without touching test data.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from config import ROOT
from estimation.ml import TARGETS, LOAD_THRESH
from estimation.twin import augment_with_physics, CALIB
from estimation.stats import r2_pred
from estimation.pinn import PINNRegressor, pinn_design_matrix, pole_priors
from scripts.pinn_gate import score

ANALYSIS = ROOT / "analysis"
FIGDATA = ANALYSIS / "fig_data"
MODES = (1, 2, 3, 4, 6)


def main():
    df = pd.read_csv(ANALYSIS / "features.csv")
    aug = augment_with_physics(df)
    lab = aug[aug.labeled == 1].reset_index(drop=True)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    X = pinn_design_matrix(lab)
    Y = lab[TARGETS].to_numpy(float)
    cid = lab.condition_id.to_numpy()

    out = {"design": "isotropic operator, condition-grouped 5-fold; forward "
                     "prediction uses the known unbalance as an input",
           "rank_note": "M=1 gives a rank-one residue per disk pair, so the inverse "
                        "cannot separate the two planes while the forward direction "
                        "is unaffected",
           "modes": {}}
    rows = []
    print(f"{'M':>2s} {'par':>4s} | inverse R2pred  slope  phase  xtalk | "
          f"forward R2pred  rel.err(loaded)")
    for M in MODES:
        inv = np.zeros_like(Y)
        fwd = np.full((len(Y), 8), np.nan)
        meas = None
        for tr, te in GroupKFold(n_splits=5).split(X, Y, cid):
            m = PINNRegressor(n_modes=M, anisotropic=False, epochs=1500,
                              lr=0.03, n_restarts=3).fit(X[tr], Y[tr])
            inv[te] = m.predict(X[te])
            fwd[te] = m.predict_response(X[te], Y[te])
            if meas is None:
                meas = m.measured_response(X)
        s = score(lab, scored, inv, "")
        k = scored & np.isfinite(fwd).all(axis=1)
        Vp, Vm = fwd[k], meas[k]
        rel = np.linalg.norm(Vp - Vm, axis=1) / np.maximum(np.linalg.norm(Vm, axis=1), 1e-12)
        loaded = (np.abs(Y[k, 0] + 1j * Y[k, 1])
                  + np.abs(Y[k, 2] + 1j * Y[k, 3])) > LOAD_THRESH
        npar = int(14 * M)          # 2 poles + 8 sensor + 4 disk residue reals
        rec = {"n_parameters_active": npar,
               "pole_priors_rpm": np.round(pole_priors(M)[0] / (2 * np.pi / 60), 0).tolist(),
               "inverse": {k2: s[k2] for k2 in
                           ("r2_pred", "slope", "rmse", "phase_hi", "crosstalk",
                            "localization")},
               "forward": {"response_r2_pred": round(float(r2_pred(Vm.ravel(), Vp.ravel())), 4),
                           "rel_error_median_loaded": round(float(np.median(rel[loaded])), 4)}}
        out["modes"][str(M)] = rec
        rows.append(dict(modes=M, n_parameters=npar,
                         inv_r2_pred=s["r2_pred"], inv_slope=s["slope"],
                         inv_phase_deg=s["phase_hi"], inv_crosstalk=s["crosstalk"],
                         fwd_r2_pred=rec["forward"]["response_r2_pred"],
                         fwd_rel_error=rec["forward"]["rel_error_median_loaded"]))
        print(f"{M:2d} {npar:4d} |        {s['r2_pred']:7.3f} {s['slope']:6.2f} "
              f"{s['phase_hi']:6.1f} {s['crosstalk']:6.2f} |        "
              f"{rec['forward']['response_r2_pred']:7.3f} "
              f"{rec['forward']['rel_error_median_loaded']:9.3f}")

    FIGDATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(FIGDATA / "pinn_modes.csv", index=False)
    best = max(out["modes"], key=lambda k: out["modes"][k]["inverse"]["r2_pred"])
    out["chosen"] = {"n_modes": int(best),
                     "reason": "best condition-level inverse predictive R2; M=2 is the "
                               "minimum for two-plane separation and M=3 adds a weakly "
                               "anchored residual-flexibility term"}
    (ANALYSIS / "pinn_modes.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANALYSIS/'pinn_modes.json'} and {FIGDATA/'pinn_modes.csv'}")
    print(f"  best inverse R2pred at M={best}")


if __name__ == "__main__":
    main()
