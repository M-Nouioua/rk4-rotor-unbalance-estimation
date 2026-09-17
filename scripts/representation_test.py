"""
Representation test — is the angle collapse a CAPACITY defect or a REPRESENTATION defect?

This is the controlled experiment behind the paper's central mechanistic claim.
Model capacity, hyperparameters and evaluation protocol are held fixed (the same
Extra-Trees and the same folds). The input-output representation changes:

  A) CARTESIAN  (the deployed design) — regress (U_re, U_im) per disk from the full
     feature table. The targets live in the lab frame, so a rotation of the true
     unbalance is a different target vector. A tree ensemble does not encode
     rotational equivariance, so transfer to an unseen angle depends on the sampled
     angular coverage.

  B) EQUIVARIANT — regress |U| per disk from rotation-INVARIANT features only, and
     regress the phase as an OFFSET from the ICM prior's phase, as (cos, sin) to
     avoid wrap-around. Reconstruct U = |U| * exp(i(arg(U_ICM) + offset)). A global
     rotation now leaves every input and every target unchanged, so equivariance is
     structural rather than learned, and angle generalization is not a capacity
     question at all.

  C) ICM prior alone — the physics reference, protocol-invariant.

Reported at condition level with predictive R^2 (see estimation/stats.py).

    python -m scripts.representation_test        -> analysis/representation_test.json

WHAT THIS DOES AND DOES NOT SHOW. It shows the catastrophic angle failure is
removable by reparameterization (R2pred -1.12 -> +0.34, phase 67 -> 28 deg), which
is why a physics-informed / equivariant estimator is the right next model rather
than a bigger black box. It does NOT show the equivariant form beats the physics
off-grid -- it does not, yet. Closing that gap is what the PINN and the dense-angle
campaign are for. Report both halves.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from config import ROOT
from estimation.ml import build_regressor, feature_columns, TARGETS
from estimation.twin import augment_with_physics, CALIB
from estimation.stats import aggregate_conditions, r2_pred, r2_line, angdiff
from processing.features import SENSORS

ANALYSIS = ROOT / "analysis"
HIGH_U = 24.0

# Feature suffixes that are invariant under a global rotation of the unbalance.
# Amplitudes, amplitude ratios, harmonic ratios and orbit-ellipse SHAPE are
# invariant; anything carrying an absolute lab-frame phase or orientation is not
# (note `_tilt_deg` is deliberately EXCLUDED -- the ellipse orientation rotates
# with the unbalance, so it is equivariant, not invariant).
INVARIANT_SUFFIXES = ("_1x_amp", "_2x_amp", "_3x_amp", "_2x_ratio", "_3x_ratio",
                      "_resp_amp", "_smajor", "_sminor", "_ellipticity", "_whirl")
INVARIANT_EXACT = ("rpm", "plane_amp_ratio", "plane_phase_diff")


def build_frames(df: pd.DataFrame):
    aug = augment_with_physics(df)
    lab = aug[aug.labeled == 1].reset_index(drop=True)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()

    pU1 = lab.phys_U1_re.to_numpy(float) + 1j * lab.phys_U1_im.to_numpy(float)
    pU2 = lab.phys_U2_re.to_numpy(float) + 1j * lab.phys_U2_im.to_numpy(float)
    ref = np.angle(pU1)                     # reference phase = ICM disk-1 angle

    U1 = lab.U1_mag.to_numpy(float) * np.exp(1j * np.radians(lab.U1_ang.fillna(0).to_numpy(float)))
    U2 = lab.U2_mag.to_numpy(float) * np.exp(1j * np.radians(lab.U2_ang.fillna(0).to_numpy(float)))

    # ---- A) Cartesian: the deployed design -------------------------------- #
    Xc = lab[feature_columns(lab)].to_numpy(float)
    Yc = lab[TARGETS].to_numpy(float)

    # ---- B) Equivariant --------------------------------------------------- #
    names, cols = [], []
    for c in lab.columns:
        if c.endswith(INVARIANT_SUFFIXES) and pd.api.types.is_numeric_dtype(lab[c]):
            names.append(c); cols.append(lab[c].to_numpy(float))
    for c in INVARIANT_EXACT:
        names.append(c); cols.append(lab[c].to_numpy(float))
    # invariant physics scalars: prior magnitudes and the INTER-disk phase difference
    names += ["phys_U1_absmag", "phys_U2_absmag", "phys_dphi_cos", "phys_dphi_sin"]
    cols += [np.abs(pU1), np.abs(pU2),
             np.cos(np.angle(pU2) - ref), np.sin(np.angle(pU2) - ref)]
    # each sensor's response phase RELATIVE to the reference -> invariant
    for p in SENSORS:
        v = lab[f"{p}_resp_re"].to_numpy(float) + 1j * lab[f"{p}_resp_im"].to_numpy(float)
        d = np.angle(v) - ref
        names += [f"{p}_relphase_cos", f"{p}_relphase_sin", f"{p}_absamp"]
        cols += [np.cos(d), np.sin(d), np.abs(v)]
    Xe = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)

    d1 = np.angle(U1) - ref
    d2 = np.angle(U2) - ref
    Ye = np.column_stack([np.abs(U1), np.cos(d1), np.sin(d1),
                          np.abs(U2), np.cos(d2), np.sin(d2)])

    return lab, scored, ref, (U1, U2), (pU1, pU2), (Xc, Yc), (Xe, Ye), names


def protocols(lab: pd.DataFrame) -> dict:
    ang = np.where(lab.U1_mag.to_numpy() >= lab.U2_mag.to_numpy(),
                   lab.U1_ang.fillna(0).to_numpy(), lab.U2_ang.fillna(0).to_numpy())
    return {"condition": lab.condition_id.to_numpy(),
            "angle_sector": np.round(ang / 45).astype(int) % 8,
            "configuration": lab.config.astype(str).to_numpy()}


def cv(X, Y, groups):
    g = np.asarray(groups)
    sp = GroupKFold(n_splits=5) if len(np.unique(g)) > 5 else LeaveOneGroupOut()
    P = np.zeros_like(Y)
    for tr, te in sp.split(X, Y, g):
        P[te] = build_regressor("et").fit(X[tr], Y[tr]).predict(X[te])
    return P


def score(lab, scored, u1h, u2h, truth) -> dict:
    U1, U2 = truth
    cid = lab.condition_id.to_numpy()[scored]
    P = pd.concat([
        pd.DataFrame(dict(cid=cid, disk=1,
                          true_mag=np.abs(U1)[scored], est_mag=np.abs(u1h)[scored],
                          true_ang=np.degrees(np.angle(U1))[scored],
                          est_ang=np.degrees(np.angle(u1h))[scored])),
        pd.DataFrame(dict(cid=cid, disk=2,
                          true_mag=np.abs(U2)[scored], est_mag=np.abs(u2h)[scored],
                          true_ang=np.degrees(np.angle(U2))[scored],
                          est_ang=np.degrees(np.angle(u2h))[scored]))],
        ignore_index=True)
    P["ang_err"] = np.abs(angdiff(P.est_ang, P.true_ang))
    P["loaded"] = P.true_mag > 1e-6
    C = aggregate_conditions(P)
    L = C[C.loaded]
    hi = L[L.true_mag >= HIGH_U]
    return {"r2_pred": round(r2_pred(L.true_mag, L.est_mag), 4),
            "r2_line": round(r2_line(L.true_mag, L.est_mag), 4),
            "slope": round(float(np.polyfit(L.true_mag, L.est_mag, 1)[0]), 4),
            "rmse": round(float(np.sqrt(np.mean((L.est_mag - L.true_mag) ** 2))), 3),
            "phase_hi": round(float(hi.ang_err.mean()), 2),
            "phase_hi_n": int(len(hi)),
            "crosstalk": round(float(C[~C.loaded].est_mag.mean()), 3),
            "n_points": int(len(C))}


def main():
    df = pd.read_csv(ANALYSIS / "features.csv")
    lab, scored, ref, truth, prior, (Xc, Yc), (Xe, Ye), names = build_frames(df)
    pU1, pU2 = prior

    out = {"question": "Is the angle collapse a capacity defect or a representation "
                       "defect? Model capacity, folds and hyperparameters are held "
                       "fixed; the input-output representation changes.",
           "n_invariant_features": len(names),
           "invariant_features": names,
           "note_tilt_excluded": "plane*_tilt_deg is equivariant, not invariant, so it "
                                 "is excluded from the invariant feature set.",
           "protocols": {}}

    print(f"invariant features: {len(names)}  (Cartesian baseline uses "
          f"{Xc.shape[1]})\n")
    for pname, g in protocols(lab).items():
        print(f"=== held out: {pname} ===")
        Pc = cv(Xc, Yc, g)
        a = score(lab, scored, Pc[:, 0] + 1j * Pc[:, 1], Pc[:, 2] + 1j * Pc[:, 3], truth)
        Pe = cv(Xe, Ye, g)
        m1 = np.clip(Pe[:, 0], 0, None); a1 = ref + np.arctan2(Pe[:, 2], Pe[:, 1])
        m2 = np.clip(Pe[:, 3], 0, None); a2 = ref + np.arctan2(Pe[:, 5], Pe[:, 4])
        b = score(lab, scored, m1 * np.exp(1j * a1), m2 * np.exp(1j * a2), truth)
        c = score(lab, scored, pU1, pU2, truth)
        out["protocols"][pname] = {"cartesian": a, "equivariant": b, "icm_prior": c}
        for tag, s in (("A cartesian (deployed)", a), ("B equivariant", b),
                       ("C ICM prior", c)):
            print(f"  {tag:24s} R2pred={s['r2_pred']:7.3f} slope={s['slope']:5.2f} "
                  f"phase={s['phase_hi']:6.1f}deg xtalk={s['crosstalk']:5.2f}")
        print()

    ang_c = out["protocols"]["angle_sector"]["cartesian"]
    ang_e = out["protocols"]["angle_sector"]["equivariant"]
    ang_p = out["protocols"]["angle_sector"]["icm_prior"]
    out["conclusion"] = {
        "representation_defect_confirmed": bool(
            ang_e["r2_pred"] > ang_c["r2_pred"] + 0.5
            and ang_e["phase_hi"] < ang_c["phase_hi"] - 20),
        "equivariant_beats_physics_off_grid": bool(
            ang_e["r2_pred"] > ang_p["r2_pred"] and ang_e["phase_hi"] < ang_p["phase_hi"]),
        "reading": "A rotation-aligned input-output representation removes the "
                   "collapse under angle holdout without increasing model capacity. "
                   "Because both features and targets change, the result does not "
                   "isolate either component. The representation still does not "
                   "beat the influence coefficient method off-grid.",
    }
    (ANALYSIS / "representation_test.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {ANALYSIS / 'representation_test.json'}")
    print(f"  representation defect confirmed: "
          f"{out['conclusion']['representation_defect_confirmed']}")
    print(f"  equivariant beats physics off-grid: "
          f"{out['conclusion']['equivariant_beats_physics_off_grid']}")


if __name__ == "__main__":
    main()
