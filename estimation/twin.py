"""
Unified digital-twin engine — physics, ML, and the hybrid that improves on both.

Three estimators, one feature table (`analysis/features.csv`), one set of labels:

  PHYSICS  — the classical Influence Coefficient Method. A per-speed influence
             matrix A is built from the two trial runs (1.6 g on each disk) and
             the balanced baseline; every acquisition's added unbalance is the
             least-squares inverse  U = A^+ (V - V0).  Interpretable, but linear,
             per-speed, and limited by the sub-critical 2-plane conditioning.

  ML       — Extra-Trees regression from physics-informed features to the added
             unbalance vector (see estimation/ml.py). One model, both speeds,
             needs no influence matrix at all; can represent rotor anisotropy that
             a single isotropic A cannot.

  HYBRID   — PHYSICS-AUGMENTED ML: the ICM estimate is appended to the feature
             vector as four extra inputs and the learner predicts the FULL complex
             target. It is *not* residual learning — nothing in the code predicts
             an ICM residual and adds it back to the ICM estimate. Describe it as
             physics-augmented, not residual-correcting, unless that changes.

CALIBRATION PARITY — read before comparing the three. The hybrid consumes
`phys_U*` features, which come from the SAME per-speed influence matrix the ICM
inverts, built from the balanced baseline plus one trial run per disk (A007/A015).
So the hybrid does NOT remove the trial-run requirement: it removes the need to
*repeat* the trial runs per deployment, and inherits whatever error the stored A
carries. Only the pure-ML estimator is genuinely calibration-free.

All three are evaluated leak-free (grouped CV) and scored on the same held-out
conditions. The condition -- not the acquisition -- is the independent unit; see
`estimation/stats.py` and `scripts/benchmark.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from processing.features import SENSORS
from estimation.ml import cross_val_estimate, LOAD_THRESH
from estimation.stats import circmean_deg

TRIAL_D1, TRIAL_D2, BASE_ID = "A007", "A015", "BASE"
CALIB = {BASE_ID, TRIAL_D1, TRIAL_D2}   # physics calibration set — never scored


def _resp_matrix(sub: pd.DataFrame) -> np.ndarray:
    """(n, 4) complex baseline-subtracted 1X response for the given rows."""
    return np.column_stack([sub[f"{p}_resp_re"].to_numpy(float)
                            + 1j * sub[f"{p}_resp_im"].to_numpy(float)
                            for p in SENSORS])


def _true_U(sub: pd.DataFrame) -> tuple[complex, complex]:
    """Mean complex added unbalance (disk1, disk2) over rows (for trial masses).

    The angle is averaged CIRCULARLY. An arithmetic mean of degrees is invalid
    across the -180/+180 branch cut; it is harmless for the present trial
    conditions (A007/A015 are both at 0 deg) but silently wrong for any trial set
    that straddles the cut, so it is fixed at the source.
    """
    def cx(mag, ang):
        m = pd.to_numeric(sub[mag], errors="coerce").mean()
        a = circmean_deg(pd.to_numeric(sub[ang], errors="coerce").to_numpy())
        return m * np.exp(1j * np.radians(0.0 if np.isnan(a) else a))
    return cx("U1_mag", "U1_ang"), cx("U2_mag", "U2_ang")


def influence_matrices(df: pd.DataFrame) -> dict:
    """Per-speed influence matrix A (4 sensors x 2 disks) + condition number."""
    out = {}
    for sp in sorted(df["speed_id"].unique()):
        d1 = df[(df.condition_id == TRIAL_D1) & (df.speed_id == sp)]
        d2 = df[(df.condition_id == TRIAL_D2) & (df.speed_id == sp)]
        if d1.empty or d2.empty:
            continue
        r1 = _resp_matrix(d1).mean(axis=0)
        r2 = _resp_matrix(d2).mean(axis=0)
        W1, _ = _true_U(d1)
        _, W2 = _true_U(d2)
        A = np.column_stack([r1 / W1, r2 / W2])
        cond = float(np.linalg.cond(A))
        out[sp] = {"A": A, "cond": cond}
    return out


def physics_cross_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-acquisition ICM estimate for every labeled row, using the fixed
    per-speed influence matrix. Returns predicted magnitude / phase / localization
    aligned to the labeled rows (same schema as ml.cross_val_estimate).
    """
    lab = df[df["labeled"] == 1].reset_index(drop=True)
    mats = influence_matrices(df)
    u1e = np.zeros(len(lab), complex)
    u2e = np.zeros(len(lab), complex)
    for i, r in lab.iterrows():
        A = mats[r["speed_id"]]["A"]
        resp = _resp_matrix(lab.iloc[[i]])[0]
        est, *_ = np.linalg.lstsq(A, resp, rcond=None)
        u1e[i], u2e[i] = est[0], est[1]

    res = lab[["file", "condition_id", "block", "config", "speed_id",
               "U1_mag", "U2_mag", "U1_ang", "U2_ang",
               "disk1_loaded", "disk2_loaded"]].rename(
        columns={c: f"true_{c}" for c in
                 ["U1_mag", "U2_mag", "U1_ang", "U2_ang", "disk1_loaded", "disk2_loaded"]})
    res["est_U1_mag"] = np.abs(u1e); res["est_U2_mag"] = np.abs(u2e)
    res["est_U1_ang"] = np.degrees(np.angle(u1e)); res["est_U2_ang"] = np.degrees(np.angle(u2e))
    res["est_disk1_loaded"] = (np.abs(u1e) >= LOAD_THRESH).astype(int)
    res["est_disk2_loaded"] = (np.abs(u2e) >= LOAD_THRESH).astype(int)
    res["est_std_gmm"] = np.nan
    res["fold"] = -1
    return res


def augment_with_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Append the ICM estimate (re/im per disk) as features -> hybrid input table."""
    mats = influence_matrices(df)
    out = df.copy()
    phys = {"phys_U1_re": [], "phys_U1_im": [], "phys_U2_re": [], "phys_U2_im": []}
    for _, r in df.iterrows():
        A = mats.get(r["speed_id"])
        if A is None:
            for k in phys:
                phys[k].append(0.0)
            continue
        resp = _resp_matrix(pd.DataFrame([r]))[0]
        est, *_ = np.linalg.lstsq(A["A"], resp, rcond=None)
        phys["phys_U1_re"].append(est[0].real); phys["phys_U1_im"].append(est[0].imag)
        phys["phys_U2_re"].append(est[1].real); phys["phys_U2_im"].append(est[1].imag)
    for k, v in phys.items():
        out[k] = v
    return out


def hybrid_cross_estimate(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Leak-free CV of the hybrid (physics-augmented) ML twin."""
    return cross_val_estimate(augment_with_physics(df), **kw)


def estimate(df: pd.DataFrame, method: str = "hybrid") -> pd.DataFrame:
    """Dispatch: 'physics' | 'ml' | 'hybrid' -> per-acquisition estimate table."""
    if method == "physics":
        return physics_cross_estimate(df)
    if method == "ml":
        return cross_val_estimate(df)
    if method == "hybrid":
        return hybrid_cross_estimate(df)
    raise ValueError(method)
