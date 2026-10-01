"""
Head-to-head benchmark of the three twin estimators.

Runs PHYSICS (ICM), ML (Extra-Trees) and HYBRID (physics-augmented Extra-Trees)
on the cached feature table with identical, leak-free evaluation, scores them on
the same held-out conditions, fits the deployable models on all labeled data and
predicts the blind set.

WHAT CHANGED (see RESEARCH_PAPER_PLAN.md "Required corrections"):

  * THE INDEPENDENT UNIT IS THE CONDITION, NOT THE ACQUISITION. Every metric is
    now reported at condition level (repeats collapsed by complex averaging) in
    `methods.<m>.condition`. The old per-acquisition block is retained verbatim
    under `methods.<m>.overall` so the existing figures/dashboard keep working,
    but it must NOT be quoted in the manuscript: its n is inflated ~6x
    (510 records from 85 physical conditions), which makes every CI ~2.4x too
    narrow.

  * TWO R-SQUARED, BOTH NAMED. `r2_line` is the fitted-line score (== squared
    Pearson r) the earlier reports called "R2"; it is a LINEARITY score, blind to
    shrinkage. `r2_pred` = 1-SSE/SST is the PREDICTIVE score and is the one to
    headline. `r2` is kept as an alias of `r2_line` for the old consumers.

  * FACTOR-AWARE VALIDATION MATRIX (`factor_matrix`). Condition-wise CV measures
    interpolation to unseen mass/angle COMBINATIONS, not generalization to unseen
    physical factors. Six protocols are scored: a random-acquisition leaky control,
    condition, magnitude (interpolating and extrapolating -- different questions),
    configuration, and angle sector.

  * THRESHOLD-FREE DETECTION (`detection`). The 6 g.mm decision threshold is
    arbitrary and the localization ranking inverts with it, so ROC-AUC, PR-AUC and
    a full threshold sweep are reported alongside the fixed-threshold confusion.

  * PAIRED CONDITION-LEVEL UNCERTAINTY (`paired`). Cluster bootstrap over
    conditions for each method difference, so "better" is stated with a CI.

  * BLIND PHASE FOR EVERY METHOD, AVERAGED CIRCULARLY. Physics and ML previously
    exported magnitude only, and repeats were averaged arithmetically in degrees.

Outputs:
    analysis/benchmark.json          all metrics (condition-level + legacy)
    analysis/estimates_long.csv      per-acquisition disk-point predictions
    analysis/estimates_condition.csv condition-level disk-points (the paper's unit)
    analysis/blind_predictions.csv   magnitude AND phase, all three methods
    analysis/feature_importance.csv  hybrid feature ranking
    analysis/RESULTS.md              generated result tables (cite these, not prose)

    python -m scripts.benchmark              # full run
    python -m scripts.benchmark --quick      # skip the factor matrix (fast)
"""
from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from config import ROOT
from estimation.twin import (physics_cross_estimate, hybrid_cross_estimate,
                             influence_matrices, augment_with_physics, CALIB)
from estimation.ml import (cross_val_estimate, UnbalanceML, LOAD_THRESH,
                           build_regressor, feature_columns, TARGETS)
from estimation.stats import (circmean_deg, angdiff, r2_line, r2_pred, wilson,
                              aggregate_conditions, paired_bootstrap,
                              cluster_bootstrap_proportion, roc_auc, pr_auc)

ANALYSIS = ROOT / "analysis"
HIGH_U = 24.0    # g.mm, "well-resolved" regime for phase claims
FIXTURE_STEP_DEG = 22.5
METHODS = ("physics", "ml", "hybrid")


def angerr(a, b):
    return angdiff(a, b)


def _fixture_angle_valid(angle_deg: float, tol: float = 1e-6) -> bool:
    """Whether a reference angle is reachable on the 22.5-degree hole grid."""
    nearest = round(float(angle_deg) / FIXTURE_STEP_DEG) * FIXTURE_STEP_DEG
    return abs(float(angdiff(float(angle_deg), nearest))) <= tol


def load(name):
    import csv
    with open(ROOT / "datasets" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wilson(k, n, z=1.96):
    return wilson(k, n, z)


def _points(res: pd.DataFrame) -> pd.DataFrame:
    """Explode a per-acquisition estimate table to one row per (disk-point)."""
    recs = []
    for _, x in res.iterrows():
        for k in (1, 2):
            tmag = x[f"true_U{k}_mag"]
            recs.append(dict(
                cid=x["condition_id"], speed=x["speed_id"], disk=k,
                true_mag=tmag, est_mag=x[f"est_U{k}_mag"],
                true_ang=x[f"true_U{k}_ang"], est_ang=x[f"est_U{k}_ang"],
                ang_err=abs(angerr(x[f"est_U{k}_ang"], x[f"true_U{k}_ang"])),
                loaded=tmag > 1e-6,
                # Recompute both calls from the unrounded magnitudes.  Older
                # feature caches used a strict >6 rule, whereas the withheld-set
                # code used >=6.  A value on the declared decision boundary must
                # have one meaning everywhere.
                true_load=int(tmag >= LOAD_THRESH),
                est_load=int(x[f"est_U{k}_mag"] >= LOAD_THRESH),
                std=x.get("est_std_gmm", np.nan)))
    return pd.DataFrame(recs)


def _reg(t, e):
    """Regression block. Reports BOTH R-squared flavours, both named."""
    t, e = np.asarray(t, float), np.asarray(e, float)
    slope, inter = np.polyfit(t, e, 1)
    rl, rp = r2_line(t, e), r2_pred(t, e)
    return dict(slope=float(slope), intercept=float(inter),
                r2=rl,            # legacy alias == r2_line; do not cite as accuracy
                r2_line=rl, r2_pred=rp,
                rmse=float(np.sqrt(np.mean((e - t) ** 2))),
                mae=float(np.mean(np.abs(e - t))), n=int(len(t)))


def _block(sub: pd.DataFrame) -> dict:
    """Magnitude/phase/cross-talk for one slice of a disk-point table."""
    L = sub[sub.loaded]
    m = _reg(L.true_mag, L.est_mag)
    m["phase_all"] = float(L.ang_err.mean())
    hi = L[L.true_mag >= HIGH_U]
    m["phase_hi"] = float(hi.ang_err.mean()) if len(hi) else float("nan")
    m["phase_hi_med"] = float(hi.ang_err.median()) if len(hi) else float("nan")
    m["phase_hi_n"] = int(len(hi))
    U = sub[~sub.loaded]
    m["crosstalk"] = float(U.est_mag.mean()) if len(U) else 0.0
    return m


def _localization(P: pd.DataFrame, thresh: float = LOAD_THRESH) -> tuple[float, int]:
    """Fraction of conditions with BOTH disks' loaded/not calls correct."""
    est = (P.est_mag >= thresh).astype(int)
    tru = (P.true_mag >= LOAD_THRESH).astype(int)
    ok = (pd.DataFrame({"cid": P.cid, "hit": est.values == tru.values})
            .groupby("cid").hit.all())
    return float(ok.mean()), int(len(ok))


def _detection(P: pd.DataFrame) -> dict:
    """
    Threshold-free detection plus the full sweep.

    The fixed 6 g.mm threshold is not neutral: it sits well for the shrunk tree
    outputs and badly for the ICM, whose cross-plane floor is ~12 g.mm. ROC-AUC /
    PR-AUC rank the methods without choosing a threshold; the sweep shows where
    (and whether) the ranking inverts. `trivial_acc` is the always-loaded
    baseline -- no method that fails to beat it is doing useful detection.
    """
    y = (P.true_mag >= LOAD_THRESH).astype(int).to_numpy()
    s = P.est_mag.to_numpy(float)
    sweep = []
    for t in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]:
        yh = (s >= t).astype(int)
        tp = int(((y == 1) & (yh == 1)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
        fp = int(((y == 0) & (yh == 1)).sum()); fn = int(((y == 1) & (yh == 0)).sum())
        loc, _ = _localization(P, t)
        sweep.append({"thresh": t, "acc": round((tp + tn) / len(y), 4),
                      "sens": round(tp / (tp + fn), 4) if tp + fn else None,
                      "spec": round(tn / (tn + fp), 4) if tn + fp else None,
                      "localization": round(loc, 4)})
    return {"roc_auc": round(roc_auc(y, s), 4), "pr_auc": round(pr_auc(y, s), 4),
            "trivial_acc": round(float((y == 1).mean()), 4),
            "prevalence_note": "trivial_acc = accuracy of always predicting 'loaded'",
            "sweep": sweep}


def _metrics(res: pd.DataFrame) -> dict:
    """LEGACY per-acquisition metric bundle. Retained for the existing figures.

    Do not quote in the manuscript: n is inflated ~6x by back-to-back repeats.
    """
    P = _points(res[~res.condition_id.isin(CALIB)])
    out = {"unit": "acquisition (LEGACY -- inflated n, do not cite)",
           "overall": _block(P)}
    for sp in sorted(P.speed.unique()):
        out[sp] = _block(P[P.speed == sp])
    for k in (1, 2):
        out[f"disk{k}"] = _block(P[P.disk == k])
    acc, n = _localization(P)
    out["localization_acc"], out["localization_n"] = acc, n
    tp = int(((P.true_load == 1) & (P.est_load == 1)).sum())
    fp = int(((P.true_load == 0) & (P.est_load == 1)).sum())
    fn = int(((P.true_load == 1) & (P.est_load == 0)).sum())
    tn = int(((P.true_load == 0) & (P.est_load == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out


def _metrics_condition(res: pd.DataFrame) -> dict:
    """PRIMARY metric bundle: one independent row per (condition, disk)."""
    P = _points(res[~res.condition_id.isin(CALIB)])
    C = aggregate_conditions(P)
    out = {"unit": "condition-disk (complex-averaged repeats) -- THE UNIT TO CITE",
           "n_conditions": int(C.cid.nunique()),
           "n_points": int(len(C)),
           "n_acq_collapsed": int(C.n_acq.sum()),
           "overall": _block(C)}
    for k in (1, 2):
        out[f"disk{k}"] = _block(C[C.disk == k])
    acc, n = _localization(C)
    out["localization_acc"], out["localization_n"] = acc, n
    out["localization_wilson95"] = wilson(round(acc * n), n)
    y = (C.true_mag >= LOAD_THRESH).astype(int)
    yh = (C.est_mag >= LOAD_THRESH).astype(int)
    tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["detection"] = _detection(C)
    return out


def _cross_speed(res: pd.DataFrame) -> dict:
    """N1-vs-N2 agreement on the *same* condition (a physical U is speed-independent)."""
    P = _points(res[~res.condition_id.isin(CALIB)])
    P = P[P.loaded]
    piv = P.pivot_table(index=["cid", "disk"], columns="speed", values="est_mag").dropna()
    if {"N1", "N2"}.issubset(piv.columns) and len(piv):
        d = (piv["N1"] - piv["N2"]).to_numpy()
        return {"mean_abs_gmm": float(np.mean(np.abs(d))),
                "rms_gmm": float(np.sqrt(np.mean(d ** 2))), "n": int(len(piv))}
    return {}


# --------------------------------------------------------------------------- #
# paired condition-level uncertainty
# --------------------------------------------------------------------------- #

def _paired(methods: dict) -> dict:
    """
    Cluster-bootstrap paired differences between methods, at condition level.

    Answers the question the manuscript actually needs: is the hybrid's advantage
    separable from zero given only 82 independent scored conditions?
    """
    C = {m: aggregate_conditions(_points(r[~r.condition_id.isin(CALIB)]))
         for m, r in methods.items()}
    key = ["cid", "disk"]
    base = C["physics"][key + ["true_mag", "loaded"]].copy()
    for m in METHODS:
        base = base.merge(C[m][key + ["est_mag", "ang_err"]].rename(
            columns={"est_mag": f"{m}_mag", "ang_err": f"{m}_ang"}), on=key, how="inner")
    L = base[base.loaded]
    hi = L[L.true_mag >= HIGH_U]
    out = {"unit": "condition-disk", "n_points": int(len(base)),
           "n_conditions": int(base.cid.nunique()), "comparisons": {}}
    for a, b in (("hybrid", "physics"), ("ml", "physics"), ("hybrid", "ml")):
        out["comparisons"][f"{a}_minus_{b}"] = {
            # magnitude: absolute error, so NEGATIVE difference = a is better
            "mag_abs_err": paired_bootstrap(
                np.abs(L[f"{a}_mag"] - L.true_mag), np.abs(L[f"{b}_mag"] - L.true_mag), L.cid),
            # phase on well-resolved points: NEGATIVE = a is better
            "phase_err_hi": paired_bootstrap(hi[f"{a}_ang"], hi[f"{b}_ang"], hi.cid),
        }
    return out


# --------------------------------------------------------------------------- #
# factor-aware validation matrix
# --------------------------------------------------------------------------- #

def _protocols(lab: pd.DataFrame) -> dict:
    """The six evaluation protocols, defined before the results are produced."""
    ang = np.where(lab.U1_mag.to_numpy() >= lab.U2_mag.to_numpy(),
                   lab.U1_ang.fillna(0).to_numpy(), lab.U2_ang.fillna(0).to_numpy())
    tot = (lab.U1_mag + lab.U2_mag).to_numpy()
    severity = np.round(tot, 6)
    levels = np.unique(severity)
    # Only levels bracketed by lower and higher training levels answer an
    # interpolation question.  Endpoint levels are deliberately left unscored
    # here and are covered by the separate high-total-unbalance holdout.
    interior_levels = levels[1:-1]
    return {
        "random_acquisition": dict(
            kind="kfold", groups=None,
            note="LEAKY NEGATIVE CONTROL -- repeats of one condition straddle folds. "
                 "Included only to quantify what the field's default split buys."),
        "condition": dict(
            kind="group_kfold", groups=lab.condition_id.to_numpy(), n_splits=5,
            note="Interpolation to unseen mass/angle COMBINATIONS on the sampled grid."),
        "magnitude_interpolating": dict(
            kind="leave_one_group_out", groups=severity,
            eligible_test_groups=interior_levels,
            note="Leave one interior total-unbalance level out at a time. Endpoint "
                 "levels are excluded, so every scored level is bracketed by lower "
                 "and higher levels in training."),
        "magnitude_extrapolating": dict(
            kind="holdout", mask=tot < 36.0,
            note="Train on total U<36 g.mm and test on total U>=36 g.mm. This is a "
                 "high-total-unbalance holdout; loading composition is not matched."),
        "configuration": dict(
            kind="leave_one_group_out", groups=lab.config.astype(str).to_numpy(),
            eligible_test_groups=np.array(["D1", "D2", "inphase", "antiphase"]),
            note="Leave-one-loaded-configuration-out (D1 / D2 / in-phase / "
                 "anti-phase); the baseline-only calibration group is not scored."),
        "angle_sector": dict(
            kind="leave_one_group_out",
            groups=(np.round(ang / 45).astype(int) % 8),
            note="Leave-one-angle-sector-out. Only 4 distinct angles exist at "
                 "U>=24 g.mm, so on-grid phase scores cannot be read as generalization."),
    }


def _protocol_splits(n_samples: int, spec: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the exact train/test folds declared by one protocol.

    Split strategy is explicit in the protocol specification.  This prevents a
    group-count heuristic from silently turning a stated leave-one-group-out test
    into five-fold grouped cross-validation.
    """
    idx = np.arange(n_samples)
    kind = spec["kind"]
    if kind == "holdout":
        tr = np.asarray(spec["mask"], dtype=bool)
        if tr.sum() < 10 or (~tr).sum() < 5:
            return []
        return [(idx[tr], idx[~tr])]
    if kind == "kfold":
        return list(KFold(n_splits=spec.get("n_splits", 5), shuffle=True,
                          random_state=0).split(idx))

    groups = np.asarray(spec["groups"])
    if kind == "group_kfold":
        n_splits = min(int(spec.get("n_splits", 5)), len(np.unique(groups)))
        return list(GroupKFold(n_splits=n_splits).split(idx, groups=groups))
    if kind == "leave_one_group_out":
        folds = list(LeaveOneGroupOut().split(idx, groups=groups))
        eligible = spec.get("eligible_test_groups")
        if eligible is not None:
            eligible = set(np.asarray(eligible).tolist())
            folds = [(tr, te) for tr, te in folds
                     if (groups[te[0]].item() if hasattr(groups[te[0]], "item")
                         else groups[te[0]]) in eligible]
        return folds
    raise ValueError(f"unknown validation protocol kind: {kind}")


def _protocol_test_mask(n_samples: int, spec: dict) -> np.ndarray:
    """Rows evaluated by a protocol, independent of estimator implementation."""
    out = np.zeros(n_samples, dtype=bool)
    for _, te in _protocol_splits(n_samples, spec):
        out[te] = True
    return out


def _cv_predict(X, Y, spec, kind_model="et", factory=None) -> np.ndarray:
    """Out-of-fold (or held-out) predictions under one protocol. NaN where unscored.

    `factory` lets a different estimator family reuse this exact fold logic, so a
    new model is scored by the same protocols rather than by a parallel code path.
    It must return an object with sklearn-style `fit(X, Y)` / `predict(X)`; see
    `estimation.pinn.PINNRegressor` and `scripts/pinn_gate.py`.
    """
    make = factory if factory is not None else (lambda: build_regressor(kind_model))
    pred = np.full_like(Y, np.nan, dtype=float)
    for tr, te in _protocol_splits(len(X), spec):
        p = make().fit(X[tr], Y[tr])
        pred[te] = p.predict(X[te])
    return pred


def _score_pred(lab: pd.DataFrame, pred: np.ndarray, scored: np.ndarray) -> dict:
    """Condition-level scores for a (n,4) component prediction."""
    keep = scored & np.isfinite(pred).all(axis=1)
    if keep.sum() < 10:
        return {}
    recs = []
    for k in (1, 2):
        u = pred[:, 2 * (k - 1)] + 1j * pred[:, 2 * (k - 1) + 1]
        recs.append(pd.DataFrame(dict(
            cid=lab.condition_id.to_numpy()[keep], disk=k,
            true_mag=lab[f"U{k}_mag"].to_numpy()[keep],
            true_ang=lab[f"U{k}_ang"].fillna(0).to_numpy()[keep],
            est_mag=np.abs(u[keep]), est_ang=np.degrees(np.angle(u[keep])))))
    P = pd.concat(recs, ignore_index=True)
    P["ang_err"] = np.abs(angdiff(P.est_ang, P.true_ang))
    P["loaded"] = P.true_mag > 1e-6
    C = aggregate_conditions(P)
    b = _block(C)
    loc, n = _localization(C)
    return {"r2_pred": round(b["r2_pred"], 4), "r2_line": round(b["r2_line"], 4),
            "slope": round(b["slope"], 4), "rmse": round(b["rmse"], 3),
            "mae": round(b["mae"], 3), "phase_hi": round(b["phase_hi"], 2),
            "phase_hi_n": b["phase_hi_n"], "crosstalk": round(b["crosstalk"], 3),
            "localization": round(loc, 4), "localization_n": n,
            "n_points": int(len(C))}


def _factor_matrix(df: pd.DataFrame, physics_res: pd.DataFrame) -> dict:
    """
    Score every estimator on the observations tested by each protocol.

    The ICM coefficients are fixed, but its metrics are not: R2, MAE and phase
    error depend on the evaluated subset.  Its predictions are therefore filtered
    by the same protocol test mask used for the learned estimators.
    """
    aug = augment_with_physics(df)
    out = {"note": "condition-level scores. Protocols are distinct deployment "
                   "questions, not an ordered difficulty ladder.",
           "protocols": {}, "methods": {}}
    for tag, models in (("hybrid", aug), ("ml", df)):
        lab = models[models.labeled == 1].reset_index(drop=True)
        feats = feature_columns(lab)
        X = lab[feats].to_numpy(float)
        Y = lab[TARGETS].to_numpy(float)
        scored = (~lab.condition_id.isin(CALIB)).to_numpy()
        specs = _protocols(lab)
        out["methods"][tag] = {}
        for name, spec in specs.items():
            out["protocols"][name] = spec["note"]
            s = _score_pred(lab, _cv_predict(X, Y, spec), scored)
            out["methods"][tag][name] = s
            if s:
                print(f"    {tag:7s} {name:26s} R2pred={s['r2_pred']:7.3f} "
                      f"slope={s['slope']:5.2f} phase={s['phase_hi']:6.1f}d "
                      f"loc={s['localization']*100:5.1f}%")
    # Fixed ICM predictions, scored separately on each protocol's test rows.
    lab = df[df.labeled == 1].reset_index(drop=True)
    if not np.array_equal(lab.file.to_numpy(), physics_res.file.to_numpy()):
        raise ValueError("physics predictions are not aligned to labeled feature rows")
    components = []
    for k in (1, 2):
        z = (physics_res[f"est_U{k}_mag"].to_numpy(float)
             * np.exp(1j * np.radians(
                 physics_res[f"est_U{k}_ang"].to_numpy(float))))
        components.extend([z.real, z.imag])
    physics_pred = np.column_stack(components)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    out["methods"]["physics"] = {}
    for name, spec in _protocols(lab).items():
        test_rows = _protocol_test_mask(len(lab), spec)
        s = _score_pred(lab, physics_pred, scored & test_rows)
        if s:
            s["note"] = ("Fixed ICM coefficients; metrics evaluated on this "
                         "protocol's test observations.")
        out["methods"]["physics"][name] = s
        if s:
            print(f"    physics {name:26s} R2pred={s['r2_pred']:7.3f} "
                  f"slope={s['slope']:5.2f} phase={s['phase_hi']:6.1f}d "
                  f"loc={s['localization']*100:5.1f}%")
    return out


def _ablation_direct_magnitude(df: pd.DataFrame) -> dict:
    """
    EXPLORATORY ablation -- NOT a confirmatory result.

    The deployed twin regresses the Cartesian components (U_re, U_im), which makes
    magnitude accuracy depend on angular coverage. Predicting |U| directly is an
    obvious alternative, but it was chosen AFTER seeing the angle-holdout failure
    on this same data. It therefore carries selection bias and is reported only as
    an ablation. A confirmatory claim needs nested model selection inside each fold
    or a fresh, untouched blind set (see RESEARCH_PAPER_PLAN.md correction 7).
    """
    aug = augment_with_physics(df)
    lab = aug[aug.labeled == 1].reset_index(drop=True)
    feats = feature_columns(lab)
    X = lab[feats].to_numpy(float)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    Ymag = lab[["U1_mag", "U2_mag"]].to_numpy(float)
    specs = _protocols(lab)
    out = {"status": "EXPLORATORY -- post-hoc target change, selection-biased, "
                     "needs nested validation or a new blind set before any claim",
           "protocols": {}}
    for name in ("condition", "angle_sector"):
        pred = _cv_predict(X, Ymag, specs[name])
        keep = scored & np.isfinite(pred).all(axis=1)
        t = np.concatenate([lab.U1_mag.to_numpy()[keep], lab.U2_mag.to_numpy()[keep]])
        e = np.concatenate([np.clip(pred[keep, 0], 0, None), np.clip(pred[keep, 1], 0, None)])
        L = t > 1e-6
        out["protocols"][name] = {
            "r2_pred_magnitude": round(r2_pred(t[L], e[L]), 4),
            "slope": round(float(np.polyfit(t[L], e[L], 1)[0]), 4),
            "unit": "acquisition-level magnitude only (ablation, not the paper's unit)"}
    return out


# --------------------------------------------------------------------------- #
# blind set
# --------------------------------------------------------------------------- #

def physics_blind(df, blind):
    """ICM COMPLEX estimate for the blind rows (per-speed A).

    Returns complex per disk so phase is exportable for physics too -- previously
    only magnitude was returned, which is why the blind table had no physics angle.
    """
    from estimation.twin import _resp_matrix
    mats = influence_matrices(df)
    out = np.zeros((len(blind), 2), complex)
    for i, (_, r) in enumerate(blind.reset_index(drop=True).iterrows()):
        A = mats.get(r["speed_id"])
        if A is None:
            continue
        resp = _resp_matrix(pd.DataFrame([r]))[0]
        est, *_ = np.linalg.lstsq(A["A"], resp, rcond=None)
        out[i] = [est[0], est[1]]
    return out


def blind_validation(bp, conds):
    """
    Score every method on all UNSEALED blind conditions (balanced + loaded), using
    ground truth revealed after prediction.

    Repeats of one blind condition are collapsed as a COMPLEX VECTOR (magnitude and
    phase together); the previous arithmetic mean of degrees was invalid across the
    -180/+180 branch cut.
    """
    ids = sorted(set(bp.condition_id) & {k for k, v in conds.items()
                 if v["block"] == "blind" and v["config"] != "sealed"})
    if not ids:
        return None

    def truth(cid, disk):
        c = conds[cid]
        r = float(c["radius_mm"] or 30)
        return float(c[f"disk{disk}_mass_g"] or 0) * r, float(c[f"disk{disk}_angle_deg"] or 0)

    pts = {m: [] for m in METHODS}
    for cid in ids:
        g = bp[bp.condition_id == cid]
        for disk in (1, 2):
            tmag, tang = truth(cid, disk)
            for m in pts:
                mag = g[f"{m}_U{disk}_gmm"].to_numpy(float)
                acol = f"{m}_U{disk}_ang"
                if acol in g:
                    z = np.mean(mag * np.exp(1j * np.radians(g[acol].to_numpy(float))))
                    emag, eang = float(np.abs(z)), float(np.degrees(np.angle(z)))
                else:
                    emag, eang = float(np.mean(mag)), None
                pts[m].append({
                    # Preserve full precision for every metric.  Rounding belongs
                    # only in tables/console formatting; doing it here can move a
                    # prediction across the 6 g.mm decision boundary.
                    "cid": cid, "disk": disk, "true": float(tmag),
                    "est": float(emag), "true_ang": float(tang),
                    "est_ang": None if eang is None else float(eang),
                    "phase_label_valid": _fixture_angle_valid(tang),
                    "n_acq": int(len(g))})

    exclusions = sorted({(p["cid"], p["disk"], p["true_ang"])
                         for p in pts["physics"] if not p["phase_label_valid"]})
    out = {"conditions": ids,
           "unit": "condition-disk (complex-averaged repeats)",
           "n_loaded_cond": sum(1 for c in ids if conds[c]["config"] != "balanced"),
           "n_balanced_cond": sum(1 for c in ids if conds[c]["config"] == "balanced"),
           "n_points": len(pts["physics"]), "points": pts, "methods": {},
           "phase_exclusions": [
               {"condition_id": cid, "disk": disk, "angle_deg": angle,
                "reason": "reference angle is not on the 22.5-degree fixture grid"}
               for cid, disk, angle in exclusions]}
    for m, P in pts.items():
        t = np.array([p["true"] for p in P]); e = np.array([p["est"] for p in P])
        tl, el = t >= LOAD_THRESH, e >= LOAD_THRESH
        tp = int((tl & el).sum()); fp = int((~tl & el).sum())
        fn = int((tl & ~el).sum()); tn = int((~tl & ~el).sum())
        det = cluster_bootstrap_proportion(
            (tl == el).astype(float), [p["cid"] for p in P])
        pe = [abs(angdiff(p["est_ang"], p["true_ang"]))
              for p in P if p["true"] >= HIGH_U and p["est_ang"] is not None
              and p["phase_label_valid"]]
        nl = int(tl.sum()); ns = int((~tl).sum())
        out["methods"][m] = {
            "mae": round(float(np.mean(np.abs(e - t))), 2),
            "rmse": round(float(np.sqrt(np.mean((e - t) ** 2))), 2),
            "mae_loaded": round(float(np.mean(np.abs(e[tl] - t[tl]))), 2) if nl else None,
            "detect_acc": round((tp + tn) / len(P), 4),
            "detect_acc_wilson95": wilson(tp + tn, len(P)),
            "detect_acc_cluster95": det["ci95"],
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "specificity_wilson95": wilson(tn, tn + fp) if (tn + fp) else None,
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "sensitivity_wilson95": wilson(tp, tp + fn) if (tp + fn) else None,
            "roc_auc": round(roc_auc(tl.astype(int), e), 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "phase_err_hi": round(float(np.mean(pe)), 1) if pe else None,
            "phase_err_hi_n": len(pe),
            "n_loaded_points": nl, "n_balanced_points": ns,
            "small_sample_warning": "10 independent conditions / 20 disk-points -- "
                                    "report intervals, never point estimates alone"}
    return out


# --------------------------------------------------------------------------- #
# generated results document (kills doc drift)
# --------------------------------------------------------------------------- #

def write_results_md(bench: dict) -> None:
    """Emit analysis/RESULTS.md from the JSON so prose can never drift again."""
    L = []
    w = L.append
    w("# Generated results — do not edit by hand\n")
    w(f"Regenerate with `python -m scripts.benchmark`. Source of truth: "
      f"`analysis/benchmark.json`.\n")
    w(f"- Labeled acquisitions: **{bench['n_labeled']}** from "
      f"**{bench['n_conditions']}** physical conditions (the independent unit).")
    w(f"- Influence-matrix conditioning: " +
      ", ".join(f"cond(A)@{k}={v}" for k, v in bench["cond_A"].items()))
    w(f"- Detection threshold used for fixed-threshold metrics: "
      f"{bench['load_thresh_gmm']} g·mm; phase reported for U ≥ "
      f"{bench['high_u_gmm']} g·mm.\n")

    w("## 1. Condition-level performance (cite these)\n")
    w("| Method | R²pred | R²line | Slope | RMSE g·mm | MAE | Phase U≥24 | Cross-talk | Localization | ROC-AUC |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m in METHODS:
        c = bench["methods"][m]["condition"]; o = c["overall"]; d = c["detection"]
        ci = c["localization_wilson95"]
        w(f"| {m} | {o['r2_pred']:.3f} | {o['r2_line']:.3f} | {o['slope']:.2f} | "
          f"{o['rmse']:.2f} | {o['mae']:.2f} | {o['phase_hi']:.1f}° | "
          f"{o['crosstalk']:.2f} | {c['localization_acc']*100:.1f}% "
          f"[{ci[0]*100:.0f}–{ci[1]*100:.0f}] | {d['roc_auc']:.3f} |")
    t = bench["methods"]["physics"]["condition"]["detection"]["trivial_acc"]
    w(f"\nAlways-loaded baseline accuracy: **{t*100:.1f}%** — compare every "
      f"fixed-threshold accuracy against it.\n")

    if "paired" in bench:
        w("## 2. Paired condition-level differences (cluster bootstrap, 95% CI)\n")
        w("Negative = the first method is better. An interval containing 0 means "
          "this campaign cannot separate them.\n")
        w("| Comparison | Δ magnitude abs-err | 95% CI | Δ phase (U≥24) | 95% CI |")
        w("|---|--:|---|--:|---|")
        for k, v in bench["paired"]["comparisons"].items():
            a, b = v["mag_abs_err"], v["phase_err_hi"]
            w(f"| {k.replace('_minus_',' − ')} | {a['diff']:+.2f} | "
              f"[{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}]"
              f"{' *' if a['excludes_zero'] else ''} | {b['diff']:+.1f}° | "
              f"[{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}]"
              f"{' *' if b['excludes_zero'] else ''} |")
        w("\n`*` = interval excludes zero.\n")

    if "factor_matrix" in bench:
        fm = bench["factor_matrix"]
        w("## 3. Factor-aware validation matrix\n")
        w(fm["note"] + "\n")
        w("| Protocol | Hybrid R²pred | Hybrid phase | ML R²pred | ML phase |")
        w("|---|--:|--:|--:|--:|")
        for p in fm["protocols"]:
            h = fm["methods"]["hybrid"].get(p) or {}
            m = fm["methods"]["ml"].get(p) or {}
            if not h and not m:
                continue
            w(f"| {p} | {h.get('r2_pred', float('nan')):.3f} | "
              f"{h.get('phase_hi', float('nan')):.1f}° | "
              f"{m.get('r2_pred', float('nan')):.3f} | "
              f"{m.get('phase_hi', float('nan')):.1f}° |")
        for p, s in fm["methods"].get("physics", {}).items():
            if s:
                w(f"| physics / {p} | {s['r2_pred']:.3f} | "
                  f"{s['phase_hi']:.1f}° | — | — |")
        w("\nProtocol definitions:\n")
        for p, note in fm["protocols"].items():
            w(f"- **{p}** — {note}")
        w("")

    if "blind_validation" in bench:
        bv = bench["blind_validation"]
        w("## 4. Blind validation (labels revealed after prediction)\n")
        w(f"{bv['n_loaded_cond']} loaded + {bv['n_balanced_cond']} balanced "
          f"conditions, {bv['n_points']} disk-points. {bv['unit']}.\n")
        w("| Method | MAE | Loaded MAE | Detect acc [95% CI] | Spec | Sens | ROC-AUC | Phase U≥24 |")
        w("|---|--:|--:|---|--:|--:|--:|--:|")
        for m in METHODS:
            s = bv["methods"][m]
            ci = s["detect_acc_cluster95"]
            ph = "—" if s["phase_err_hi"] is None else f"{s['phase_err_hi']:.0f}°"
            w(f"| {m} | {s['mae']:.1f} | {s['mae_loaded']:.1f} | "
              f"{s['detect_acc']*100:.0f}% [{ci[0]*100:.0f}–{ci[1]*100:.0f}] | "
              f"{s['specificity']*100:.0f}% | {s['sensitivity']*100:.0f}% | "
              f"{s['roc_auc']:.3f} | {ph} |")
        w("\nDetection intervals resample the 10 whole conditions, preserving the "
          "two disk outcomes within each condition.\n")

    if "ablation_direct_mag" in bench:
        ab = bench["ablation_direct_mag"]
        w("## 5. Exploratory ablation — direct |U| target\n")
        w(f"**{ab['status']}**\n")
        w("| Protocol | R²pred (magnitude) | Slope |")
        w("|---|--:|--:|")
        for p, v in ab["protocols"].items():
            w(f"| {p} | {v['r2_pred_magnitude']:.3f} | {v['slope']:.2f} |")
        w("")

    w("## Legacy per-acquisition block\n")
    w("`methods.<m>.overall` in the JSON is the old per-acquisition scoring, kept "
      "so the existing figures/dashboard keep running. Its n counts records, not "
      "independent conditions, so it must not be cited.\n")
    (ANALYSIS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main():
    quick = "--quick" in sys.argv
    df = pd.read_csv(ANALYSIS / "features.csv")
    print(f"loaded {len(df)} rows ({(df.labeled==1).sum()} labeled, "
          f"{df[df.labeled==1].condition_id.nunique()} conditions)")

    methods = {
        "physics": physics_cross_estimate(df),
        "ml": cross_val_estimate(df),
        "hybrid": hybrid_cross_estimate(df),
    }
    mats = influence_matrices(df)
    bench = {
        "cond_A": {s: round(v["cond"], 2) for s, v in mats.items()},
        "n_labeled": int((df.labeled == 1).sum()),
        "n_conditions": int(df[df.labeled == 1].condition_id.nunique()),
        "n_mounts_per_condition": int(df[df.labeled == 1]
                                      .groupby("condition_id").mount_idx.nunique().max()),
        "load_thresh_gmm": LOAD_THRESH, "high_u_gmm": HIGH_U,
        "statistical_unit": "condition_id (see methods.<m>.condition). The "
                            "per-acquisition block methods.<m>.overall is legacy.",
        "methods": {},
    }
    if bench["n_mounts_per_condition"] <= 1:
        bench["sigma_mount_status"] = ("UNMEASURED -- every condition has exactly one "
                                       "physical mount, so setup variance is not "
                                       "identifiable and no GUM U95 budget can be built "
                                       "from this campaign. Needs the U_anchor block.")

    long_rows, cond_rows = [], []
    print(f"\n{'method':8s} | slope | R2pred | R2line | RMSE | phase(U>=24) | xtalk | loc%  | AUC")
    print("-" * 92)
    for name, res in methods.items():
        m = _metrics(res)                      # legacy, per acquisition
        m["condition"] = _metrics_condition(res)   # primary, per condition
        m["cross_speed"] = _cross_speed(res)
        bench["methods"][name] = m
        c = m["condition"]; o = c["overall"]
        print(f"{name:8s} | {o['slope']:.3f} | {o['r2_pred']:6.3f} | {o['r2_line']:6.3f} | "
              f"{o['rmse']:.2f} | {o['phase_hi']:11.1f}  | {o['crosstalk']:5.2f} | "
              f"{c['localization_acc']*100:5.1f} | {c['detection']['roc_auc']:.3f}")
        pts = _points(res[~res.condition_id.isin(CALIB)])
        pts.insert(0, "method", name)
        long_rows.append(pts)
        cp = aggregate_conditions(pts)
        cp.insert(0, "method", name)
        cond_rows.append(cp)

    pd.concat(long_rows, ignore_index=True).to_csv(ANALYSIS / "estimates_long.csv", index=False)
    pd.concat(cond_rows, ignore_index=True).to_csv(ANALYSIS / "estimates_condition.csv", index=False)

    print("\npaired condition-level bootstrap (negative = first method better):")
    bench["paired"] = _paired(methods)
    for k, v in bench["paired"]["comparisons"].items():
        a, b = v["mag_abs_err"], v["phase_err_hi"]
        print(f"  {k:18s} mag {a['diff']:+6.2f} g.mm CI[{a['ci95'][0]:+.2f},{a['ci95'][1]:+.2f}]"
              f"{'*' if a['excludes_zero'] else ' '}   "
              f"phase {b['diff']:+6.1f} deg CI[{b['ci95'][0]:+.1f},{b['ci95'][1]:+.1f}]"
              f"{'*' if b['excludes_zero'] else ' '}")

    if not quick:
        print("\nfactor-aware validation matrix (condition-level):")
        bench["factor_matrix"] = _factor_matrix(df, methods["physics"])
        print("\nexploratory ablation (direct |U| target):")
        bench["ablation_direct_mag"] = _ablation_direct_magnitude(df)
        for p, v in bench["ablation_direct_mag"]["protocols"].items():
            print(f"    {p:26s} R2pred(mag)={v['r2_pred_magnitude']:7.3f} "
                  f"slope={v['slope']:.2f}")

    # ---- deployable models -> blind set (physics / ML / hybrid) ----
    blind = df[(df.labeled == 0) & (df.block == "blind")].copy()
    if len(blind):
        aug = augment_with_physics(df)
        hyb_model = UnbalanceML("et").fit(aug)               # physics-augmented
        ml_model = UnbalanceML("et").fit(df)                 # pure ML
        blind_aug = aug[aug.file.isin(blind.file)]
        blind_pl = df[df.file.isin(blind.file)]
        hyb = hyb_model.predict(blind_aug)
        mlp = ml_model.predict(blind_pl)
        phys_blind = physics_blind(df, blind)
        bp = blind[["file", "condition_id", "speed_id", "rpm_measured"]].reset_index(drop=True)
        for k in (1, 2):
            bp[f"hybrid_U{k}_gmm"] = hyb[f"U{k}_mag"]
            bp[f"hybrid_U{k}_ang"] = hyb[f"U{k}_ang"]
            bp[f"ml_U{k}_gmm"] = mlp[f"U{k}_mag"]
            bp[f"ml_U{k}_ang"] = mlp[f"U{k}_ang"]          # was missing
            bp[f"physics_U{k}_gmm"] = np.abs(phys_blind[:, k - 1])
            bp[f"physics_U{k}_ang"] = np.degrees(np.angle(phys_blind[:, k - 1]))  # was missing
        bp.to_csv(ANALYSIS / "blind_predictions.csv", index=False)
        bench["n_blind"] = int(len(blind))
        print(f"\nblind set: {len(blind)} acquisitions predicted "
              f"(physics / ML / hybrid, magnitude + phase)")

        # ---- specificity on the UNSEALED balanced blind cases ----
        conds = {c["condition_id"]: c for c in load("conditions.csv")}
        bal_ids = [k for k, v in conds.items()
                   if v["block"] == "blind" and v["config"] == "balanced"]
        balp = bp[bp.condition_id.isin(bal_ids)]
        if len(balp):
            spec = {}
            for m in METHODS:
                mags = np.concatenate([balp[f"{m}_U1_gmm"].to_numpy(),
                                       balp[f"{m}_U2_gmm"].to_numpy()])
                correct = int((mags < LOAD_THRESH).sum()); n = int(len(mags))
                spec[m] = {"specificity": correct / n, "correct": correct, "n": n,
                           "max_gmm": float(np.max(mags)), "mean_gmm": float(np.mean(mags)),
                           "wilson95": wilson(correct, n),
                           "n_independent_conditions": len(bal_ids)}
            bench["blind_balanced"] = {"conditions": sorted(bal_ids),
                                       "n_conditions": len(bal_ids), "methods": spec}
            print(f"  balanced blind ({len(bal_ids)} cond, {spec['hybrid']['n']} disk-points) "
                  f"specificity <{LOAD_THRESH:g} g.mm:")
            for m in METHODS:
                s = spec[m]
                print(f"    {m:8s}: {s['correct']}/{s['n']} = {s['specificity']*100:.0f}% "
                      f"CI[{s['wilson95'][0]*100:.0f}-{s['wilson95'][1]*100:.0f}]  "
                      f"(max {s['max_gmm']:.1f} g.mm)")

        # ---- full blind validation on every UNSEALED blind condition ----
        bv = blind_validation(bp, conds)
        if bv:
            bench["blind_validation"] = bv
            print(f"\n  blind validation ({bv['n_loaded_cond']} loaded + "
                  f"{bv['n_balanced_cond']} balanced conditions, {bv['n_points']} disk-points):")
            print(f"    {'method':8s}  MAE   RMSE  loadedMAE  detect  spec  sens   AUC   phase")
            for m in METHODS:
                s = bv["methods"][m]
                ph = "  n/a" if s["phase_err_hi"] is None else f"{s['phase_err_hi']:5.0f}"
                print(f"    {m:8s}  {s['mae']:4.1f}  {s['rmse']:4.1f}   {s['mae_loaded']:5.1f}    "
                      f"{s['detect_acc']*100:4.0f}% {s['specificity']*100:4.0f}% "
                      f"{s['sensitivity']*100:4.0f}%  {s['roc_auc']:.3f} {ph}")

    # ---- feature importance (hybrid) ----
    fi = UnbalanceML("et").fit(augment_with_physics(df)).feature_importance()
    fi.to_csv(ANALYSIS / "feature_importance.csv", header=["importance"])
    bench["top_features"] = {k: float(v) for k, v in fi.head(12).items()}

    (ANALYSIS / "benchmark.json").write_text(json.dumps(bench, indent=2))
    write_results_md(bench)
    print(f"\nwrote benchmark.json, RESULTS.md, estimates_long.csv, "
          f"estimates_condition.csv, feature_importance.csv")


if __name__ == "__main__":
    main()
