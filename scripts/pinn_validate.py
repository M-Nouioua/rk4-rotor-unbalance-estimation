"""
Three validations the PINN result needs before it can be written up.

1. NESTED MODEL SELECTION. The isotropic-versus-anisotropic choice in
   `scripts/pinn_gate.py` was made after reading the angle-sector scores on the
   same data the model is evaluated on. That is post-hoc selection, which is the
   objection this project raises against everyone else, so the claim cannot stand
   on it. Here the variant is chosen inside each outer training fold and never
   sees the outer test fold.

   The inner folds group by `condition_id`, not by the outer factor. That is the
   conservative choice: it asks whether a practitioner who selects a model the
   ordinary way still gets one that generalizes across an unseen physical factor.
   Selecting inside folds grouped by the outer factor would be standard nested CV
   and would score better, but it assumes the deployment question is known in
   advance, which is a weaker claim. The recorded `chosen` list shows which variant
   won in each fold, so any instability is visible rather than hidden.

2. WITHHELD-SET SCORING. The operator had not been run on BLND01 to BLND10 when
   it was developed. These labels had already been unsealed on 2026-07-26 (see
   `analysis/blind_freeze.json`), so the result is retrospective evidence rather
   than a confirmatory test. The mixed set includes one (48, 48) g.mm two-plane
   condition and does not isolate magnitude extrapolation from the other factors.

3. FORWARD VALIDATION. A capability available from the explicit forward operator:
   fit the operator, then predict the measured 1X response of a condition it was
   not fitted to, using the known unbalance as an input. The tree baselines cannot
   be tested this way at all, because they only invert. This is the defining
   forward-model claim and had not previously been tested in the project.

    python -m scripts.pinn_validate                 # all three
    python -m scripts.pinn_validate --nested        # one part only
    python -m scripts.pinn_validate --blind
    python -m scripts.pinn_validate --forward

Outputs
    analysis/pinn_validate.json
    analysis/fig_data/pinn_forward.csv
    analysis/fig_data/pinn_blind.csv
"""
from __future__ import annotations

import csv
import json
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from config import ROOT
from estimation.ml import TARGETS, LOAD_THRESH
from estimation.twin import augment_with_physics, influence_matrices, CALIB
from estimation.stats import (aggregate_conditions, r2_pred, angdiff, wilson,
                              cluster_bootstrap_proportion, roc_auc)
from estimation.pinn import PINNRegressor, pinn_design_matrix
from scripts.benchmark import (_protocols, _protocol_splits,
                               _fixture_angle_valid)
from scripts.pinn_gate import invariant_columns, score

ANALYSIS = ROOT / "analysis"
FIGDATA = ANALYSIS / "fig_data"
HIGH_U = 24.0
INNER_EPOCHS, FINAL_EPOCHS = 800, 1500
# Mode count selected by scripts/pinn_modes.py.
MODES = 3


def _variants(n_inv: int) -> dict:
    return {"iso":       dict(anisotropic=False, n_invariant=0),
            "aniso":     dict(anisotropic=True,  n_invariant=0),
            "aniso+res": dict(anisotropic=True,  residual=True, n_invariant=n_inv)}


def _fit(kw, X, Y, epochs):
    return PINNRegressor(n_modes=MODES, epochs=epochs, lr=0.03,
                         n_restarts=3, **kw).fit(X, Y)


def _splitter(groups, n=5):
    g = np.asarray(groups)
    u = len(np.unique(g))
    sp = GroupKFold(n_splits=min(n, u)) if u > n else LeaveOneGroupOut()
    return sp.split(np.zeros(len(g)), None, g)


# --------------------------------------------------------------------------- #
# 1. nested selection
# --------------------------------------------------------------------------- #

def nested_selection(lab, Xp, Xi, Y, scored, inner_splits=3) -> dict:
    """Choose the operator variant inside each training fold, then score outside it."""
    n_inv = Xi.shape[1] - Xp.shape[1]
    variants = _variants(n_inv)
    Xof = {"iso": Xp, "aniso": Xp, "aniso+res": Xi}
    cid = lab.condition_id.to_numpy()
    out = {}
    for pname, spec in _protocols(lab).items():
        pred = np.full_like(Y, np.nan, dtype=float)
        chosen = []
        folds = _protocol_splits(len(lab), spec)
        for tr, te in folds:
            # inner selection, grouped by condition (see module docstring)
            best, best_score = None, -np.inf
            for vname, kw in variants.items():
                X = Xof[vname]
                inner = np.full_like(Y, np.nan, dtype=float)
                for itr, ite in _splitter(cid[tr], n=inner_splits):
                    m = _fit(kw, X[tr][itr], Y[tr][itr], INNER_EPOCHS)
                    inner[tr[ite]] = m.predict(X[tr][ite])
                s = score(lab, scored & np.isfinite(inner).all(axis=1), inner, "")
                val = s.get("r2_pred", -np.inf) if s else -np.inf
                if val > best_score:
                    best, best_score = vname, val
            chosen.append(best)
            m = _fit(variants[best], Xof[best][tr], Y[tr], FINAL_EPOCHS)
            pred[te] = m.predict(Xof[best][te])
        s = score(lab, scored, pred, pname)
        s["chosen_per_fold"] = chosen
        s["selection_stable"] = len(set(chosen)) == 1
        out[pname] = s
        print(f"  {pname:26s} R2pred={s['r2_pred']:7.3f} phase={s['phase_hi']:6.1f}deg "
              f"chosen={'/'.join(chosen)}")
    return out


# --------------------------------------------------------------------------- #
# 2. blind scoring
# --------------------------------------------------------------------------- #

def blind_scoring(aug, lab, Xp, Xi, Y, scored) -> dict:
    """Fit on all labeled data, predict the unsealed blind conditions."""
    blind = aug[(aug.labeled == 0) & (aug.block == "blind")].copy()
    if not len(blind):
        return {}
    with open(ROOT / "datasets" / "conditions.csv", newline="", encoding="utf-8") as f:
        conds = {c["condition_id"]: c for c in csv.DictReader(f)}
    ids = sorted(set(blind.condition_id) & {k for k, v in conds.items()
                 if v["block"] == "blind" and v["config"] != "sealed"})
    if not ids:
        return {}
    n_inv = Xi.shape[1] - Xp.shape[1]
    res = {"status": "REPORTED EVIDENCE, NOT CONFIRMATORY -- BLND01-10 were "
                     "unsealed 2026-07-26 and the operator family was chosen after "
                     "that date. A confirmatory claim needs a new, prospectively "
                     "frozen block.",
           "conditions": ids, "methods": {}}
    rows = []
    for vname, kw in _variants(n_inv).items():
        X = Xi if vname == "aniso+res" else Xp
        Xb = pinn_design_matrix(blind, invariant_columns(lab)) if vname == "aniso+res" \
            else pinn_design_matrix(blind)
        m = _fit(kw, X[scored], Y[scored], FINAL_EPOCHS)
        P = m.predict(Xb)
        pts = []
        for cid in ids:
            sel = (blind.condition_id == cid).to_numpy()
            c = conds[cid]
            r = float(c["radius_mm"] or 30)
            for disk in (1, 2):
                tmag = float(c[f"disk{disk}_mass_g"] or 0) * r
                tang = float(c[f"disk{disk}_angle_deg"] or 0)
                u = P[sel, 2 * (disk - 1)] + 1j * P[sel, 2 * (disk - 1) + 1]
                z = u.mean()                       # complex average of repeats
                pts.append(dict(cid=cid, disk=disk, true=float(tmag),
                                est=float(abs(z)), true_ang=float(tang),
                                est_ang=float(np.degrees(np.angle(z))),
                                phase_label_valid=_fixture_angle_valid(tang)))
                rows.append(dict(variant=vname, condition_id=cid, disk=disk,
                                 true_gmm=tmag, est_gmm=float(abs(z)),
                                 true_ang=tang,
                                 est_ang=float(np.degrees(np.angle(z))),
                                 phase_label_valid=int(_fixture_angle_valid(tang)),
                                 balanced=int(tmag < 1e-6)))
        t = np.array([p["true"] for p in pts]); e = np.array([p["est"] for p in pts])
        tl, el = t >= LOAD_THRESH, e >= LOAD_THRESH
        tp = int((tl & el).sum()); fp = int((~tl & el).sum())
        fn = int((tl & ~el).sum()); tn = int((~tl & ~el).sum())
        det = cluster_bootstrap_proportion(
            (tl == el).astype(float), [p["cid"] for p in pts])
        pe = [abs(float(angdiff(p["est_ang"], p["true_ang"])))
              for p in pts if p["true"] >= HIGH_U and p["phase_label_valid"]]
        res["methods"][f"pinn_{vname}"] = {
            "mae": round(float(np.mean(np.abs(e - t))), 2),
            "rmse": round(float(np.sqrt(np.mean((e - t) ** 2))), 2),
            "mae_loaded": round(float(np.mean(np.abs(e[tl] - t[tl]))), 2) if tl.any() else None,
            "detect_acc": round((tp + tn) / len(pts), 4),
            "detect_acc_wilson95": wilson(tp + tn, len(pts)),
            "detect_acc_cluster95": det["ci95"],
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "roc_auc": round(roc_auc(tl.astype(int), e), 4),
            "phase_err_hi": round(float(np.mean(pe)), 1) if pe else None,
            "phase_err_hi_n": len(pe),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "points": pts}
        s = res["methods"][f"pinn_{vname}"]
        print(f"  {vname:10s} MAE={s['mae']:5.1f} loadedMAE={s['mae_loaded']:5.1f} "
              f"detect={s['detect_acc']*100:4.0f}% spec={s['specificity']*100:4.0f}% "
              f"phase={s['phase_err_hi']}")
    # Record which reference angles were dropped from the phase figures, so the
    # exclusion is auditable rather than implicit. Every variant scores the same
    # points, so the first one carries the list.
    res["phase_exclusions"] = [
        {"condition_id": p["cid"], "disk": p["disk"],
         "angle_deg": p["true_ang"],
         "reason": "reference angle is not on the 22.5-degree fixture grid"}
        for p in next(iter(res["methods"].values()))["points"]
        if not p["phase_label_valid"]]
    FIGDATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(FIGDATA / "pinn_blind.csv", index=False)
    return res


# --------------------------------------------------------------------------- #
# 3. forward validation
# --------------------------------------------------------------------------- #

def forward_validation(lab, Xp, Y, scored) -> dict:
    """Predict response from known unbalance with the operator and fixed ICM."""
    cid = lab.condition_id.to_numpy()
    pred = np.full((len(Y), 8), np.nan)
    for tr, te in _splitter(cid):
        m = _fit(_variants(0)["iso"], Xp[tr], Y[tr], FINAL_EPOCHS)
        pred[te] = m.predict_response(Xp[te], Y[te])

    # The design matrix stores all real response components followed by all
    # imaginary components.  Pack them in the interleaved representation used by
    # the operator and by the forward score without fitting an unnecessary model.
    meas = np.empty((len(Xp), 8), dtype=float)
    meas[:, 0::2] = Xp[:, :4]
    meas[:, 1::2] = Xp[:, 4:8]

    # Eq. (1) is also a forward model.  The ICM coefficients are calibrated once
    # from BASE/A007/A015 and then evaluated on the same observations as the
    # condition-held-out operator predictions.
    mats = influence_matrices(lab)
    icm_pred = np.full_like(pred, np.nan)
    for i, row in lab.iterrows():
        mat = mats.get(row["speed_id"])
        if mat is None:
            continue
        u = np.array([Y[i, 0] + 1j * Y[i, 1],
                      Y[i, 2] + 1j * Y[i, 3]])
        v = mat["A"] @ u
        icm_pred[i, 0::2] = v.real
        icm_pred[i, 1::2] = v.imag

    keep = scored & np.isfinite(pred).all(axis=1)
    Vp, Vi, Vm = pred[keep], icm_pred[keep], meas[keep]
    loaded = (np.abs(Y[keep, 0] + 1j * Y[keep, 1])
              + np.abs(Y[keep, 2] + 1j * Y[keep, 3])) >= LOAD_THRESH

    def summarize(P):
        amp = np.linalg.norm(Vm, axis=1)
        rel = np.linalg.norm(P - Vm, axis=1) / np.maximum(amp, 1e-12)
        d = pd.DataFrame(dict(condition_id=cid[keep], response_amp=amp,
                              rel_error=rel, loaded=loaded.astype(int)))
        by = d.groupby("condition_id").agg(
            rel_error=("rel_error", "median"),
            response_amp=("response_amp", "mean"),
            loaded=("loaded", "max")).reset_index()
        bl = by.loaded.astype(bool).to_numpy()
        return {
            "response_r2_pred": round(float(r2_pred(Vm.ravel(), P.ravel())), 4),
            "rel_error_median_all": round(float(np.median(by.rel_error)), 4),
            "rel_error_median_loaded": round(
                float(np.median(by.loc[bl, "rel_error"])), 4),
            "rel_error_median_unloaded": round(
                float(np.median(by.loc[~bl, "rel_error"])) if (~bl).any()
                else float("nan"), 4),
            "n_rows": int(keep.sum()),
            "n_conditions": int(by.condition_id.nunique()),
        }

    operator = summarize(Vp)
    icm = summarize(Vi)
    FIGDATA.mkdir(exist_ok=True)
    pd.DataFrame(dict(condition_id=cid[keep],
                      **{f"meas_{i}": Vm[:, i] for i in range(8)},
                      **{f"pred_{i}": Vp[:, i] for i in range(8)},
                      **{f"icm_pred_{i}": Vi[:, i] for i in range(8)})
                 ).to_csv(FIGDATA / "pinn_forward.csv", index=False)
    out = {"what": "held-out response prediction from the KNOWN unbalance; the label "
                   "is an input here, not a target",
           "response_r2_pred": operator["response_r2_pred"],
           "rel_error_median_all": operator["rel_error_median_all"],
           "rel_error_median_loaded": operator["rel_error_median_loaded"],
           "rel_error_median_unloaded": operator["rel_error_median_unloaded"],
           "n_rows": operator["n_rows"],
           "n_conditions": operator["n_conditions"],
           "methods": {"operator": operator, "icm": icm},
           "note": "relative error is inflated on near-balanced conditions because the "
                   "denominator is the small measured amplitude; relative-error "
                   "medians are first aggregated within condition"}
    for name, s in out["methods"].items():
        print(f"  {name:8s} response R2pred={s['response_r2_pred']:.3f}  "
              f"median rel.err loaded={s['rel_error_median_loaded']:.3f} "
              f"unloaded={s['rel_error_median_unloaded']:.3f} "
              f"(n={s['n_conditions']} conditions)")
    return out


def main():
    parts = [a[2:] for a in sys.argv[1:] if a.startswith("--")]
    if not parts:
        parts = ["nested", "blind", "forward"]
    df = pd.read_csv(ANALYSIS / "features.csv")
    aug = augment_with_physics(df)
    lab = aug[aug.labeled == 1].reset_index(drop=True)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    Y = lab[TARGETS].to_numpy(float)
    inv = invariant_columns(lab)
    Xp, Xi = pinn_design_matrix(lab), pinn_design_matrix(lab, inv)

    path = ANALYSIS / "pinn_validate.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if "nested" in parts:
        print("\n=== 1. nested model selection (variant chosen inside each fold) ===")
        out["nested_selection"] = nested_selection(lab, Xp, Xi, Y, scored)
    if "blind" in parts:
        print("\n=== 2. blind scoring (reported evidence, not confirmatory) ===")
        out["blind"] = blind_scoring(aug, lab, Xp, Xi, Y, scored)
    if "forward" in parts:
        print("\n=== 3. forward validation (the digital-twin claim) ===")
        out["forward"] = forward_validation(lab, Xp, Y, scored)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
