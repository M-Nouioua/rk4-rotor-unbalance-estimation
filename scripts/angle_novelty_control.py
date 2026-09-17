"""
Angle-novelty control — is the angle-holdout collapse about UNSEEN ANGLE, or about
having less training data?

This closes the obvious referee objection to the factor matrix. The campaign's
angular sectors are badly unbalanced (33/1/16/1/16/1/16/1 conditions), so holding
out a sector also removes a large slice of training data. A naive reading cannot
tell the two effects apart.

Design: hold out ONE well-populated 16-condition sector (unseen angle), and compare
against holding out 16 RANDOMLY chosen conditions (same n_train, angular coverage
preserved). Everything else -- model, hyperparameters, features, scoring -- is fixed.

    python -m scripts.angle_novelty_control   -> analysis/angle_novelty_control.json

Result on this campaign: the degradation is essentially ALL angle novelty. The two
arms train on the same 69 conditions, yet differ by ~2.6 in predictive R^2 and ~66
degrees of phase error.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from config import ROOT
from estimation.ml import build_regressor, feature_columns, TARGETS
from estimation.twin import augment_with_physics, CALIB
from estimation.stats import aggregate_conditions, r2_pred, angdiff

ANALYSIS = ROOT / "analysis"
HIGH_U = 24.0
SECTORS = (2, 4, 6)      # the well-populated 16-condition sectors (90/180/270 deg)
N_RANDOM = 5


def main():
    df = pd.read_csv(ANALYSIS / "features.csv")
    aug = augment_with_physics(df)
    lab = aug[aug.labeled == 1].reset_index(drop=True)
    X = lab[feature_columns(lab)].to_numpy(float)
    Y = lab[TARGETS].to_numpy(float)
    scored = (~lab.condition_id.isin(CALIB)).to_numpy()
    cids = lab.condition_id.to_numpy()
    U = [lab[f"U{k}_mag"].to_numpy() * np.exp(1j * np.radians(lab[f"U{k}_ang"].fillna(0).to_numpy()))
         for k in (1, 2)]
    ang = np.where(lab.U1_mag.to_numpy() >= lab.U2_mag.to_numpy(),
                   lab.U1_ang.fillna(0).to_numpy(), lab.U2_ang.fillna(0).to_numpy())
    sec = np.round(ang / 45).astype(int) % 8

    def run(test_mask, tag):
        P = build_regressor("et").fit(X[~test_mask], Y[~test_mask]).predict(X[test_mask])
        idx = np.where(test_mask)[0]
        keep = scored[idx]
        rows = []
        for k in (1, 2):
            uh = P[:, 2 * (k - 1)] + 1j * P[:, 2 * (k - 1) + 1]
            ut = U[k - 1][idx]
            rows.append(pd.DataFrame(dict(
                cid=cids[idx][keep], disk=k,
                true_mag=np.abs(ut[keep]), est_mag=np.abs(uh[keep]),
                true_ang=np.degrees(np.angle(ut[keep])),
                est_ang=np.degrees(np.angle(uh[keep])))))
        T = pd.concat(rows, ignore_index=True)
        T["ang_err"] = np.abs(angdiff(T.est_ang, T.true_ang))
        T["loaded"] = T.true_mag > 1e-6
        C = aggregate_conditions(T)
        L = C[C.loaded]; hi = L[L.true_mag >= HIGH_U]
        rec = {"tag": tag,
               "n_train_conditions": int(len(set(cids[~test_mask]))),
               "n_test_conditions": int(len(set(cids[test_mask]))),
               "r2_pred": round(float(r2_pred(L.true_mag, L.est_mag)), 4) if len(L) > 2 else None,
               "phase_hi": round(float(hi.ang_err.mean()), 2) if len(hi) else None,
               "phase_hi_n": int(len(hi))}
        print(f"  {tag:42s} n_train={rec['n_train_conditions']:3d} "
              f"n_test={rec['n_test_conditions']:2d} "
              f"R2pred={rec['r2_pred']:7.3f} phase={rec['phase_hi']:6.1f}deg")
        return rec

    out = {"design": "one 16-condition angle sector withheld vs 16 random conditions "
                     "withheld; identical n_train, model and scoring",
           "sector_condition_counts": {int(k): int(v) for k, v in
                                       pd.Series(sec).value_counts().sort_index().items()},
           "unseen_angle": [], "size_matched_random": []}

    print("=== A) unseen angle sector withheld ===")
    for s in SECTORS:
        out["unseen_angle"].append(run(sec == s, f"sector {s} (={s*45} deg) withheld"))
    print("\n=== B) size-matched random control ===")
    uniq = np.array(sorted(set(cids)))
    rng = np.random.default_rng(0)
    for i in range(N_RANDOM):
        pick = rng.choice(uniq, size=16, replace=False)
        out["size_matched_random"].append(run(np.isin(cids, pick), f"random draw {i+1}"))

    def mean(arm, key):
        v = [r[key] for r in out[arm] if r[key] is not None]
        return float(np.mean(v))

    out["verdict"] = {
        "unseen_angle_r2_pred": round(mean("unseen_angle", "r2_pred"), 4),
        "unseen_angle_phase": round(mean("unseen_angle", "phase_hi"), 2),
        "random_r2_pred": round(mean("size_matched_random", "r2_pred"), 4),
        "random_phase": round(mean("size_matched_random", "phase_hi"), 2),
        "delta_r2_pred_from_angle_novelty":
            round(mean("unseen_angle", "r2_pred") - mean("size_matched_random", "r2_pred"), 4),
        "delta_phase_from_angle_novelty":
            round(mean("unseen_angle", "phase_hi") - mean("size_matched_random", "phase_hi"), 2),
        "reading": "Both arms train on the same number of conditions, so the gap is "
                   "attributable to angle novelty rather than to reduced training data.",
    }
    v = out["verdict"]
    print("\n=== VERDICT ===")
    print(f"  unseen angle : R2pred {v['unseen_angle_r2_pred']:6.3f}  phase {v['unseen_angle_phase']:5.1f} deg")
    print(f"  size-matched : R2pred {v['random_r2_pred']:6.3f}  phase {v['random_phase']:5.1f} deg")
    print(f"  from ANGLE NOVELTY: dR2pred={v['delta_r2_pred_from_angle_novelty']:+.3f} "
          f"dphase={v['delta_phase_from_angle_novelty']:+.1f} deg")
    (ANALYSIS / "angle_novelty_control.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANALYSIS / 'angle_novelty_control.json'}")


if __name__ == "__main__":
    main()
