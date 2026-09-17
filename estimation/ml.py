"""
Data-driven unbalance estimator — the "with-ML" twin.

Where the physics twin (`estimation/icm.py`) inverts a *linear, speed-specific*
influence matrix A built from trial runs, this learns a single nonlinear map

    physics-informed features  ->  added unbalance per disk  (complex g.mm)

directly from the labeled campaign. One model spans both dwell speeds (rpm is a
feature), needs no per-speed trial run at inference, and can absorb the rotor
anisotropy and 2-plane cross-talk that a single isotropic A cannot.

Targets are the real/imaginary parts of the added unbalance at each disk
(U1_re, U1_im, U2_re, U2_im) — regressing the vector avoids phase wrap-around;
magnitude, keyphasor phase and plane localization are derived afterwards.

Evaluation is **leak-free**: repeats and re-mounts of one condition are never
split across folds (GroupKFold on condition_id), and per-condition standardization
is refit inside each fold. `cross_val_estimate` returns honest out-of-fold
predictions for every labeled acquisition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

TARGETS = ["U1_re", "U1_im", "U2_re", "U2_im"]
LOAD_THRESH = 6.0   # g.mm, matches scripts/analyze.py localization


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Physics-informed inputs a deployed twin actually has at inference:
    baseline-subtracted 1X response, harmonic ratios, orbit-ellipse descriptors,
    cross-plane cues and rpm. Excludes raw absolute 1X phase (wrap-around) and the
    label/metadata columns.
    """
    drop_prefixes = ("U1_", "U2_")
    drop_exact = {"file", "condition_id", "block", "config", "speed_id",
                  "rpm_measured", "mount_idx", "acq_idx", "labeled",
                  "disk1_loaded", "disk2_loaded"}
    cols = []
    for c in df.columns:
        if c in drop_exact or c.startswith(drop_prefixes):
            continue
        if c.endswith("_1x_phase"):          # keep re/im instead of wrapped degrees
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_regressor(kind: str = "et") -> Pipeline:
    """A regression pipeline over the 4 unbalance components.

    'et' (Extra-Trees) is the default — best cross-validated magnitude R2 and
    lowest cross-talk on this campaign, with a tree ensemble for uncertainty.
    'rf' Random-Forest and 'ridge' (linear baseline) are kept for comparison.
    """
    if kind in ("et", "rf"):
        Core = ExtraTreesRegressor if kind == "et" else RandomForestRegressor
        core = Core(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                    n_jobs=-1, random_state=0)
        return Pipeline([("scale", StandardScaler()), ("model", core)])
    if kind == "ridge":
        return Pipeline([("scale", StandardScaler()),
                         ("model", MultiOutputRegressor(Ridge(alpha=1.0)))])
    raise ValueError(kind)


def _derive(pred: np.ndarray) -> dict:
    """Complex U per disk + magnitude/phase/localization from a (…,4) prediction."""
    u1 = pred[:, 0] + 1j * pred[:, 1]
    u2 = pred[:, 2] + 1j * pred[:, 3]
    return {
        "U1": u1, "U2": u2,
        "U1_mag": np.abs(u1), "U2_mag": np.abs(u2),
        "U1_ang": np.degrees(np.angle(u1)), "U2_ang": np.degrees(np.angle(u2)),
        "disk1_loaded": (np.abs(u1) > LOAD_THRESH).astype(int),
        "disk2_loaded": (np.abs(u2) > LOAD_THRESH).astype(int),
    }


def cross_val_estimate(df: pd.DataFrame, kind: str = "et", n_splits: int = 5,
                       return_std: bool = True):
    """
    Leak-free out-of-fold predictions for every labeled acquisition.

    Groups = condition_id (all repeats/mounts of a condition share a fold).
    Returns a DataFrame aligned to `df` with predicted components, magnitude,
    phase, localization and (for the RF) a per-sample uncertainty from the tree
    ensemble spread, propagated to magnitude.
    """
    lab = df[df["labeled"] == 1].reset_index(drop=True)
    feats = feature_columns(lab)
    X = lab[feats].to_numpy(float)
    Y = lab[TARGETS].to_numpy(float)
    groups = lab["condition_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros_like(Y)
    std = np.zeros(len(lab))
    fold = np.full(len(lab), -1)

    for k, (tr, te) in enumerate(gkf.split(X, Y, groups)):
        pipe = build_regressor(kind)
        pipe.fit(X[tr], Y[tr])
        pred[te] = pipe.predict(X[te])
        fold[te] = k
        if return_std and kind in ("et", "rf"):
            # ensemble spread on U1/U2 vector length -> magnitude uncertainty
            scaler = pipe.named_steps["scale"]
            rf = pipe.named_steps["model"]
            Xs = scaler.transform(X[te])
            per_tree = np.stack([t.predict(Xs) for t in rf.estimators_])  # (T, n, 4)
            comp_std = per_tree.std(axis=0)                                # (n, 4)
            std[te] = 0.5 * (np.hypot(comp_std[:, 0], comp_std[:, 1])
                             + np.hypot(comp_std[:, 2], comp_std[:, 3]))

    out = _derive(pred)
    res = lab[["file", "condition_id", "block", "config", "speed_id",
               "U1_mag", "U2_mag", "U1_ang", "U2_ang",
               "disk1_loaded", "disk2_loaded"]].copy()
    res = res.rename(columns={c: f"true_{c}" for c in
                              ["U1_mag", "U2_mag", "U1_ang", "U2_ang",
                               "disk1_loaded", "disk2_loaded"]})
    for k, v in out.items():
        if k in ("U1", "U2"):
            continue
        res[f"est_{k}"] = v
    res["est_std_gmm"] = std
    res["fold"] = fold
    return res


class UnbalanceML:
    """Fit-once / predict-many wrapper for deployment (e.g. on the blind set)."""

    def __init__(self, kind: str = "et"):
        self.kind = kind
        self.pipe = build_regressor(kind)
        self.feats: list[str] = []

    def fit(self, df: pd.DataFrame):
        lab = df[df["labeled"] == 1]
        self.feats = feature_columns(lab)
        self.pipe.fit(lab[self.feats].to_numpy(float),
                      lab[TARGETS].to_numpy(float))
        return self

    def predict(self, df: pd.DataFrame) -> dict:
        X = df[self.feats].to_numpy(float)
        return _derive(self.pipe.predict(X))

    def feature_importance(self) -> pd.Series:
        rf = self.pipe.named_steps["model"]
        imp = getattr(rf, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.feats).sort_values(ascending=False)
