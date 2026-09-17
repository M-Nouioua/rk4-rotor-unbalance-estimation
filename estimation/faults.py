"""
Fault-state classifier — "which rotor fault is on the machine right now".

Where `estimation/ml.py` regresses *how much* unbalance each disk carries, this
answers the discrete question a console shows an operator: is the machine clean,
is one disk loaded, or is this a two-plane (static / couple) load. Four classes,
all of them real labels from `datasets/conditions.csv`:

    balanced    neither disk above the load threshold
    disk1       disk 1 loaded, disk 2 clean
    disk2       disk 2 loaded, disk 1 clean
    two_plane   both disks loaded (in-phase, anti-phase or arbitrary)

LOAD_THRESH is the campaign's own 6 g.mm detection floor (`estimation.ml`), so
"balanced" means *no detectable load*, not literally zero mass — a 3 g.mm
condition is below what these probes resolve and is labelled clean on purpose.

NOT a bearing-fault classifier. The campaign never seeded a bearing defect:
bearings appear in this repo only as model stiffnesses in `config/rig.yaml` and
as the orbit trip limit. There is no rolling-element fault, no defect frequency
and no label for one anywhere in the 561 acquisitions, so no honest classifier
for one can be fitted from this data.

Evaluation is leak-free in the same sense as `ml.py`: GroupKFold on
condition_id, so the three consecutive records and the re-mounts of one
condition are never split across folds, and scaling is refit inside every fold.
Because the regulariser is tuned, the headline figure comes from *nested* CV
(`nested_cross_validate`) — C is chosen on an inner split of the training folds
only, so the reported score pays for its own model selection.

Two feature sets are fitted and reported:

  FULL        every inference-time column `ml.feature_columns` allows, i.e. what
              an offline analyst with both probes per plane can compute.
  DEPLOYABLE  only what the TwinOps rotor console can build from one live
              `RotorSample` — one 1X phasor per plane, plane-1 2X/3X, the
              baseline-subtracted response, rpm, and ratios derived from those.
              The console publishes no y-probe, so every orbit-ellipse
              descriptor (smajor, ellipticity, whirl, tilt) is out, and with it
              the campaign's own `plane_amp_ratio`, which is built from the
              ellipse semi-major axes.

The deployed model is fitted on DEPLOYABLE: a metric measured on features the
console cannot feed is not a metric it is entitled to display.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from estimation.ml import LOAD_THRESH, feature_columns

CLASSES = ["balanced", "disk1", "disk2", "two_plane"]

#: Regularisation grid searched by the inner fold of `nested_cross_validate`.
C_GRID = (0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

#: Columns the rotor console can compute from a single live sample, after
#: `add_derived`. The `resp_*` family is the baseline-subtracted 1X response —
#: the added-mass signature the influence-coefficient method localizes on, and
#: the reason a linear model has anything to work with here.
DEPLOYABLE = [
    "rpm",
    "plane1_x_1x_amp", "plane1_x_1x_re", "plane1_x_1x_im",
    "plane2_x_1x_amp", "plane2_x_1x_re", "plane2_x_1x_im",
    "plane1_x_2x_amp", "plane1_x_2x_ratio",
    "plane1_x_3x_amp", "plane1_x_3x_ratio",
    "plane1_x_resp_amp", "plane1_x_resp_re", "plane1_x_resp_im",
    "plane2_x_resp_amp", "plane2_x_resp_re", "plane2_x_resp_im",
    "p2_over_p1_amp", "plane_phase_diff",
    "resp_log_ratio", "resp_phase_diff", "resp_phase_diff_abs",
    "resp_sum", "resp_norm1",
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the console-computable cross-plane cues. Idempotent, no label input.

    Trees and linear models alike cannot form a ratio of two inputs, and the
    localization cue on this rig *is* a ratio: which plane responded more to the
    added mass, and with what relative phase. Handing those over explicitly is
    worth about +0.07 macro-F1 over the raw columns alone.
    """
    df = df.copy()
    a1 = df["plane1_x_1x_amp"].to_numpy(float)
    a2 = df["plane2_x_1x_amp"].to_numpy(float)
    df["p2_over_p1_amp"] = a2 / np.maximum(a1, 1e-9)

    r1 = np.hypot(df["plane1_x_resp_re"], df["plane1_x_resp_im"]).to_numpy(float)
    r2 = np.hypot(df["plane2_x_resp_re"], df["plane2_x_resp_im"]).to_numpy(float)
    df["resp_log_ratio"] = np.log10(np.maximum(r2, 1e-9) / np.maximum(r1, 1e-9))
    df["resp_sum"] = r1 + r2
    df["resp_norm1"] = r1 / np.maximum(r1 + r2, 1e-9)

    a_1 = np.degrees(np.arctan2(df["plane1_x_resp_im"], df["plane1_x_resp_re"]))
    a_2 = np.degrees(np.arctan2(df["plane2_x_resp_im"], df["plane2_x_resp_re"]))
    diff = (a_2 - a_1 + 180.0) % 360.0 - 180.0   # wrapped to (-180, 180]
    df["resp_phase_diff"] = diff
    # In-phase (static) and anti-phase (couple) two-plane loads sit at opposite
    # ends of this; the magnitude alone separates "same direction" from "opposed"
    # without the model having to learn the wrap.
    df["resp_phase_diff_abs"] = np.abs(diff)
    return df


def label(df: pd.DataFrame, thresh: float = LOAD_THRESH) -> np.ndarray:
    """Class per row, from the ground-truth added unbalance per disk."""
    d1 = df["U1_mag"].to_numpy(float) >= thresh
    d2 = df["U2_mag"].to_numpy(float) >= thresh
    return np.where(d1 & d2, "two_plane",
                    np.where(d1, "disk1", np.where(d2, "disk2", "balanced")))


def deployable_columns() -> list[str]:
    return list(DEPLOYABLE)


def full_columns(df: pd.DataFrame) -> list[str]:
    """Every inference-time column, including the y-probe and ellipse features."""
    return feature_columns(add_derived(df))


def build_classifier(kind: str = "logreg", *, C: float = 3.0,
                     compact: bool = False) -> Pipeline:
    """
    'logreg'  multinomial logistic regression — the deployed model. On 85
              conditions it generalizes better across conditions than either
              ensemble, which is not a surprise: the trees have enough capacity
              to memorise condition identity from a 24-dimensional fingerprint.
    'et'      Extra-Trees, the family `ml.py` uses for regression
    'rf'      Random-Forest

    `compact` caps ensemble depth and tree count; it exists so the tree variants
    are comparable at a size that could be exported, not because it helps.
    """
    if kind in ("et", "rf"):
        Core = ExtraTreesClassifier if kind == "et" else RandomForestClassifier
        core = Core(
            n_estimators=120 if compact else 500,
            max_depth=8 if compact else None,
            min_samples_leaf=3 if compact else 2,
            max_features=0.5,
            n_jobs=-1,
            random_state=0,
        )
        return Pipeline([("scale", StandardScaler()), ("model", core)])
    if kind == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=5000)),
        ])
    raise ValueError(kind)


def _score(y: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics over pooled out-of-fold predictions.

    Pooled rather than averaged per fold, so a fold that happens to hold few
    `balanced` conditions can neither flatter nor wreck the macro figure.
    """
    cm = confusion_matrix(y, pred, labels=CLASSES)
    per_class = {
        c: {
            "support": int((y == c).sum()),
            "recall": float((pred[y == c] == c).mean()) if (y == c).any() else float("nan"),
            "precision": float((y[pred == c] == c).mean()) if (pred == c).any() else float("nan"),
        }
        for c in CLASSES
    }
    # Detection collapses the three loaded classes into one — "is anything on
    # the rotor at all", which is the question an alarm actually asks and the
    # one this rig answers well.
    det_true = y != "balanced"
    det_pred = pred != "balanced"
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro")),
        "balanced_accuracy": float(np.mean([per_class[c]["recall"] for c in CLASSES])),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": CLASSES,
        "detection_recall": float(det_pred[det_true].mean()),
        "detection_specificity": float((~det_pred[~det_true]).mean()),
        "majority_baseline": float(max((y == c).mean() for c in CLASSES)),
        "chance": 1.0 / len(CLASSES),
    }


def _prepare(df: pd.DataFrame, feats: list[str] | None):
    lab = add_derived(df[df["labeled"] == 1]).reset_index(drop=True)
    feats = feats or deployable_columns()
    return lab, feats, lab[feats].to_numpy(float), label(lab), lab["condition_id"].to_numpy()


def cross_validate(df: pd.DataFrame, kind: str = "logreg", *,
                   feats: list[str] | None = None, C: float = 3.0,
                   compact: bool = False, n_splits: int = 5) -> dict:
    """Flat leak-free CV at a fixed hyper-parameter. Comparison only — the
    headline number is `nested_cross_validate`, because C is tuned."""
    lab, feats, X, y, groups = _prepare(df, feats)
    pred = np.empty(len(lab), dtype=object)
    fold = np.full(len(lab), -1)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        pipe = build_classifier(kind, C=C, compact=compact)
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
        fold[te] = k
    pred = pred.astype(str)
    return {
        "kind": kind, "compact": compact, "C": C if kind == "logreg" else None,
        "selection": "fixed", "n_features": len(feats), "features": feats,
        "n": int(len(lab)), "n_conditions": int(lab["condition_id"].nunique()),
        "n_splits": n_splits, "load_thresh_gmm": LOAD_THRESH,
        **_score(y, pred),
        "y_true": y.tolist(), "pred": pred.tolist(), "fold": fold.tolist(),
    }


def select_C(df: pd.DataFrame, *, feats: list[str] | None = None,
             n_splits: int = 5, c_grid=C_GRID) -> tuple[float, list[dict]]:
    """
    Pick the deployed regulariser by the one-standard-error rule over all
    labelled data.

    `nested_cross_validate` scores the *procedure*; this runs the same procedure
    once on everything to produce the model that actually ships. Returns the
    chosen C and the full curve, so the training report can show that the choice
    sits on a plateau rather than a peak.
    """
    lab, feats, X, y, groups = _prepare(df, feats)
    curve = []
    for C in c_grid:
        fold_f1 = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            pipe = build_classifier("logreg", C=C).fit(X[tr], y[tr])
            fold_f1.append(f1_score(y[te], pipe.predict(X[te]),
                                    labels=CLASSES, average="macro"))
        curve.append({
            "C": float(C),
            "macro_f1": float(np.mean(fold_f1)),
            "se": float(np.std(fold_f1, ddof=1) / np.sqrt(len(fold_f1))),
        })
    top = max(curve, key=lambda r: r["macro_f1"])
    floor = top["macro_f1"] - top["se"]
    chosen = min(r["C"] for r in curve if r["macro_f1"] >= floor)
    return chosen, curve


def nested_cross_validate(df: pd.DataFrame, *, feats: list[str] | None = None,
                          n_splits: int = 5, inner_splits: int = 4,
                          c_grid=C_GRID) -> dict:
    """
    Honest score for a tuned model: C is selected on an inner GroupKFold of the
    outer training folds only and the selected model is then scored on the
    held-out outer fold it has never seen.

    Selection uses the one-standard-error rule, not the arg-max. Macro-F1 on
    this campaign is flat in C above ~1 (0.473 to 0.503 from C=1 to C=30) while
    the fold-to-fold standard error is ~0.06, so an arg-max search walks
    straight to whatever ceiling the grid has — it picked the top value at every
    ceiling offered, which is a boundary artefact rather than a fit. Taking the
    *most regularised* C whose inner score is within one standard error of the
    best one lands on a stable interior value and is reproducible when the grid
    changes.

    `chosen_C` records what each outer fold picked; agreement across folds is
    the evidence that the rule is stable.
    """
    lab, feats, X, y, groups = _prepare(df, feats)
    pred = np.empty(len(lab), dtype=object)
    fold = np.full(len(lab), -1)
    chosen: list[float] = []

    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        g_tr = groups[tr]
        # Inner score per C, kept per fold so the rule has a spread to work with.
        per_c: list[tuple[float, float, float]] = []
        for C in c_grid:
            fold_f1 = []
            for i_tr, i_te in GroupKFold(n_splits=inner_splits).split(X[tr], y[tr], g_tr):
                pipe = build_classifier("logreg", C=C).fit(X[tr][i_tr], y[tr][i_tr])
                fold_f1.append(f1_score(y[tr][i_te], pipe.predict(X[tr][i_te]),
                                        labels=CLASSES, average="macro"))
            mean = float(np.mean(fold_f1))
            se = float(np.std(fold_f1, ddof=1) / np.sqrt(len(fold_f1)))
            per_c.append((C, mean, se))

        top_c, top_mean, top_se = max(per_c, key=lambda r: r[1])
        floor = top_mean - top_se
        best_c = min(C for C, mean, _ in per_c if mean >= floor)
        pipe = build_classifier("logreg", C=best_c).fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
        fold[te] = k
        chosen.append(best_c)

    pred = pred.astype(str)
    return {
        "kind": "logreg", "compact": False, "selection": "nested",
        "c_grid": list(c_grid), "chosen_C": chosen,
        "n_features": len(feats), "features": feats,
        "n": int(len(lab)), "n_conditions": int(lab["condition_id"].nunique()),
        "n_splits": n_splits, "inner_splits": inner_splits,
        "load_thresh_gmm": LOAD_THRESH,
        **_score(y, pred),
        "y_true": y.tolist(), "pred": pred.tolist(), "fold": fold.tolist(),
    }


class FaultClassifier:
    """Fit-once / predict-many wrapper, mirroring `ml.UnbalanceML`."""

    def __init__(self, kind: str = "logreg", *, C: float = 3.0,
                 feats: list[str] | None = None, compact: bool = False):
        self.kind = kind
        self.C = C
        self.compact = compact
        self.pipe = build_classifier(kind, C=C, compact=compact)
        self.feats = feats or deployable_columns()

    def fit(self, df: pd.DataFrame):
        lab = add_derived(df[df["labeled"] == 1])
        self.pipe.fit(lab[self.feats].to_numpy(float), label(lab))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(add_derived(df)[self.feats].to_numpy(float))

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X = add_derived(df)[self.feats].to_numpy(float)
        return pd.DataFrame(self.pipe.predict_proba(X),
                            columns=list(self.pipe.named_steps["model"].classes_))

    def export(self) -> dict:
        """
        Serialise to plain numbers so the browser can run this model directly.

        A standardiser plus a multinomial linear layer is the whole model, which
        is the other reason to prefer it here: the console does real inference on
        real fitted coefficients instead of calling out to a Python service that
        this deployment does not have.
        """
        if self.kind != "logreg":
            raise ValueError("only the logistic model is exportable to the console")
        scale: StandardScaler = self.pipe.named_steps["scale"]
        clf: LogisticRegression = self.pipe.named_steps["model"]
        return {
            "kind": "multinomial-logistic",
            "features": list(self.feats),
            "classes": [str(c) for c in clf.classes_],
            "mean": [float(v) for v in scale.mean_],
            "scale": [float(v) for v in scale.scale_],
            "coef": [[float(v) for v in row] for row in clf.coef_],
            "intercept": [float(v) for v in clf.intercept_],
            "C": float(self.C),
        }

    def coefficients(self) -> pd.DataFrame:
        clf = self.pipe.named_steps["model"]
        return pd.DataFrame(clf.coef_, index=clf.classes_, columns=self.feats).T
