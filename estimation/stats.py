"""
Statistics helpers shared by the analysis scripts.

Exists to remove four defects that made the earlier benchmark numbers
un-publishable (see RESEARCH_PAPER_PLAN.md "Required corrections"):

  1. STATISTICAL UNIT. Magnitude/phase/confusion were computed per *acquisition*
     while localization was computed per *condition*, so the reported n mixed two
     different independent units. A condition is one physical mount of one mass at
     one angle; its 3 back-to-back records are near-duplicates (repeatability
     +-0.001-0.003 mils), so the independent unit is `condition_id`, not the record.
     `aggregate_conditions` collapses repeats by COMPLEX vector averaging.

  5. CIRCULAR STATISTICS. Angles were averaged arithmetically, which is wrong
     across the -180/+180 deg branch cut (mean(179, -179) = 0, not 180).
     `circmean_deg` averages on the unit circle.

  Plus two metric-naming fixes that follow from the above:
     - `r2_line` is the coefficient of determination of the fitted line est~true,
       i.e. squared Pearson correlation. It is blind to slope != 1 and to bias, so
       it is a LINEARITY score, not an accuracy score.
     - `r2_pred` = 1 - SSE/SST about the truth mean is the PREDICTIVE score and is
       the one to headline. A shrunk estimator scores well on the first and badly
       on the second; reporting only the first overstates accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["circmean_deg", "angdiff", "r2_line", "r2_pred", "wilson",
           "aggregate_conditions", "paired_bootstrap",
           "cluster_bootstrap_proportion", "roc_auc", "pr_auc"]


# --------------------------------------------------------------------------- #
# circular statistics
# --------------------------------------------------------------------------- #

def circmean_deg(a, weights=None) -> float:
    """Circular mean of angles in degrees, returned on (-180, 180].

    Arithmetic `mean` is invalid across the branch cut; this averages the unit
    vectors instead. Returns nan if no finite input (or a zero resultant).
    """
    a = np.asarray(a, float)
    m = np.isfinite(a)
    if not m.any():
        return float("nan")
    z = np.exp(1j * np.radians(a[m]))
    if weights is not None:
        w = np.asarray(weights, float)[m]
        if not np.isfinite(w).any() or np.nansum(np.abs(w)) == 0:
            return float("nan")
        z = z * w
    r = z.sum()
    if r == 0:
        return float("nan")
    return float(np.degrees(np.angle(r)))


def angdiff(a, b):
    """Signed smallest angular difference a-b in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# regression scores
# --------------------------------------------------------------------------- #

def r2_line(true, est) -> float:
    """LINEARITY score: R^2 of the fitted line est ~ true (== squared Pearson r).

    Insensitive to slope != 1 and to offset, so it must never be presented as
    predictive accuracy. Kept only for continuity with the earlier reports.
    """
    t, e = np.asarray(true, float), np.asarray(est, float)
    if len(t) < 3:
        return float("nan")
    slope, inter = np.polyfit(t, e, 1)
    ss_res = np.sum((e - (slope * t + inter)) ** 2)
    ss_tot = np.sum((e - e.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")


def r2_pred(true, est) -> float:
    """PREDICTIVE score: 1 - SSE/SST about the truth mean. Can go negative."""
    t, e = np.asarray(true, float), np.asarray(est, float)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - np.sum((e - t) ** 2) / ss_tot)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a proportion (small-n friendly)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# --------------------------------------------------------------------------- #
# the statistical unit
# --------------------------------------------------------------------------- #

def aggregate_conditions(points: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a per-acquisition disk-point table to one row per (condition, disk).

    `points` needs columns: cid, disk, true_mag, est_mag, true_ang, est_ang
    (optionally speed, which is dropped -- pass one speed at a time if you want
    per-speed condition-level numbers).

    Magnitude and phase are averaged as ONE COMPLEX VECTOR, which is what a
    condition-level operational decision actually uses, and which is circularly
    correct by construction. `n_acq` records how many records were collapsed.
    """
    p = points.copy()
    p["_Ut"] = p.true_mag * np.exp(1j * np.radians(p.true_ang.fillna(0.0)))
    p["_Ue"] = p.est_mag * np.exp(1j * np.radians(p.est_ang.fillna(0.0)))
    g = (p.groupby(["cid", "disk"], as_index=False)
           .agg(_Ut=("_Ut", "mean"), _Ue=("_Ue", "mean"), n_acq=("_Ut", "size")))
    out = pd.DataFrame({
        "cid": g.cid, "disk": g.disk, "n_acq": g.n_acq,
        "true_mag": np.abs(g._Ut), "est_mag": np.abs(g._Ue),
        "true_ang": np.degrees(np.angle(g._Ut)),
        "est_ang": np.degrees(np.angle(g._Ue)),
    })
    out["ang_err"] = np.abs(angdiff(out.est_ang, out.true_ang))
    out["loaded"] = out.true_mag > 1e-6
    return out


# --------------------------------------------------------------------------- #
# paired uncertainty
# --------------------------------------------------------------------------- #

def paired_bootstrap(values_a, values_b, groups, n_boot: int = 5000,
                     seed: int = 0, stat=np.mean) -> dict:
    """
    Cluster bootstrap of the paired difference stat(a) - stat(b), resampling whole
    GROUPS (conditions) with replacement so correlated repeats stay together.

    `values_a`/`values_b` are per-row scores for the two methods on the SAME rows.
    Returns the point difference and a percentile 95% CI. An interval containing 0
    means the campaign cannot separate the two methods on that metric.
    """
    a = np.asarray(values_a, float)
    b = np.asarray(values_b, float)
    g = np.asarray(groups)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[ok], b[ok], g[ok]
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        diffs[i] = stat(a[sel]) - stat(b[sel])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(stat(a) - stat(b))
    return {"diff": round(point, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_groups": int(len(uniq)), "n_rows": int(len(a))}


def cluster_bootstrap_proportion(outcomes, groups, n_boot: int = 5000,
                                 seed: int = 0) -> dict:
    """Percentile interval for a proportion with whole-group resampling.

    This is used when several binary outcomes belong to one independent unit,
    such as the two disk decisions within one rotor condition. Resampling rows
    independently would give an interval that is too narrow when the outcomes
    within a group are correlated.
    """
    y = np.asarray(outcomes, float)
    g = np.asarray(groups)
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        stats[i] = np.mean(y[sel])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"value": round(float(np.mean(y)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_groups": int(len(uniq)), "n_rows": int(len(y))}


# --------------------------------------------------------------------------- #
# threshold-free detection
# --------------------------------------------------------------------------- #

def roc_auc(y_true, score) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties handled by mid-ranks)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):                      # mid-rank for ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y_true, score) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if (y == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * y) / (y == 1).sum())
