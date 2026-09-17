"""
Quantitative misalignment estimation.

Misalignment does not have a single textbook estimator the way unbalance does
(ICM) — this is the paper's novelty surface. Two complementary routes are
provided and reconciled:

  (A) Empirical calibration: from the shim test matrix (known parallel offset
      in mm / angular offset in mrad -> measured 2X/1X ratio, cross-plane phase,
      orbit ellipticity), fit a regression and invert it to estimate severity
      from a new run.

  (B) Physics fitting: model misalignment in ROSS as a coupling-induced 2X
      (and 1X) reaction force parameterized by the offset; optimize the offset
      so simulated features match measured features.

Reconciling (A) and (B) — empirical vs physics — is the publishable result.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---- feature vector shared by both routes -----------------------------------

def misalignment_features(orders_p1: dict, orders_p2: dict) -> np.ndarray:
    """
    Discriminative feature vector:
      [2X/1X @p1, 2X/1X @p2, |2X p1|, |2X p2|, cross-plane 1X phase diff (rad),
       cross-plane 2X phase diff (rad)]
    """
    def ratio(d):
        a1, a2 = abs(d.get(1, 0)), abs(d.get(2, 0))
        return a2 / a1 if a1 else 0.0

    dphi1 = np.angle(orders_p2.get(1, 1)) - np.angle(orders_p1.get(1, 1))
    dphi2 = np.angle(orders_p2.get(2, 1)) - np.angle(orders_p1.get(2, 1))
    return np.array([
        ratio(orders_p1), ratio(orders_p2),
        abs(orders_p1.get(2, 0)), abs(orders_p2.get(2, 0)),
        np.angle(np.exp(1j * dphi1)), np.angle(np.exp(1j * dphi2)),
    ])


# ---- (A) empirical calibration ----------------------------------------------

class EmpiricalCalibrator:
    """Linear (or polynomial) map: features -> [parallel_mm, angular_mrad]."""

    def __init__(self, degree: int = 1):
        self.degree = degree
        self.coef_: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(X))]
        for d in range(1, self.degree + 1):
            cols.append(X ** d)
        return np.hstack([c.reshape(len(X), -1) for c in cols])

    def fit(self, feats: np.ndarray, targets: np.ndarray) -> "EmpiricalCalibrator":
        """feats (N, F); targets (N, 2) = [parallel_mm, angular_mrad]."""
        Phi = self._design(feats)
        self.coef_, *_ = np.linalg.lstsq(Phi, targets, rcond=None)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "fit() first"
        return self._design(np.atleast_2d(feats)) @ self.coef_


# ---- (B) physics fitting ----------------------------------------------------

def fit_offset_to_features(measured_feats: np.ndarray, simulate_feats,
                           x0=(0.1, 1.0), bounds=((0, 0), (5, 20))):
    """
    Estimate [parallel_mm, angular_mrad] by matching simulated to measured
    features. `simulate_feats(offset)` must return a feature vector from the
    ROSS misalignment model for a candidate offset.
    """
    def residual(offset):
        return simulate_feats(offset) - measured_feats

    sol = least_squares(residual, x0=np.asarray(x0, float), bounds=bounds)
    return {"parallel_mm": sol.x[0], "angular_mrad": sol.x[1],
            "cost": float(sol.cost), "success": bool(sol.success)}
