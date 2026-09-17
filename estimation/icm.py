"""
Influence Coefficient Method (ICM) — quantitative unbalance estimation.

This is the core deliverable of the paper: given keyphasor-referenced 1X vectors
measured at the sensor planes, estimate the unbalance vector at each balancing
disk in *engineering units* (g·mm and degrees).

Procedure (multi-plane, multi-speed capable):
  1. Baseline run  -> V0  (complex 1X per sensor)                shape (S,)
  2. For each balancing plane j, add a known trial mass Wj and
     re-measure -> Vj. The influence coefficient is
         A[:, j] = (Vj - V0) / Wj                                shape (S, P)
  3. The unknown initial unbalance U0 solves  A @ U0 = V0
     (least squares if over-determined; regularized if ill-conditioned).
     The balancing correction to apply is simply W = -U0 (same magnitude,
     opposite phase).

Trial/estimated unbalances are complex: magnitude = mass*radius (g·mm),
angle = keyphasor-referenced phase (deg).
"""
from __future__ import annotations

import numpy as np


def influence_matrix(V0: np.ndarray, trial_runs: list[dict]) -> np.ndarray:
    """
    Build A (S sensors x P balancing planes) from trial runs.

    trial_runs[j] = {"mass": complex trial unbalance (g·mm ∠deg),
                     "response": complex 1X vector per sensor (S,)}
    """
    S = V0.size
    P = len(trial_runs)
    A = np.zeros((S, P), dtype=complex)
    for j, run in enumerate(trial_runs):
        Vj = np.asarray(run["response"], dtype=complex)
        A[:, j] = (Vj - V0) / run["mass"]
    return A


def estimate_unbalance(A: np.ndarray, V0: np.ndarray,
                       ridge: float = 0.0) -> np.ndarray:
    """
    Solve A @ U0 = V0 for the *unbalance present* U0 (P,) in trial-mass units
    (g·mm∠°). The balancing correction to apply is W = -U0.

    ridge > 0 applies Tikhonov regularization for ill-conditioned A (common when
    balancing planes respond similarly). Returns complex unbalance per plane.
    """
    b = np.asarray(V0, dtype=complex)
    if ridge > 0:
        # (A^H A + ridge I) U = A^H b
        AhA = A.conj().T @ A
        U = np.linalg.solve(AhA + ridge * np.eye(A.shape[1]), A.conj().T @ b)
    else:
        U, *_ = np.linalg.lstsq(A, b, rcond=None)
    return U


def correction_weights(U0: np.ndarray) -> np.ndarray:
    """Balancing correction weights = negative of the estimated unbalance."""
    return -np.asarray(U0, dtype=complex)


def to_mass_at_radius(unbalance_g_mm: complex, radius_mm: float) -> dict:
    """Convert an unbalance vector (g·mm∠°) to a correction mass at a radius."""
    mag = abs(unbalance_g_mm) / radius_mm       # grams
    angle = np.degrees(np.angle(unbalance_g_mm))
    return {"mass_g": mag, "angle_deg": angle, "radius_mm": radius_mm}


def residual_response(A: np.ndarray, U0: np.ndarray, V0: np.ndarray) -> np.ndarray:
    """
    Predicted residual 1X after applying the ideal correction W = -U0:
        V_res = V0 + A @ W = V0 - A @ U0  (-> ~0 for a perfect estimate).
    """
    return V0 + A @ correction_weights(U0)


def condition_report(A: np.ndarray) -> dict:
    """Diagnostics on the balancing problem (well-posedness of the estimate)."""
    s = np.linalg.svd(A, compute_uv=False)
    return {"singular_values": s, "condition_number": float(s[0] / s[-1])}
