"""
Influence Coefficient Method demo — quantitative unbalance estimation.

Runs a self-contained synthetic 2-plane example so the ICM solver is verifiable
without hardware. Replace the synthetic V0 / trial responses with measured 1X
vectors (from processing.order_tracking.harmonic_vectors) for the real study.

    python -m scripts.estimate_unbalance
"""
from __future__ import annotations

import numpy as np

from estimation.icm import (
    influence_matrix, estimate_unbalance, to_mass_at_radius,
    residual_response, condition_report, correction_weights,
)


def synthetic_case():
    """Ground-truth unbalance -> synthetic 1X responses via a known A."""
    # true influence coefficients (2 sensors x 2 balancing planes), complex
    A_true = np.array([[0.9 + 0.1j, 0.2 - 0.1j],
                       [0.15 + 0.05j, 0.8 - 0.2j]])
    U_true = np.array([12.0 * np.exp(1j * np.radians(30)),   # g·mm ∠30°
                        7.0 * np.exp(1j * np.radians(-75))])  # g·mm ∠-75°
    V0 = A_true @ U_true                                      # baseline response
    # trial masses added one plane at a time
    trials = []
    for j, w in enumerate([5.0 * np.exp(1j * 0.0), 5.0 * np.exp(1j * 0.0)]):
        Uj = U_true.copy().astype(complex)
        Wj = np.zeros(2, dtype=complex); Wj[j] = w
        Vj = A_true @ (Uj + Wj)
        trials.append({"mass": w, "response": Vj})
    return V0, trials, U_true


def main():
    V0, trials, U_true = synthetic_case()
    A = influence_matrix(V0, trials)
    print("condition number:", round(condition_report(A)["condition_number"], 2))

    U_est = estimate_unbalance(A, V0)
    print("\nplane   estimated (g·mm ∠°)      true (g·mm ∠°)")
    for j in range(len(U_est)):
        ae, pe = abs(U_est[j]), np.degrees(np.angle(U_est[j]))
        at, pt = abs(U_true[j]), np.degrees(np.angle(U_true[j]))
        print(f"  {j+1}   {ae:6.2f} ∠ {pe:7.1f}     {at:6.2f} ∠ {pt:7.1f}")

    W = correction_weights(U_est)
    unb = to_mass_at_radius(U_est[0], radius_mm=30.0)
    cor = to_mass_at_radius(W[0], radius_mm=30.0)
    print(f"\ndisk1 unbalance present: {unb['mass_g']:.3f} g @ {unb['angle_deg']:.1f}° "
          f"(r={unb['radius_mm']} mm)")
    print(f"disk1 correction to add: {cor['mass_g']:.3f} g @ {cor['angle_deg']:.1f}°")
    print("predicted residual 1X after correction:",
          np.round(np.abs(residual_response(A, U_est, V0)), 4))


if __name__ == "__main__":
    main()
