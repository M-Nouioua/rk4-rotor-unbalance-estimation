"""
ROSS rotordynamic model of the 2-disk RK4 rotor kit.

Provides the *physics core* of the digital twin:
  - Campbell diagram + first critical speed (validate against a run-up).
  - Unbalance forced response (predicts influence coefficients for the ICM).
  - A hook for injecting misalignment as coupling reaction forces (2X source).

Geometry/bearing values come from config/rig.yaml and MUST be calibrated to the
real rig (see model updating in estimation/). ROSS API: ross-rotordynamics>=1.3.
"""
from __future__ import annotations

import numpy as np

try:
    import ross as rs
    _HAS_ROSS = True
except ImportError:
    _HAS_ROSS = False

MM = 1e-3


def build_rotor(cfg: dict):
    """Assemble a ROSS Rotor from the rig config."""
    if not _HAS_ROSS:
        raise RuntimeError("ross not installed. `pip install ross-rotordynamics`.")

    r = cfg["rotor"]
    steel = rs.materials.steel

    L = r["shaft"]["length_mm"] * MM
    n_el = int(r["shaft"]["n_elements"])
    od = r["shaft"]["outer_diameter_mm"] * MM
    le = L / n_el

    shaft = [
        rs.ShaftElement(
            L=le, idl=0.0, odl=od, material=steel,
            shear_effects=True, rotary_inertia=True,
        )
        for _ in range(n_el)
    ]

    disks = [
        rs.DiskElement.from_geometry(
            n=d["node"],
            material=steel,
            width=d["thickness_mm"] * MM,
            i_d=od,
            o_d=d["outer_diameter_mm"] * MM,
        )
        for d in r["disks"]
    ]

    # float() guards against PyYAML parsing "1.0e6"-style values as strings
    bearings = [
        rs.BearingElement(
            n=int(b["node"]),
            kxx=float(b["kxx_N_per_m"]), kyy=float(b["kyy_N_per_m"]),
            cxx=float(b["cxx_Ns_per_m"]), cyy=float(b["cyy_Ns_per_m"]),
        )
        for b in r["bearings"]
    ]

    return rs.Rotor(shaft, disks, bearings)


def critical_speeds(rotor, max_rpm: float = 15000, n: int = 3):
    """First undamped critical speeds via ROSS run_critical_speed (ROSS 2.x)."""
    crit = rotor.run_critical_speed(num_modes=2 * n)
    try:
        rpm = np.asarray(crit.wn("rpm"), dtype=float)   # ROSS 2.x: wn() is a method
    except TypeError:
        rpm = np.asarray(crit.wn, dtype=float) * 60 / (2 * np.pi)
    rpm = np.sort(rpm[rpm > 0])
    in_range = rpm[rpm <= max_rpm]                       # criticals within operating range
    return {"critical_rpm": in_range if in_range.size else rpm, "crit": crit}


def unbalance_response(rotor, node: int, mag_kg_m: float, phase_rad: float,
                       rpm_range: np.ndarray):
    """
    Synchronous (1X) forced response to an unbalance at `node`.
    mag_kg_m is the unbalance magnitude m*e in kg*m. Returns the ROSS results
    object (Bode/orbit data) used to derive influence coefficients.
    """
    speed_rad = rpm_range * 2 * np.pi / 60
    return rotor.run_unbalance_response(
        node=node, unbalance_magnitude=mag_kg_m,
        unbalance_phase=phase_rad, frequency=speed_rad,
    )


def predicted_influence_coefficients(rotor, balancing_nodes, sensor_nodes,
                                     rpm: float, trial_kg_m: float = 1e-4):
    """
    Model-predicted influence-coefficient matrix A (sensor x balancing).

    A[i, j] = complex 1X response at sensor i per unit unbalance at plane j.
    Compared against the *experimental* ICM (estimation/icm.py) this validates
    the twin and gives a physics prior for the unbalance estimate.
    """
    rpm_range = np.array([rpm])
    A = np.zeros((len(sensor_nodes), len(balancing_nodes)), dtype=complex)
    for j, bn in enumerate(balancing_nodes):
        res = unbalance_response(rotor, bn, trial_kg_m, 0.0, rpm_range)
        # extract complex 1X amplitude at each sensor node from ROSS results
        for i, sn in enumerate(sensor_nodes):
            resp = res.forced_resp[sn, 0]  # complex displacement at 1X
            A[i, j] = resp / trial_kg_m
    return A
