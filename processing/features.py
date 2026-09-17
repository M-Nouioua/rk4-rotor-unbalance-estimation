"""
Feature extraction for the RK-4 digital twin.

Turns one raw acquisition (keyphasor + 4 proximity channels) into a compact,
*physics-informed* feature vector that both the physics ICM and the data-driven
(ML) estimators consume. Everything here is derived from the same keyphasor-based
order tracking used by the ICM, so the two estimators see a consistent view of the
physics — the only difference is the inverse map (linear A^-1 vs learned regressor).

Per acquisition we compute, for each of the four sensor channels:
  * runout-compensated complex 1X/2X/3X vectors (mils, keyphasor-referenced),
  * amplitude / phase and harmonic ratios (2X/1X, 3X/1X),
  * baseline-subtracted 1X ("response to the added mass") when a baseline is given.
Per measurement plane (X+Y pair) we add orbit-ellipse descriptors (semi-major /
minor axis, ellipticity, whirl direction, tilt) from the forward/backward
decomposition of the 1X phasors, plus cross-plane amplitude ratios and phase
differences that carry the plane-localization information.

Conventions match `scripts/analyze.py`:
  * amplitudes in mils (via the probe sensitivity in cfg),
  * WHIRL_CONJ conjugates the measured vectors so measured phase runs with the
    mounting-angle sense (fixes the +/-90 flips), and
  * runout (mechanical/electrical, speed-independent) is subtracted as a complex
    1X vector captured at slow roll.
"""
from __future__ import annotations

import numpy as np

from processing.order_tracking import harmonic_vectors, detect_pulses, instantaneous_rpm

SENSORS = ["plane1_x", "plane1_y", "plane2_x", "plane2_y"]
PLANES = {"plane1": ("plane1_x", "plane1_y"), "plane2": ("plane2_x", "plane2_y")}
WHIRL_CONJ = True   # keep in lock-step with scripts/analyze.py


def to_mils_factor(cfg: dict) -> float:
    """Volts -> mils using the probe sensitivity (mV/um)."""
    s = cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"]
    return 1000.0 / (s * 25.4)


def _orders(signal, fs, kp, cfg):
    """Complex {order: vector} in mils, whirl-corrected."""
    h = harmonic_vectors(signal, fs, kp, cfg)
    f = to_mils_factor(cfg)
    out = {k: v * f for k, v in h.items()}
    if WHIRL_CONJ:
        out = {k: np.conj(v) for k, v in out.items()}
    return out


def runout_vectors(cfg: dict, runout_npz) -> dict:
    """Complex 1X/2X/3X runout reference per sensor (mils), from a slow-roll .npz."""
    d = runout_npz
    fs = float(d["fs"]); kp = d["keyphasor"]
    return {p: _orders(d[p], fs, kp, cfg) for p in SENSORS}


def acquisition_vectors(npz, cfg: dict, runout: dict | None = None) -> dict:
    """
    Runout-compensated complex 1X/2X/3X per sensor for one acquisition.

    Returns {sensor: {order: complex mils}} plus the measured rpm under key "rpm".
    Runout is subtracted order-by-order (speed-independent geometric/electrical
    artifact); if `runout` is None nothing is subtracted.
    """
    fs = float(npz["fs"]); kp = npz["keyphasor"]
    out = {}
    for p in SENSORS:
        v = _orders(npz[p], fs, kp, cfg)
        if runout is not None:
            v = {k: v[k] - runout[p].get(k, 0) for k in v}
        out[p] = v
    pulses = detect_pulses(kp, fs, cfg["daq"]["keyphasor"]["threshold_v"],
                           cfg["daq"]["keyphasor"]["edge"])
    rpm = float(np.median(instantaneous_rpm(pulses))) if len(pulses) > 2 else np.nan
    out["rpm"] = rpm
    return out


def _orbit(X: complex, Y: complex) -> dict:
    """
    Orbit-ellipse descriptors from the 1X phasors of an X/Y probe pair.

    Forward/backward decomposition: z(t)=x+jy = F e^{jwt} + B e^{-jwt} with
    F=(X+jY)/2, B=(X-jY)/2 (|B|=backward). semi-major = |F|+|B|,
    semi-minor = ||F|-|B||, ellipticity in [0,1], whirl>0 forward. tilt = major-axis
    orientation in the probe frame (deg).
    """
    F = (X + 1j * Y) / 2.0
    B = (X - 1j * Y) / 2.0
    rf, rb = abs(F), abs(B)
    smaj = rf + rb
    smin = abs(rf - rb)
    ecc = smin / smaj if smaj else 0.0
    whirl = 1.0 if rf >= rb else -1.0
    tilt = np.degrees((np.angle(F) + np.angle(B)) / 2.0) % 180.0
    return {"smajor": smaj, "sminor": smin, "ellipticity": ecc,
            "whirl": whirl, "tilt_deg": tilt}


def feature_row(npz, cfg: dict, runout: dict | None = None,
                baseline: dict | None = None) -> dict:
    """
    Full flat feature dict for one acquisition.

    `baseline` (same structure as acquisition_vectors) enables the
    baseline-subtracted 1X "response" features — the physical signature of the
    *added* mass, which is what maps most directly to the added-unbalance labels.
    """
    v = acquisition_vectors(npz, cfg, runout)
    row = {"rpm": v["rpm"]}

    # per-sensor harmonic features
    for p in SENSORS:
        o = v[p]
        one = o.get(1, 0j); two = o.get(2, 0j); three = o.get(3, 0j)
        row[f"{p}_1x_re"] = one.real
        row[f"{p}_1x_im"] = one.imag
        row[f"{p}_1x_amp"] = abs(one)
        row[f"{p}_1x_phase"] = np.degrees(np.angle(one))
        row[f"{p}_2x_amp"] = abs(two)
        row[f"{p}_3x_amp"] = abs(three)
        row[f"{p}_2x_ratio"] = abs(two) / abs(one) if abs(one) > 1e-9 else 0.0
        row[f"{p}_3x_ratio"] = abs(three) / abs(one) if abs(one) > 1e-9 else 0.0

    # per-plane orbit descriptors (from 1X phasors)
    for plane, (xs, ys) in PLANES.items():
        orb = _orbit(v[xs].get(1, 0j), v[ys].get(1, 0j))
        for k, val in orb.items():
            row[f"{plane}_{k}"] = val

    # cross-plane localization cues
    a1 = row["plane1_smajor"]; a2 = row["plane2_smajor"]
    row["plane_amp_ratio"] = a2 / a1 if a1 > 1e-9 else 0.0
    row["plane_phase_diff"] = (row["plane2_x_1x_phase"] - row["plane1_x_1x_phase"]
                               + 180) % 360 - 180

    # baseline-subtracted 1X response (added-mass signature)
    if baseline is not None:
        for p in SENSORS:
            resp = v[p].get(1, 0j) - baseline[p].get(1, 0j)
            row[f"{p}_resp_re"] = resp.real
            row[f"{p}_resp_im"] = resp.imag
            row[f"{p}_resp_amp"] = abs(resp)
    return row
