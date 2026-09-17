"""
Keyphasor-based order tracking.

The keyphasor gives one pulse per revolution. We (1) detect pulse instants,
(2) resample each vibration channel onto a uniform *angular* grid, and
(3) extract complex harmonic vectors (1X, 2X, 3X ...) whose phase is referenced
to the keyphasor. Angular resampling removes speed-jitter smearing, giving clean
amplitude AND phase — essential for the Influence Coefficient Method and for
distinguishing imbalance (1X) from misalignment (2X).
"""
from __future__ import annotations

import numpy as np


def detect_pulses(kp: np.ndarray, fs: float, threshold: float,
                  edge: str = "rising") -> np.ndarray:
    """Return keyphasor pulse times (seconds) by threshold crossing."""
    above = kp > threshold
    crossings = np.diff(above.astype(int))
    idx = np.where(crossings == (1 if edge == "rising" else -1))[0] + 1
    # sub-sample refinement by linear interpolation across the threshold
    times = []
    for i in idx:
        y0, y1 = kp[i - 1], kp[i]
        frac = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
        times.append((i - 1 + frac) / fs)
    return np.asarray(times)


def instantaneous_rpm(pulse_times: np.ndarray, pulses_per_rev: int = 1) -> np.ndarray:
    """RPM between consecutive pulses (length = n_pulses - 1)."""
    dt = np.diff(pulse_times)
    rev_per_s = 1.0 / (dt * pulses_per_rev)
    return rev_per_s * 60.0


def angular_resample(signal: np.ndarray, fs: float, pulse_times: np.ndarray,
                     samples_per_rev: int) -> np.ndarray:
    """
    Resample `signal` onto a uniform angle grid using the keyphasor pulses as
    the angular reference (integer revolutions between pulses). Returns an array
    of shape (n_revs, samples_per_rev).
    """
    if len(pulse_times) < 2:
        raise ValueError("need >= 2 keyphasor pulses for order tracking")
    t = np.arange(signal.size) / fs
    revs = []
    for r in range(len(pulse_times) - 1):
        t0, t1 = pulse_times[r], pulse_times[r + 1]
        # uniform angle -> uniform time within a rev (const speed per rev)
        t_grid = np.linspace(t0, t1, samples_per_rev, endpoint=False)
        revs.append(np.interp(t_grid, t, signal))
    return np.asarray(revs)


def extract_orders(resampled: np.ndarray, orders=(1, 2, 3)) -> dict[int, complex]:
    """
    Complex harmonic vector per order, averaged over revolutions.

    Phase convention: 0 rad = keyphasor pulse (start of each revolution).
    Returns {order: amplitude*exp(j*phase)} where amplitude is 0-peak in the
    signal's own units.
    """
    revs, spr = resampled.shape
    theta = np.arange(spr) / spr * 2 * np.pi
    out: dict[int, complex] = {}
    for k in orders:
        # synchronous vector filter over each rev, then average
        vec = np.mean(resampled * np.exp(-1j * k * theta), axis=1)
        out[k] = 2.0 * np.mean(vec)   # factor 2 -> 0-peak amplitude
    return out


def harmonic_vectors(signal: np.ndarray, fs: float, kp: np.ndarray,
                     cfg: dict) -> dict[int, complex]:
    """End-to-end: raw signal + keyphasor -> {order: complex vector}."""
    kcfg = cfg["daq"]["keyphasor"]
    ot = cfg["order_tracking"]
    pulses = detect_pulses(kp, fs, kcfg["threshold_v"], kcfg["edge"])
    resampled = angular_resample(signal, fs, pulses, ot["samples_per_rev"])
    return extract_orders(resampled, tuple(ot["orders"]))


def amp_phase(vec: complex) -> tuple[float, float]:
    """Convenience: (amplitude, phase_degrees) from a complex harmonic vector."""
    return abs(vec), np.degrees(np.angle(vec))
