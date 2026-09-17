"""
Spectral & orbit features for fault discrimination.

- FFT spectrum (windowed, 0-peak scaled)
- Orbit (X vs Y filtered to 1X or 2X, or raw)
- Full spectrum: forward/backward whirl decomposition from an XY probe pair.
  The forward/backward asymmetry is a strong misalignment discriminator.
"""
from __future__ import annotations

import numpy as np


def fft_spectrum(signal: np.ndarray, fs: float, window: str = "hann"):
    """Return (freqs_hz, amplitude_0pk)."""
    n = signal.size
    w = np.hanning(n) if window == "hann" else np.ones(n)
    coherent_gain = w.mean()
    X = np.fft.rfft(signal * w)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.abs(X) / n / coherent_gain * 2.0
    return freqs, amp


def bandpass_at_order(signal: np.ndarray, fs: float, rpm: float, order: float,
                      bw_frac: float = 0.1) -> np.ndarray:
    """Zero-phase band-pass around a running-speed order (for orbit filtering)."""
    from scipy.signal import butter, filtfilt

    f0 = rpm / 60.0 * order
    lo = max(f0 * (1 - bw_frac), 0.1)
    hi = min(f0 * (1 + bw_frac), fs / 2 - 1)
    b, a = butter(4, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, signal)


def orbit(x: np.ndarray, y: np.ndarray, fs: float, rpm: float,
          order: float | None = 1.0):
    """
    Return (x, y) filtered for an orbit plot. order=None -> raw (unfiltered)
    orbit. order=1 -> imbalance-like ellipse; order=2 -> misalignment banana.
    """
    if order is None:
        return x, y
    return (bandpass_at_order(x, fs, rpm, order),
            bandpass_at_order(y, fs, rpm, order))


def full_spectrum(x: np.ndarray, y: np.ndarray, fs: float):
    """
    Full spectrum from an orthogonal probe pair.

    Combine X and Y into the analytic rotating vector z = x + j*y and take its
    FFT: positive frequencies = forward whirl, negative = backward whirl.
    Returns (freqs_signed_hz, amplitude).
    """
    n = x.size
    w = np.hanning(n)
    z = (x + 1j * y) * w
    Z = np.fft.fftshift(np.fft.fft(z)) / n * 2.0
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    return freqs, np.abs(Z)


def diagnostic_features(orders_p1: dict, orders_p2: dict) -> dict:
    """
    Compact per-run feature vector from the harmonic dicts of both planes.
    Keys chosen for imbalance vs misalignment separability.
    """
    def amp(d, k):
        return abs(d.get(k, 0.0))

    feats = {}
    for tag, d in (("p1", orders_p1), ("p2", orders_p2)):
        feats[f"{tag}_1x"] = amp(d, 1)
        feats[f"{tag}_2x"] = amp(d, 2)
        feats[f"{tag}_3x"] = amp(d, 3)
        feats[f"{tag}_2x_over_1x"] = amp(d, 2) / amp(d, 1) if amp(d, 1) else 0.0
    return feats
