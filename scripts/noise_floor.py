"""
Stationary noise-floor characterisation (rotor OFF).

Establishes the practical detection limit for small unbalance and reveals
electrical interference (mains hum, ground loops). Run with the rotor stopped:

    python -m scripts.noise_floor --duration 4

Per channel: RMS in V and (for probes) in microns / mils, plus the strongest
spectral lines so you can spot 50/60 Hz pickup vs broadband noise.
"""
from __future__ import annotations

import argparse
import numpy as np

from scipy.signal import butter, filtfilt

from config import load_config
from acquisition.daq import acquire_record, CHANNEL_ORDER
from processing.spectra import fft_spectrum


def inband_rms(x, fs, fcut=1000.0):
    """RMS after a low-pass at fcut Hz — the noise that matters for unbalance."""
    b, a = butter(4, fcut, btype="low", fs=fs)
    return float(np.sqrt(np.mean(filtfilt(b, a, x) ** 2)))


def top_lines(freqs, amp, n=3, fmin=2.0):
    """Indices of the n strongest spectral lines above fmin Hz."""
    mask = freqs >= fmin
    idx = np.where(mask)[0]
    order = idx[np.argsort(amp[idx])[::-1][:n]]
    return [(float(freqs[i]), float(amp[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()

    cfg = load_config()
    rec = acquire_record(cfg, args.duration)
    fs = rec["fs"]
    chans = cfg["daq"]["channels"]
    print(f"stationary noise floor — {args.duration:.0f} s @ {fs} S/s\n")

    for name in CHANNEL_ORDER:
        x = np.asarray(rec[name], dtype=float)
        rms = float(np.sqrt(np.mean(x ** 2)))
        sens = chans[name].get("sensitivity_mV_per_um")
        unit = ""
        if sens:
            um = rms * 1000.0 / sens
            ib = inband_rms(x, fs) * 1000.0 / sens          # <1 kHz, in µm
            unit = (f"  = {um:6.3f} µm rms  |  IN-BAND(<1kHz): "
                    f"{ib:6.3f} µm = {ib/25.4:6.4f} mils rms")
        freqs, amp = fft_spectrum(x, fs)
        lines = top_lines(freqs, amp)
        peaks = "  ".join(f"{f:6.1f}Hz({a*1000:.1f}mV)" for f, a in lines)
        print(f"{name:10s} {rms:8.4f} V rms{unit}")
        print(f"           top lines: {peaks}")

    print("\nInterpretation: a strong 50/60 Hz line (or its harmonics) = mains "
          "pickup / ground loop -> check cable shields & a common ground. A high "
          "broadband RMS with no dominant line = a floating/ungapped probe or a "
          "bad Proximitor channel.")


if __name__ == "__main__":
    main()
