"""
Day-1 run-up / coast-down analysis: identify the first critical speed, damping, and
the synchronous (1X) Bode + polar response — the data that seeds the ROSS model update
and fixes the dwell speeds N1, N2.

Method: keyphasor gives one pulse per revolution. For EACH revolution we angularly
resample and extract the complex 1X vector (vector filter), and pair it with that
revolution's instantaneous speed. Sorting by rpm gives amplitude/phase vs speed with
no spectral leakage. The critical speed is the amplitude peak (and the ~-90 deg phase
crossing); damping from the half-power bandwidth.

    python -m scripts.runup_critical --sim                      # synthetic self-test
    python -m scripts.runup_critical --file datasets/runup.npz  # real record
"""
from __future__ import annotations

import argparse
import numpy as np

from config import load_config
from processing.order_tracking import detect_pulses, instantaneous_rpm, angular_resample


def per_rev_1x(signal, fs, pulses, spr):
    """Complex 1X vector per revolution + that rev's mean rpm."""
    resampled = angular_resample(signal, fs, pulses, spr)      # (n_rev, spr)
    theta = np.arange(spr) / spr * 2 * np.pi
    vec = 2.0 * np.mean(resampled * np.exp(-1j * theta), axis=1)  # (n_rev,)
    rpm = instantaneous_rpm(pulses)                            # (n_rev,)
    return rpm, vec


def moving_avg(x, k):
    if k < 2:
        return x
    kern = np.ones(k) / k
    return np.convolve(x, kern, mode="same")


def identify_critical(rpm, vec, smooth=15):
    """Return dict: Nc (peak), amplitude, damping ratio, Q, half-power band."""
    order = np.argsort(rpm)
    rpm, vec = rpm[order], vec[order]
    amp = np.abs(vec)
    amp_s = moving_avg(amp, smooth)
    ipk = int(np.argmax(amp_s))
    Nc = float(rpm[ipk]); peak = float(amp_s[ipk])
    half = peak / np.sqrt(2.0)

    # half-power crossings on each side of the peak
    def cross(idx_range):
        for i in idx_range:
            if amp_s[i] < half:
                return float(rpm[i])
        return None
    f1 = cross(range(ipk, -1, -1))          # searching downward in speed
    f2 = cross(range(ipk, len(rpm)))        # searching upward
    zeta = Q = bw = None
    if f1 and f2 and Nc > 0:
        bw = f2 - f1
        zeta = bw / (2.0 * Nc)
        Q = Nc / bw if bw else None
    return {"Nc_rpm": Nc, "peak_amp": peak, "f1": f1, "f2": f2,
            "bandwidth_rpm": bw, "zeta": zeta, "Q": Q,
            "rpm": rpm, "amp": amp, "phase_deg": np.degrees(np.angle(vec))}


def synth_runup(fs=12800, duration=20.0, rpm0=600, rpm1=5000,
                Nc=2400, zeta=0.04, ecc=1.0, seed=1):
    """Synthetic SDOF rotating-unbalance run-up for offline self-test."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    rpm = rpm0 + (rpm1 - rpm0) * t / duration
    f = rpm / 60.0
    phase = 2 * np.pi * np.cumsum(f) / fs           # integrated angle
    fn = Nc / 60.0
    r = f / fn
    M = r**2 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)   # unbalance FRF magnitude
    lag = np.arctan2(2 * zeta * r, 1 - r**2)               # phase lag
    x = ecc * M * np.sin(phase - lag) + 0.01 * rng.standard_normal(n)
    y = ecc * M * np.sin(phase - lag - np.pi / 2) + 0.01 * rng.standard_normal(n)
    kp = -5.0 * ((phase / (2 * np.pi)) % 1.0 < 0.01).astype(float)   # 1 neg pulse/rev
    return {"keyphasor": kp, "plane1_x": x, "plane1_y": y, "fs": fs, "_true_Nc": Nc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="npz with keyphasor + plane1_x (from run_baseline --out)")
    ap.add_argument("--channel", default="plane1_x")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--plot", action="store_true", help="save Bode + polar PNGs to analysis/")
    ap.add_argument("--out", default="datasets/runup_1x.csv")
    a = ap.parse_args()

    cfg = load_config()
    if a.sim:
        rec = synth_runup()
    elif a.file:
        d = np.load(a.file)
        rec = {k: d[k] for k in d.files}
    else:
        raise SystemExit("provide --file <npz> or --sim")

    fs = float(rec["fs"])
    kp = rec["keyphasor"]
    kc = cfg["daq"]["keyphasor"]
    pulses = detect_pulses(kp, fs, kc["threshold_v"], kc["edge"])
    spr = cfg["order_tracking"]["samples_per_rev"]
    rpm, vec = per_rev_1x(rec[a.channel], fs, pulses, spr)
    res = identify_critical(rpm, vec)

    print(f"revolutions analysed : {len(rpm)}")
    print(f"speed span           : {rpm.min():.0f} - {rpm.max():.0f} rpm")
    print(f"FIRST CRITICAL (Nc)  : {res['Nc_rpm']:.0f} rpm  (1X amplitude peak)")
    if res["zeta"] is not None:
        print(f"half-power band      : {res['f1']:.0f} - {res['f2']:.0f} rpm  (Δ={res['bandwidth_rpm']:.0f})")
        print(f"damping ratio ζ      : {res['zeta']:.4f}    Q ≈ {res['Q']:.1f}")
    else:
        print("damping: half-power points not both found (widen the speed sweep)")
    if "_true_Nc" in rec:
        print(f"[sim] true Nc = {rec['_true_Nc']} rpm  -> error {res['Nc_rpm']-rec['_true_Nc']:+.0f} rpm")
    print(f"\nsuggested dwell speeds:  N1 = {0.5*res['Nc_rpm']:.0f} rpm (0.5 Nc),  "
          f"N2 = {0.8*res['Nc_rpm']:.0f} rpm (0.8 Nc)")

    # save Bode table
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    order = np.argsort(rpm)
    np.savetxt(a.out, np.column_stack([rpm[order], np.abs(vec)[order],
                                       np.degrees(np.angle(vec))[order]]),
               delimiter=",", header="rpm,amp_1x,phase_1x_deg", comments="")
    print(f"1X Bode table -> {a.out}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs("analysis", exist_ok=True)
            rs = rpm[order]; amp = np.abs(vec)[order]; ph = np.degrees(np.angle(vec))[order]
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
            ax[0].plot(rs, amp); ax[0].axvline(res["Nc_rpm"], ls="--", c="r")
            ax[0].set_ylabel("1X amplitude"); ax[0].set_title(f"Run-up Bode — Nc≈{res['Nc_rpm']:.0f} rpm")
            ax[1].plot(rs, np.unwrap(np.radians(ph)) * 180 / np.pi)
            ax[1].axvline(res["Nc_rpm"], ls="--", c="r")
            ax[1].set_ylabel("1X phase [deg]"); ax[1].set_xlabel("speed [rpm]")
            fig.tight_layout(); fig.savefig("analysis/runup_bode.png", dpi=130)
            fig2, axp = plt.subplots(figsize=(5, 5))
            axp.plot(vec.real, vec.imag, ".", ms=2); axp.axhline(0, c="k", lw=.5); axp.axvline(0, c="k", lw=.5)
            axp.set_aspect("equal"); axp.set_title("1X polar (Nyquist)")
            fig2.tight_layout(); fig2.savefig("analysis/runup_polar.png", dpi=130)
            print("plots -> analysis/runup_bode.png, analysis/runup_polar.png")
        except ImportError:
            print("matplotlib not installed; skipped plots (CSV still written)")


if __name__ == "__main__":
    main()
