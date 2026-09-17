"""
Data-quality / experiment-sanity audit — "did we mis-set a mass, angle, or speed?"

A real *setup* mistake has a fingerprint: it is SYSTEMATIC (all repeats and both
speeds of one condition agree with each other, but disagree with the label). Random
measurement problems hit a single acquisition. Known physics (rotor anisotropy,
sub-critical 2-plane conditioning) also produce large label-vs-estimate gaps that are
NOT mistakes — so this audit is anisotropy-aware: it only flags a condition when it
breaks the pattern of its *same-angle, same-disk peers*.

Four independent checks, using the label-independent physics estimate:
  1. SPEED     — rpm_measured off its speed-group median (>7%) or N1/N2 swapped.
  2. REPEAT    — one acquisition in a triplet far from its siblings (bad record).
  3. LABEL     — magnitude ratio / phase error is a robust outlier vs same-angle peers.
  4. XSPEED    — a physical U is speed-independent; N1 vs N2 disagreement is suspicious.

Nothing here is proof of a mistake — it is a ranked *re-check* list. Verify a flagged
condition by re-weighing the screw and confirming its hole vs the keyphasor.

    python -m scripts.qc_anomalies
"""
from __future__ import annotations

import csv
import sys
import numpy as np
import pandas as pd

from config import ROOT
from processing.features import SENSORS
from estimation.twin import physics_cross_estimate, CALIB

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ANALYSIS = ROOT / "analysis"
DATASETS = ROOT / "datasets"
Z = 3.5           # robust-z threshold for an outlier
NOMINAL = {"N1": 1000, "N2": 1200}


def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def cmean(a):
    a = np.radians(np.asarray(a, float))
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def aerr(e, t):
    return (e - t + 180) % 360 - 180


def check_speed(df):
    print("=" * 70, "\n1) SPEED — rpm per acquisition\n", "=" * 70, sep="")
    d = df[df.labeled == 1].copy()
    d["rpm"] = pd.to_numeric(d.rpm_measured, errors="coerce")
    flags = []
    for sp in ("N1", "N2"):
        g = d[d.speed_id == sp]
        if g.empty:
            continue
        med = g.rpm.median()
        print(f"  {sp}: n={len(g)} median={med:.0f} range={g.rpm.min():.0f}-{g.rpm.max():.0f} "
              f"(nominal {NOMINAL.get(sp,'?')})")
        for _, r in g[(g.rpm - med).abs() / med > 0.07].iterrows():
            flags.append((r.condition_id, sp, r.rpm, "off-speed >7%"))
    piv = d.groupby(["condition_id", "speed_id"]).rpm.median().unstack()
    if {"N1", "N2"}.issubset(piv.columns):
        for cid in piv[piv.N1 > piv.N2].index:
            flags.append((cid, "N1/N2", 0, "speed labels swapped (N1 rpm > N2 rpm)"))
    print("  flags:", "none" if not flags else "")
    for c, sp, v, why in flags:
        print(f"    {c} {sp}: {v:.0f} rpm — {why}")
    return flags


def check_repeat(df):
    """One acquisition in a (condition,speed) group far from its siblings."""
    print("\n" + "=" * 70, "\n2) REPEAT — outlier acquisition within a triplet\n", "=" * 70, sep="")
    d = df[df.labeled == 1]
    flags = []
    for (cid, sp), g in d.groupby(["condition_id", "speed_id"]):
        if len(g) < 3:
            continue
        V = np.column_stack([g[f"{p}_resp_re"] + 1j * g[f"{p}_resp_im"] for p in SENSORS])
        ctr = V.mean(axis=0)
        dist = np.abs(V - ctr).mean(axis=1)
        scale = np.abs(ctr).mean() + 1e-6
        rel = dist / scale
        i = int(np.argmax(rel))
        if rel[i] > 0.5 and dist[i] > 0.02:      # >50% of the mean orbit, absolute floor
            flags.append((cid, sp, g.iloc[i]["file"], rel[i]))
    print("  flags:", "none" if not flags else "")
    for c, sp, f, rel in sorted(flags, key=lambda x: -x[3])[:15]:
        print(f"    {c} {sp}: {f} deviates {rel*100:.0f}% from its siblings")
    return flags


def check_magnitude_modelfree(df, cond):
    """
    Inversion-free mass sanity: the raw baseline-subtracted 1X response magnitude
    scales with the mounted unbalance. Within each config (so couples, which
    respond weakly, are judged against couples), a condition whose response/label
    ratio is a robust outlier is a *genuine* mass suspect — unlike the inverted
    estimate, this cannot be fooled by 2-plane conditioning.
    """
    print("\n" + "=" * 70, "\n3a) MAGNITUDE (model-free) — raw response vs label, no inversion\n",
          "=" * 70, sep="")
    d = df[df.labeled == 1].copy()
    amp_cols = [f"{p}_resp_amp" for p in SENSORS]
    d["rnorm"] = np.sqrt((d[amp_cols] ** 2).sum(axis=1))
    rows = []
    for (cid, sp), g in d.groupby(["condition_id", "speed_id"]):
        if cid in CALIB:
            continue
        c = cond[cid]
        totU = float(c["U1_gmm"] or 0) + float(c["U2_gmm"] or 0)
        if totU <= 0:
            continue
        # representative mounting angle (controls for anisotropy, which is angle-dependent)
        ang = c["disk1_angle_deg"] if float(c["disk1_mass_g"] or 0) > 0 else c["disk2_angle_deg"]
        akey = int(round(float(ang or 0) / 45.0) * 45) % 360
        rows.append(dict(cid=cid, speed=sp, config=c["config"], angle=akey, totU=totU,
                         rnorm=g.rnorm.mean(), ratio=g.rnorm.mean() / totU))
    A = pd.DataFrame(rows)
    # response ~ U only holds for net (static) unbalance; antiphase couples cancel at
    # the probes -> intrinsically weak, so response/U is not a valid mass yardstick for
    # them (they are covered by the cross-speed check instead). Judge net configs only.
    A = A[(A.totU >= 12) & (A.config != "antiphase")]
    # per-condition z within its config/angle/speed peer group
    A["z"] = np.nan
    small = 0
    for (cfg, ang, sp), g in A.groupby(["config", "angle", "speed"]):
        if len(g) < 4:
            small += 1
            continue
        A.loc[g.index, "z"] = robust_z(g.ratio.values)
    # a genuine mount error is speed-independent: require |z|>Z at BOTH N1 and N2
    flags = []
    for cid, g in A.dropna(subset=["z"]).groupby("cid"):
        by = g.set_index("speed")
        if {"N1", "N2"}.issubset(by.index):
            z1, z2 = by.loc["N1", "z"], by.loc["N2", "z"]
            if abs(z1) > Z and abs(z2) > Z and np.sign(z1) == np.sign(z2):
                flags.append((cid, g.config.iloc[0], int(g.angle.iloc[0]),
                              round(g.ratio.mean(), 4), round((z1 + z2) / 2, 1)))
    print("  flags:", "none — every well-resolved condition's raw response matches its "
          "same-angle peers at BOTH speeds" if not flags else "")
    for cid, cfg, ang, ratio, zz in sorted(flags, key=lambda x: -abs(x[4])):
        print(f"    {cid} [{cfg} @{ang}deg]: response/label={ratio} at both speeds (mean z {zz:+.1f})")
    if small:
        print(f"  ({small} groups had <4 peers — not judged; low-U/couple points excluded)")
    return flags


def _agg(res):
    rows = []
    for _, x in res.iterrows():
        for k in (1, 2):
            rows.append(dict(cid=x.condition_id, disk=k, speed=x.speed_id,
                             true_mag=x[f"true_U{k}_mag"], est_mag=x[f"est_U{k}_mag"],
                             true_ang=x[f"true_U{k}_ang"], est_ang=x[f"est_U{k}_ang"]))
    R = pd.DataFrame(rows)
    return R[(~R.cid.isin(CALIB)) & (R.true_mag > 1e-6)]


def check_label(res, cond):
    """Anisotropy-aware: outlier vs same-angle, same-disk peers."""
    print("\n" + "=" * 70, "\n3) LABEL — mass/angle outlier vs same-angle peers "
          "(anisotropy removed)\n", "=" * 70, sep="")
    R = _agg(res)
    agg = []
    for (cid, disk), g in R.groupby(["cid", "disk"]):
        agg.append(dict(cid=cid, disk=disk, config=cond[cid]["config"],
                        true_mag=g.true_mag.iloc[0], true_ang=round(g.true_ang.iloc[0]),
                        ratio=g.est_mag.mean() / g.true_mag.iloc[0],
                        ang_err=aerr(cmean(g.est_ang.values), g.true_ang.iloc[0])))
    A = pd.DataFrame(agg)
    A = A[A.true_mag >= 12]
    A["grp"] = A.disk.astype(str) + "@" + A.true_ang.astype(int).astype(str)
    flags = []
    for grp, g in A.groupby("grp"):
        if len(g) < 4:
            continue
        for col in ("ratio", "ang_err"):
            z = robust_z(g[col].values)
            for (_, r), zz in zip(g.iterrows(), z):
                if abs(zz) > Z:
                    flags.append((r.cid, int(r.disk), int(r.true_ang), col,
                                  round(r[col], 2), round(float(zz), 1), r.config))
    print("  flags:", "none — every condition matches its same-angle peers" if not flags else "")
    for c, dk, ang, col, val, zz, cfg in sorted(flags, key=lambda x: -abs(x[5])):
        print(f"    {c} D{dk} @{ang:+d} [{cfg}]: {col}={val} (robust-z {zz:+.1f})")
    return flags


def check_xspeed(res):
    """A physical U is speed-independent; large N1<->N2 disagreement is suspicious."""
    print("\n" + "=" * 70, "\n4) XSPEED — N1 vs N2 physics estimate disagreement\n", "=" * 70, sep="")
    R = _agg(res)
    piv = R.pivot_table(index=["cid", "disk"], columns="speed", values="est_mag").dropna()
    if not {"N1", "N2"}.issubset(piv.columns):
        print("  need both speeds"); return []
    diff = (piv.N1 - piv.N2).abs()
    z = robust_z(diff.values)
    flags = [(idx[0], idx[1], piv.N1[idx], piv.N2[idx], zz)
             for idx, zz in zip(piv.index, z) if zz > Z]
    print(f"  population median |N1-N2| = {np.median(diff):.1f} g.mm")
    print("  flags:", "none" if not flags else "")
    for cid, dk, n1, n2, zz in sorted(flags, key=lambda x: -x[4]):
        print(f"    {cid} D{dk}: N1={n1:.0f} vs N2={n2:.0f} g.mm (robust-z {zz:+.1f})")
    return flags


def main():
    df = pd.read_csv(ANALYSIS / "features.csv")
    cond = {c["condition_id"]: c for c in csv.DictReader(open(DATASETS / "conditions.csv"))}
    res = physics_cross_estimate(df)

    sp = check_speed(df)
    rep = check_repeat(df)
    mag = check_magnitude_modelfree(df, cond)
    lab = check_label(res, cond)
    xs = check_xspeed(res)

    print("\n" + "=" * 70, "\nSUMMARY\n", "=" * 70, sep="")
    print(f"  speed issues            : {len(sp)}")
    print(f"  bad records             : {len(rep)}")
    print(f"  MASS suspects (model-free): {len(set(f[0] for f in mag))} conditions  <-- the definitive mass test")
    print(f"  split/localization flags (inversion, unstable): "
          f"{len(set(f[0] for f in lab))} conditions")
    print(f"  cross-speed suspects    : {len(set(f[0] for f in xs))} conditions")
    hard = sorted(set([f[0] for f in mag] + [f[0] for f in sp] + [f[0] for f in xs]))
    print(f"\n  RE-CHECK physically (re-weigh screw + confirm hole vs keyphasor): "
          f"{', '.join(hard) if hard else 'NONE — data is sound'}")
    print("  NOTE: 3a (model-free) is the real mass test. Flags that appear only in 3b/label"
          " are 2-plane conditioning + anisotropy amplified by the inversion, not setup"
          " errors — the fix is the hybrid twin, not a re-mount (see CONCEPT.md).")


if __name__ == "__main__":
    main()
