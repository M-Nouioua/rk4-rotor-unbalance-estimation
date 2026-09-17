"""
Publication-quality figure set for the digital-twin paper.

Reads the benchmark outputs (estimates_long.csv, benchmark.json,
feature_importance.csv, features.csv) and renders a coherent, accessible figure
suite to analysis/. Colors follow a CVD-validated 3-slot categorical palette
(physics=blue, ml=orange, hybrid=aqua); marks are thin, grids recessive.

    python -m scripts.benchmark   # first, to refresh the data
    python -m scripts.figures
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT

ANALYSIS = ROOT / "analysis"

# --- CVD-validated categorical slots 1-3 (light surface) ---
C = {"physics": "#2a78d6", "ml": "#eb6834", "hybrid": "#1baf7a"}
LABEL = {"physics": "Physics (ICM)", "ml": "ML (Extra-Trees)", "hybrid": "Hybrid"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
ORDER = ["physics", "ml", "hybrid"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#c3c2b7",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
})


def _load():
    L = pd.read_csv(ANALYSIS / "estimates_long.csv")
    B = json.load(open(ANALYSIS / "benchmark.json"))
    return L, B


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(ANALYSIS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def fig_scatter(L, B):
    """Estimated vs true added unbalance, one panel per method (loaded points)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    lim = 66
    for ax, m in zip(axes, ORDER):
        d = L[(L.method == m) & (L.loaded)]
        for disk, mk in [(1, "o"), (2, "^")]:
            s = d[d.disk == disk]
            ax.scatter(s.true_mag, s.est_mag, s=24, marker=mk, alpha=.6,
                       color=C[m], edgecolor="white", linewidth=.4,
                       label=f"disk {disk}")
        ax.plot([0, lim], [0, lim], "--", color=MUTED, lw=1)
        o = B["methods"][m]["overall"]
        ax.text(.04, .96, f"$R^2$ = {o['r2']:.2f}\nslope {o['slope']:.2f}\n"
                f"RMSE {o['rmse']:.1f}", transform=ax.transAxes, va="top",
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                        ec=GRID, alpha=.85))
        ax.set_title(LABEL[m], color=C[m], fontweight="bold")
        ax.set_xlabel("true added unbalance  (g·mm)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    axes[0].set_ylabel("estimated  (g·mm)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Unbalance magnitude — estimated vs true", fontsize=13,
                 fontweight="bold", y=1.02)
    _save(fig, "fig_scatter.png")


def fig_metrics(B):
    """Four small-multiple bar panels: the headline metrics across methods."""
    specs = [("r2", "Magnitude  $R^2$", "higher better", False),
             ("localization", "Localization accuracy  (%)", "higher better", False),
             ("phase_hi", "Phase error, U≥24  (°)", "lower better", True),
             ("crosstalk", "Cross-plane leakage  (g·mm)", "lower better", True)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (key, title, sub, lower) in zip(axes, specs):
        vals = []
        for m in ORDER:
            M = B["methods"][m]
            if key == "localization":
                vals.append(M["localization_acc"] * 100)
            else:
                vals.append(M["overall"][key])
        bars = ax.bar(range(3), vals, color=[C[m] for m in ORDER], width=.66,
                      zorder=3)
        best = np.argmin(vals) if lower else np.argmax(vals)
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if i == best else "normal", color=INK)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[m].split(" (")[0] for m in ORDER],
                                                    rotation=12, fontsize=9)
        ax.text(.5, -.30, sub, transform=ax.transAxes, ha="center",
                fontsize=8.5, color=MUTED)
        ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
        ax.margins(y=.18)
    fig.suptitle("Physics vs ML vs Hybrid — headline metrics", fontsize=13,
                 fontweight="bold", y=1.03)
    _save(fig, "fig_metrics.png")


def fig_bland_altman(L):
    """Bland–Altman: hybrid agreement, physics faded for reference (loaded points)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, alpha, z in [("physics", .28, 1), ("hybrid", .8, 3)]:
        d = L[(L.method == m) & (L.loaded)]
        diff = d.est_mag - d.true_mag
        mean = (d.est_mag + d.true_mag) / 2
        bias, sd = diff.mean(), diff.std()
        ax.scatter(mean, diff, s=22, color=C[m], alpha=alpha, zorder=z,
                   edgecolor="white", linewidth=.3, label=LABEL[m])
        for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
            ax.axhline(y, color=C[m], ls=ls, lw=1, alpha=alpha, zorder=z)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xlabel("mean of estimate & truth  (g·mm)")
    ax.set_ylabel("estimate − truth  (g·mm)")
    ax.set_title("Bland–Altman agreement (hybrid vs physics)", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_bland_altman.png")


def fig_phase(L):
    """Polar: estimated vs true keyphasor phase for well-resolved points (hybrid)."""
    d = L[(L.method == "hybrid") & (L.loaded) & (L.true_mag >= 24)]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    for _, r in d.iterrows():
        t = np.radians(r.true_ang); e = np.radians(r.est_ang)
        ax.plot([t, e], [r.true_mag, r.est_mag], color=GRID, lw=.7, zorder=1)
    ax.scatter(np.radians(d.true_ang), d.true_mag, s=42, color="#c3c2b7",
               edgecolor=MUTED, label="true", zorder=2)
    ax.scatter(np.radians(d.est_ang), d.est_mag, s=42, color=C["hybrid"],
               edgecolor="white", linewidth=.4, label="hybrid estimate", zorder=3)
    ax.set_title("Keyphasor phase — true vs estimated  (U ≥ 24 g·mm)\n",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.14, 1.10), frameon=False)
    _save(fig, "fig_phase.png")


def fig_confusion(B):
    """Per-disk detection confusion (loaded vs not) for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, m in zip(axes, ORDER):
        c = B["methods"][m]["confusion"]
        M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        Mn = M / M.sum()
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=Mn.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i,j]}\n{Mn[i,j]*100:.0f}%", ha="center",
                        va="center", fontsize=11,
                        color="white" if Mn[i, j] > Mn.max() * .55 else INK)
        spec = c["tn"] / (c["tn"] + c["fp"]) * 100
        ax.set_title(f"{LABEL[m]}\nspecificity {spec:.0f}%", color=C[m],
                     fontweight="bold", fontsize=10.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred\nempty", "pred\nloaded"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["truly\nempty", "truly\nloaded"], fontsize=9)
        ax.grid(False)
    fig.suptitle("Per-disk detection — cross-plane leakage shrinks with the hybrid",
                 fontsize=12.5, fontweight="bold", y=1.04)
    _save(fig, "fig_confusion.png")


def fig_importance():
    """Top hybrid feature importances — what the learner leans on."""
    fi = pd.read_csv(ANALYSIS / "feature_importance.csv", index_col=0).head(12).iloc[::-1]
    colname = fi.columns[0]
    names = fi.index.str.replace("_", " ")
    is_phys = fi.index.str.startswith("phys_")
    colors = [C["physics"] if p else C["hybrid"] for p in is_phys]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(range(len(fi)), fi[colname], color=colors, zorder=3)
    ax.set_yticks(range(len(fi))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("permutation-free impurity importance")
    ax.set_title("Hybrid feature importance — physics prior anchors, "
                 "orbit/anisotropy features refine", fontweight="bold", fontsize=11)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["physics"], label="physics-ICM prior"),
                       Patch(color=C["hybrid"], label="measured signal feature")],
              frameon=False, loc="lower right", fontsize=9)
    _save(fig, "fig_importance.png")


def fig_cross_speed(L):
    """N1 vs N2 estimated magnitude — a physical unbalance is speed-independent."""
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = 66
    for m in ORDER:
        d = L[(L.method == m) & (L.loaded)]
        piv = d.pivot_table(index=["cid", "disk"], columns="speed", values="est_mag").dropna()
        if not {"N1", "N2"}.issubset(piv.columns):
            continue
        ax.scatter(piv.N1, piv.N2, s=26, color=C[m], alpha=.6, edgecolor="white",
                   linewidth=.3, label=LABEL[m])
    ax.plot([0, lim], [0, lim], "--", color=MUTED, lw=1)
    ax.set_xlabel("estimate at N1 = 1000 rpm  (g·mm)")
    ax.set_ylabel("estimate at N2 = 1200 rpm  (g·mm)")
    ax.set_title("Cross-speed consistency (internal validation)", fontweight="bold")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig_cross_speed.png")


def fig_uncertainty(L):
    """Hybrid: ensemble-predicted uncertainty vs actual absolute error."""
    d = L[(L.method == "hybrid") & (L.loaded)].copy()
    d["abserr"] = (d.est_mag - d.true_mag).abs()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(d["std"], d["abserr"], s=20, color=C["hybrid"], alpha=.5,
               edgecolor="white", linewidth=.3)
    lim = np.nanpercentile(d[["std", "abserr"]].to_numpy(), 99)
    ax.plot([0, lim], [0, lim], "--", color=MUTED, lw=1, label="1:1")
    ax.set_xlabel("predicted uncertainty  σ  (g·mm, tree ensemble)")
    ax.set_ylabel("actual |error|  (g·mm)")
    ax.set_title("Uncertainty calibration (hybrid)", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "fig_uncertainty.png")


def fig_orbits():
    """
    Response-orbit severity ladder: baseline-subtracted 1X orbit (the added-mass
    signature the estimators see) grows cleanly with the disk-1 unbalance.
    """
    feat = pd.read_csv(ANALYSIS / "features.csv")
    ladder = [("A001", "3 g·mm"), ("A003", "12 g·mm"), ("A004", "24 g·mm"),
              ("A005", "30 g·mm"), ("A007", "48 g·mm"), ("A008", "60 g·mm")]
    ladder = [(c, t) for c, t in ladder if c in set(feat.condition_id)]
    n = len(ladder)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.9), sharex=True, sharey=True)
    th = np.linspace(0, 2 * np.pi, 240)
    lim = 0.5
    for ax, (cid, title) in zip(axes, ladder):
        g = feat[(feat.condition_id == cid) & (feat.speed_id == "N1")]
        for plane, col, lab in [("plane1", C["physics"], "plane 1 (disk 1)"),
                                ("plane2", C["hybrid"], "plane 2 (disk 2)")]:
            X = g[f"{plane}_x_resp_re"].mean() + 1j * g[f"{plane}_x_resp_im"].mean()
            Y = g[f"{plane}_y_resp_re"].mean() + 1j * g[f"{plane}_y_resp_im"].mean()
            x = np.real(X * np.exp(1j * th)); y = np.real(Y * np.exp(1j * th))
            ax.plot(x, y, color=col, lw=1.8, label=lab)
        ax.set_title(f"{cid}  ·  {title}", fontsize=9.5)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color=GRID, lw=.6); ax.axvline(0, color=GRID, lw=.6)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("Y response (mils)", fontsize=8.5)
    fig.suptitle("Baseline-subtracted 1X response orbit grows with disk-1 unbalance "
                 "(N1, runout-compensated)", fontsize=12, fontweight="bold", y=1.05)
    _save(fig, "fig_orbits.png")


def fig_blind(B):
    """Blind balanced validation: estimated |U| per disk-point on unseen empty cases."""
    bb = B.get("blind_balanced")
    bp_path = ANALYSIS / "blind_predictions.csv"
    if not bb or not bp_path.exists():
        return
    bp = pd.read_csv(bp_path)
    bp = bp[bp.condition_id.isin(bb["conditions"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    rng = np.random.default_rng(0)
    for i, m in enumerate(ORDER):
        vals = np.concatenate([bp[f"{m}_U1_gmm"].to_numpy(), bp[f"{m}_U2_gmm"].to_numpy()])
        x = i + (rng.random(len(vals)) - 0.5) * 0.5
        ax.scatter(x, vals, s=26, color=C[m], alpha=.6, edgecolor="white", linewidth=.4)
        sp = bb["methods"][m]["specificity"] * 100
        ax.text(i, -1.6, f"{sp:.0f}%", ha="center", fontsize=12, fontweight="bold",
                color=C[m])
    ax.axhline(6.0, color="#d03b3b", ls="--", lw=1.4)
    ax.text(2.46, 6.3, "6 g·mm detection threshold", color="#d03b3b", fontsize=9,
            ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[m] for m in ORDER])
    ax.set_ylabel("estimated added |U|  (g·mm)")
    ax.set_ylim(-3, max(12, ax.get_ylim()[1]))
    ax.set_title("Blind balanced validation — every point should be below the line\n"
                 f"({bb['n_conditions']} unseen empty-disk conditions, "
                 f"{bb['methods']['hybrid']['n']} disk-points)", fontweight="bold", fontsize=11)
    ax.text(.5, -.16, "specificity (correctly called empty):", transform=ax.transAxes,
            ha="center", fontsize=9, color=MUTED)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig_blind.png")


def fig_blind_scatter(B):
    """Blind est-vs-true on all unsealed conditions — shows the detection/magnitude trade-off."""
    bv = B.get("blind_validation")
    if not bv:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    lim = 64
    for ax, m in zip(axes, ORDER):
        P = bv["points"][m]
        for p in P:
            empty = p["true"] < 6
            ax.scatter(p["true"], p["est"], s=40,
                       color="#c3c2b7" if empty else C[m],
                       edgecolor="white", linewidth=.5, zorder=3,
                       marker="o" if p["disk"] == 1 else "^")
        ax.plot([0, lim], [0, lim], "--", color=MUTED, lw=1, zorder=1)
        ax.axhline(6, color="#d03b3b", lw=.9, ls=":", zorder=1)
        ax.axvline(6, color="#d03b3b", lw=.9, ls=":", zorder=1)
        s = bv["methods"][m]
        ax.text(.04, .96, f"MAE {s['mae']:.1f} g·mm\nspecificity {s['specificity']*100:.0f}%\n"
                f"loaded MAE {s['mae_loaded']:.0f}", transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, alpha=.85))
        ax.set_title(LABEL[m], color=C[m], fontweight="bold")
        ax.set_xlabel("true unbalance (g·mm)")
        ax.set_xlim(-3, lim); ax.set_ylim(-3, lim); ax.set_aspect("equal")
    axes[0].set_ylabel("estimated (g·mm)")
    fig.suptitle("Blind validation — 10 unseen conditions (grey = truly balanced). "
                 "Physics tracks large two-plane magnitude; trees underestimate it but "
                 "reject empties.", fontsize=11.5, fontweight="bold", y=1.03)
    _save(fig, "fig_blind_scatter.png")


def main():
    L, B = _load()
    print("rendering figures ->", ANALYSIS)
    fig_blind(B)
    fig_blind_scatter(B)
    fig_scatter(L, B)
    fig_metrics(B)
    fig_bland_altman(L)
    fig_phase(L)
    fig_confusion(B)
    fig_importance()
    fig_cross_speed(L)
    fig_uncertainty(L)
    fig_orbits()
    print("done.")


if __name__ == "__main__":
    main()
