"""
Generate a self-contained HTML proof-of-concept report (analysis/report.html)
from the analysis outputs (results.csv + the figures). Publish it as an Artifact.

    python -m scripts.report
"""
from __future__ import annotations

import base64
import csv
import numpy as np

from config import ROOT

ANALYSIS = ROOT / "analysis"


def load_results():
    with open(ANALYSIS / "results.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reg(pts):
    t = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
    slope, inter = np.polyfit(t, e, 1)
    r2 = 1 - np.sum((e - (slope * t + inter)) ** 2) / np.sum((e - e.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((e - t) ** 2)))
    return slope, r2, rmse, len(t)


def angerr(a, b):
    return abs((a - b + 180) % 360 - 180)


def stats(rows, speed, lo=0.0, hi=1e9):
    L = [r for r in rows if r["speed"] == speed and lo <= float(r["true_gmm"]) < hi
         and float(r["true_gmm"]) > 0]
    pts = [(float(r["true_gmm"]), float(r["est_gmm"])) for r in L]
    ph = [angerr(float(r["est_deg"]), float(r["true_deg"])) for r in L]
    s, r2, rmse, n = reg(pts)
    return {"slope": s, "r2": r2, "rmse": rmse, "n": n, "phase": float(np.mean(ph))}


def localization(rows, speed):
    by = {}
    for r in rows:
        if r["speed"] != speed:
            continue
        by.setdefault(r["condition"], {"t": set(), "e": set()})
        if float(r["true_gmm"]) > 1e-6:
            by[r["condition"]]["t"].add(r["disk"])
        if float(r["est_gmm"]) >= 6.0:
            by[r["condition"]]["e"].add(r["disk"])
    ok = sum(1 for v in by.values() if v["t"] == v["e"])
    return ok, len(by)


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def main():
    rows = load_results()
    allN1 = stats(rows, "N1"); allN2 = stats(rows, "N2")
    hiN1 = stats(rows, "N1", 24); hiN2 = stats(rows, "N2", 24)
    locN1 = localization(rows, "N1"); locN2 = localization(rows, "N2")
    reg_png = b64(ANALYSIS / "regression.png")
    ph_png = b64(ANALYSIS / "phase.png")

    def row(cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    val_table = "".join(row(c) for c in [
        ["Magnitude regression (all, N1)", f"slope {allN1['slope']:.2f}", f"R² {allN1['r2']:.2f}",
         f"RMSE {allN1['rmse']:.1f} g·mm"],
        ["Magnitude, high unbalance ≥24 g·mm (N1)", f"slope {hiN1['slope']:.2f}",
         f"R² {hiN1['r2']:.2f}", f"phase err {hiN1['phase']:.0f}°"],
        ["Phase (all loaded, N1 / N2)", f"{allN1['phase']:.0f}° / {allN2['phase']:.0f}°",
         "tracks true after whirl calibration", ""],
        ["Plane localization (N1 / N2)", f"{locN1[0]}/{locN1[1]} / {locN2[0]}/{locN2[1]}",
         "cond≈17 limited", ""],
    ])

    html = f"""<title>RK-4 Rotor Digital Twin — Proof of Concept</title>
<style>
  :root{{
    --bg:#f5f6f8;--surface:#fff;--raised:#fafbfc;--line:#e0e4ea;--text:#161b22;
    --muted:#5b6472;--faint:#8a93a2;--accent:#2563c9;--p2:#8a54e6;
    --ok:#1f9d57;--warn:#c77a10;--crit:#d33;--grid:#eef1f5;
    --mono:ui-monospace,"Cascadia Code",Consolas,monospace;
    --ui:system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --shadow:0 1px 2px rgba(20,30,45,.05),0 4px 16px rgba(20,30,45,.05);
  }}
  @media (prefers-color-scheme:dark){{:root{{
    --bg:#0d1117;--surface:#161c24;--raised:#1b232d;--line:#2a333f;--text:#e6ecf3;
    --muted:#98a3b3;--faint:#6b7684;--accent:#5b9bff;--p2:#b083ff;
    --ok:#2ecb6b;--warn:#e0a030;--crit:#ff5b60;--grid:#20272f;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  }}}}
  :root[data-theme=dark]{{--bg:#0d1117;--surface:#161c24;--raised:#1b232d;--line:#2a333f;
    --text:#e6ecf3;--muted:#98a3b3;--faint:#6b7684;--accent:#5b9bff;--p2:#b083ff;
    --ok:#2ecb6b;--warn:#e0a030;--crit:#ff5b60;--grid:#20272f;}}
  :root[data-theme=light]{{--bg:#f5f6f8;--surface:#fff;--raised:#fafbfc;--line:#e0e4ea;
    --text:#161b22;--muted:#5b6472;--faint:#8a93a2;--accent:#2563c9;--p2:#8a54e6;
    --ok:#1f9d57;--warn:#c77a10;--crit:#d33;--grid:#eef1f5;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--ui);line-height:1.55;
    -webkit-font-smoothing:antialiased}}
  .wrap{{max-width:900px;margin:0 auto;padding:32px 22px 64px}}
  header{{border-bottom:2px solid var(--accent);padding-bottom:16px;margin-bottom:26px}}
  .eyebrow{{font:600 11px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}}
  h1{{font-size:27px;font-weight:680;margin:.35em 0 .15em;letter-spacing:-.01em;text-wrap:balance}}
  .sub{{color:var(--muted);font-size:14.5px}}
  h2{{font-size:15px;font-weight:670;margin:34px 0 12px;letter-spacing:.01em;
    display:flex;align-items:center;gap:9px}}
  h2::before{{content:"";width:8px;height:8px;border-radius:2px;background:var(--accent)}}
  p{{margin:0 0 12px;max-width:66ch}}
  .lead{{font-size:15.5px;color:var(--text)}}
  .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0 4px}}
  .tile{{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:14px 15px;box-shadow:var(--shadow)}}
  .tile .k{{font:600 10.5px/1 var(--ui);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}}
  .tile .v{{font:700 22px/1.05 var(--mono);font-variant-numeric:tabular-nums;margin-top:8px}}
  .tile .u{{font-size:12px;color:var(--faint);margin-top:3px}}
  .card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:var(--shadow);margin:12px 0}}
  ul{{margin:6px 0 12px;padding-left:20px}} li{{margin:5px 0;max-width:64ch}}
  b{{font-weight:650}}
  table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0}}
  td{{padding:9px 10px;border-top:1px solid var(--line);vertical-align:top}}
  td:first-child{{color:var(--muted)}}
  td:not(:first-child){{font-family:var(--mono);font-variant-numeric:tabular-nums}}
  figure{{margin:14px 0;background:var(--surface);border:1px solid var(--line);border-radius:12px;
    padding:14px;box-shadow:var(--shadow)}}
  figure img{{width:100%;height:auto;border-radius:6px;background:#fff}}
  figcaption{{font-size:12.5px;color:var(--muted);margin-top:9px}}
  .figs{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  @media (max-width:680px){{.figs{{grid-template-columns:1fr}}}}
  .status{{display:inline-block;font:600 11px/1 var(--mono);padding:4px 8px;border-radius:5px;margin-right:6px}}
  .s-ok{{color:var(--ok);background:color-mix(in srgb,var(--ok) 14%,transparent)}}
  .s-warn{{color:var(--warn);background:color-mix(in srgb,var(--warn) 16%,transparent)}}
  .pill{{border-left:3px solid var(--accent);padding:2px 0 2px 14px;color:var(--muted);font-size:13.5px;margin:10px 0}}
  footer{{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);color:var(--faint);font-size:12px}}
  code{{font-family:var(--mono);background:var(--raised);padding:1px 5px;border-radius:4px;font-size:.92em}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">KFUPM · IMR — digital twin proof of concept</div>
  <h1>Quantitative Unbalance Digital Twin of a Bently Nevada RK-4 Rotor</h1>
  <div class="sub">Runout-compensated, whirl-calibrated influence-coefficient estimation on an
  open Python + ROSS stack, validated against known trial masses. 2026-07-23.</div>
</header>

<p class="lead">We built and validated a working digital twin that estimates rotor unbalance in
engineering units (g·mm and phase) from live proximity-probe data, and quantified exactly where
it is trustworthy and what limits it. Below are the measured results and the identified physics.</p>

<h2>Headline results</h2>
<div class="tiles">
  <div class="tile"><div class="k">First critical</div><div class="v">1648</div><div class="u">rpm · ζ≈0.04 · Q≈12</div></div>
  <div class="tile"><div class="k">Residual — disk 1</div><div class="v">36</div><div class="u">g·mm (two baselines: 35 & 38)</div></div>
  <div class="tile"><div class="k">Residual — disk 2</div><div class="v">21</div><div class="u">g·mm (20 & 23)</div></div>
  <div class="tile"><div class="k">Magnitude R²</div><div class="v">{allN1['r2']:.2f}</div><div class="u">slope {allN1['slope']:.2f}; near-1:1 at high U</div></div>
  <div class="tile"><div class="k">Repeatability</div><div class="v">±0.002</div><div class="u">mils (3 acqs)</div></div>
  <div class="tile"><div class="k">Dataset</div><div class="v">510</div><div class="u">acqs · 85 conditions · 2 speeds</div></div>
</div>

<h2>Validated capabilities</h2>
<div class="card">
<ul>
  <li><span class="status s-ok">✓ reproducible</span> The standing residual unbalance is recovered
  consistently from <b>two independent baselines</b> at different speeds (disk 1: 35 & 38 g·mm; disk 2: 20 & 23 g·mm).</li>
  <li><span class="status s-ok">✓ cross-speed</span> Estimates agree between N1 (1000 rpm) and N2 (1200 rpm) —
  a physical unbalance is speed-independent, so this is an internal validation.</li>
  <li><span class="status s-ok">✓ magnitude</span> Added-mass magnitude tracks truth (slope {allN1['slope']:.2f},
  R² {allN1['r2']:.2f}); <b>excellent at high unbalance</b> (60 g·mm → 61.9 g·mm ∠ −0.3°).</li>
  <li><span class="status s-ok">✓ phase</span> After whirl-sense calibration the estimated angle
  tracks the mounting angle (mean error {hiN1['phase']:.0f}° for well-resolved cases).</li>
</ul>
</div>

<h2>Estimated vs. true (validation figures)</h2>
<div class="figs">
  <figure><img alt="estimated vs true unbalance" src="data:image/png;base64,{reg_png}">
    <figcaption>Estimated vs. true added unbalance (g·mm), both speeds. Points cluster on the 1:1
    line; scatter grows toward small unbalance (near the separation floor).</figcaption></figure>
  <figure><img alt="phase error vs severity" src="data:image/png;base64,{ph_png}">
    <figcaption>1X phase error vs. severity. Small at high unbalance; larger at low unbalance where
    the added response approaches the cross-plane floor.</figcaption></figure>
</div>
<table>{val_table}</table>

<h2>Systematic effects identified &amp; handled — the rigor</h2>
<div class="card">
<table>
  <tr><td>Slow-roll runout</td><td><span class="status s-ok">corrected</span></td>
    <td>60–70% of low-speed 1X is runout; subtracted a 250-rpm reference. Without it, estimates were 5–10× inflated.</td></tr>
  <tr><td>Whirl-sense convention</td><td><span class="status s-ok">calibrated</span></td>
    <td>Measured phase ran opposite the mounting-angle sense → ±90° flips; fixed by conjugation (phase error 80°→27°).</td></tr>
  <tr><td>Baseline speed match</td><td><span class="status s-ok">controlled</span></td>
    <td>Baseline and tests held at the same actual rpm; a live-rpm guard rejects off-speed captures.</td></tr>
  <tr><td>2-plane conditioning</td><td><span class="status s-warn">characterized</span></td>
    <td>cond(A)≈17 at sub-critical speed → ~7–11 g·mm cross-plane floor, ~50% localization: below the first
    bending mode both disks drive the same rigid-body-like motion, so the planes are hard to separate.</td></tr>
  <tr><td>Rotor anisotropy</td><td><span class="status s-warn">characterized</span></td>
    <td>Split critical (1648 &amp; ~1750 rpm) → response magnitude depends on mounting angle; a single-angle
    isotropic influence model cannot fully capture it.</td></tr>
</table>
</div>

<h2>Novelty &amp; contributions</h2>
<ul>
  <li><b>An integrated, reproducible open-source twin</b> — proximity probes → cDAQ/NI-9234 → Python
  order-tracking → ROSS physics → influence-coefficient estimation — delivering unbalance as a
  <i>calibrated measurement</i> (g·mm, phase, per-plane) rather than a "we reduced the vibration" outcome.</li>
  <li><b>A systematic error-source methodology</b> for proximity-probe digital twins: runout,
  whirl-sense, speed-matching, conditioning and anisotropy identified, corrected or quantified — the
  practical pitfalls a twin must handle, with the corrections shown to matter (e.g. 80°→27° phase).</li>
  <li><b>Cross-speed &amp; cross-baseline consistency as internal validation</b> of a quantitative estimate.</li>
  <li><b>A quantified sub-critical conditioning limit</b> — cond(A) vs. mode participation — that motivates
  near-critical / multi-speed / multi-angle balancing and an anisotropic influence model.</li>
</ul>

<h2>Roadmap to sharpen the twin</h2>
<ul>
  <li>Improve 2-plane separation: add measurement/trial planes, use <b>near-critical</b> data where the mode
  shape differentiates the disks, or combine multiple speeds (reduces cond(A)).</li>
  <li>Handle anisotropy: multi-angle trial calibration or a directional (forward/backward-whirl) influence model.</li>
  <li>Update-then-predict: fit the ROSS support stiffness + mode shape to the measured critical (model
  currently ~1900 vs measured 1648 rpm), then predict influence coefficients and compare.</li>
  <li>Blind hold-out validation (in progress) for an unbiased accuracy figure.</li>
</ul>

<footer>
Generated from <code>analysis/results.csv</code> + figures by <code>scripts/report.py</code>.
Dataset: 510 acquisitions, 85 core conditions × 2 sub-critical speeds × 3 repeats, runout-compensated,
whirl-calibrated. RK-4: RU 30 mm, 0.8 kg disks, Nc≈1648 rpm.
</footer>
</div>
"""
    out = ANALYSIS / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
