"""
Build the interactive digital-twin dashboard (self-contained HTML).

Assembles a compact data payload from the benchmark outputs + feature table and
renders a single-page, theme-aware, offline dashboard:
  * headline stat tiles + the physics/ML/hybrid concept,
  * interactive method comparison (bars + table),
  * interactive estimated-vs-true scatter (method toggle),
  * interactive 1X response-orbit explorer (condition + speed),
  * validation-figure gallery (embedded PNGs), feature importance, blind set.

Writes two files that share one body:
  analysis/dashboard.html          standalone (opens directly in a browser)
  <scratchpad>/twin_dashboard.html content-only (for publishing as an Artifact)

    python -m scripts.benchmark && python -m scripts.figures   # refresh data first
    python -m scripts.dashboard
"""
from __future__ import annotations

import base64
import json
import numpy as np
import pandas as pd

from config import ROOT
from estimation.stats import circmean_deg
from processing.features import SENSORS

ANALYSIS = ROOT / "analysis"
SCRATCH = ROOT / "analysis"   # standalone lives here; artifact copy path printed


def _orbit_payload(feat: pd.DataFrame) -> dict:
    """Mean baseline-subtracted response phasors per condition+speed for the explorer."""
    out = {}
    labeled = feat[feat.labeled == 1]
    for (cid, sp), g in labeled.groupby(["condition_id", "speed_id"]):
        rec = {"cid": cid, "speed": sp, "config": g.config.iloc[0]}
        for p in SENSORS:
            rec[p] = [float(g[f"{p}_resp_re"].mean()), float(g[f"{p}_resp_im"].mean())]
        # angles averaged CIRCULARLY -- an arithmetic mean of degrees is wrong
        # across the -180/+180 branch cut
        for k in (1, 2):
            a = circmean_deg(pd.to_numeric(g[f"U{k}_ang"], errors="coerce").to_numpy())
            rec[f"U{k}"] = [float(g[f"U{k}_mag"].mean()),
                            0.0 if pd.isna(a) else float(a)]
        out[f"{cid}|{sp}"] = rec
    return out


def _scatter_payload(L: pd.DataFrame) -> dict:
    out = {}
    for m in ("physics", "ml", "hybrid"):
        d = L[(L.method == m) & (L.loaded)]
        out[m] = [[round(float(t), 2), round(float(e), 2), int(dk)]
                  for t, e, dk in zip(d.true_mag, d.est_mag, d.disk)]
    return out


def _est_payload(L: pd.DataFrame) -> dict:
    """Per-condition mean estimate per method (for the orbit-explorer readout)."""
    out = {}
    for (cid, sp, m), g in L.groupby(["cid", "speed", "method"]):
        rec = {}
        for k in (1, 2):
            d = g[g.disk == k]
            if len(d):
                # collapse repeats as one complex vector (circular-safe)
                z = (d.est_mag.to_numpy(float)
                     * np.exp(1j * np.radians(d.est_ang.to_numpy(float)))).mean()
                rec[f"U{k}"] = round(float(abs(z)), 1)
                rec[f"A{k}"] = round(float(np.degrees(np.angle(z))), 0)
            else:
                rec[f"U{k}"] = rec[f"A{k}"] = None
        out.setdefault(f"{cid}|{sp}", {})[m] = rec
    return out


def b64(name):
    p = ANALYSIS / name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def build_payload():
    B = json.load(open(ANALYSIS / "benchmark.json"))
    L = pd.read_csv(ANALYSIS / "estimates_long.csv")
    feat = pd.read_csv(ANALYSIS / "features.csv")
    fi = pd.read_csv(ANALYSIS / "feature_importance.csv", index_col=0)
    blind = (pd.read_csv(ANALYSIS / "blind_predictions.csv")
             if (ANALYSIS / "blind_predictions.csv").exists() else pd.DataFrame())

    payload = {
        "bench": B,
        "scatter": _scatter_payload(L),
        "orbits": _orbit_payload(feat),
        "ests": _est_payload(L),
        "importance": [[k, round(float(v), 4)] for k, v in
                       fi.iloc[:, 0].head(12).items()],
        "blind": (blind.round(1).to_dict("records") if len(blind) else []),
    }
    figs = {n: b64(f"fig_{n}.png") for n in
            ["scatter", "metrics", "bland_altman", "phase", "confusion",
             "importance", "cross_speed", "uncertainty", "orbits", "blind",
             "blind_scatter"]}
    return payload, figs


def render(payload, figs) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    B = payload["bench"]
    m = B["methods"]

    def tile(k, v, u, cls=""):
        return (f'<div class="tile {cls}"><div class="k">{k}</div>'
                f'<div class="v">{v}</div><div class="u">{u}</div></div>')

    bb = B.get("blind_balanced")
    spec_tile = ""
    if bb:
        sh = bb["methods"]["hybrid"]["specificity"] * 100
        sp = bb["methods"]["physics"]["specificity"] * 100
        spec_tile = tile("Blind specificity",
                         f'{sh:.0f}<span class="pct">%</span>',
                         f'{bb["n_conditions"]} unseen balanced cases · physics {sp:.0f}%', "hero")

    tiles = "".join([
        tile("Hybrid localization", f'{m["hybrid"]["localization_acc"]*100:.0f}<span class="pct">%</span>',
             f'up from {m["physics"]["localization_acc"]*100:.0f}% physics-only', "hero"),
        spec_tile,
        tile("Magnitude R²", f'{m["hybrid"]["overall"]["r2"]:.2f}',
             f'physics {m["physics"]["overall"]["r2"]:.2f} · ML {m["ml"]["overall"]["r2"]:.2f}'),
        tile("Phase error", f'{m["hybrid"]["overall"]["phase_hi"]:.0f}<span class="pct">°</span>',
             f'U≥24 g·mm · physics {m["physics"]["overall"]["phase_hi"]:.0f}°'),
        tile("Cross-plane leakage", f'{m["hybrid"]["overall"]["crosstalk"]:.1f}',
             f'g·mm · physics {m["physics"]["overall"]["crosstalk"]:.1f}'),
        tile("Cross-speed agree", f'{m["hybrid"]["cross_speed"]["mean_abs_gmm"]:.1f}',
             f'g·mm N1↔N2 · physics {m["physics"]["cross_speed"]["mean_abs_gmm"]:.1f}'),
        tile("Dataset", f'{B["n_labeled"]}',
             f'labeled acqs · {B["n_conditions"]} conditions · 2 speeds'),
    ])

    figblock = lambda key, cap: (
        f'<figure><img alt="{cap}" src="data:image/png;base64,{figs[key]}">'
        f'<figcaption>{cap}</figcaption></figure>')

    return f"""<title>RK-4 Rotor — Hybrid Digital Twin</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">KFUPM · IMR — hybrid physics + ML digital twin</div>
  <h1>Quantitative Unbalance Digital Twin of a Bently Nevada RK-4 Rotor</h1>
  <p class="sub">A physics-informed machine-learning twin that estimates rotor
  unbalance in engineering units (g·mm, keyphasor phase, per-disk) from live
  proximity-probe data — and beats both the classical influence-coefficient method
  and pure ML on the same held-out conditions.</p>
</header>

<section class="tiles">{tiles}</section>

<h2>Three twins, one signal chain</h2>
<p class="lead">The same keyphasor order-tracking front end feeds three inverse models.
The <b>hybrid</b> hands the physics estimate to the learner as a prior, so the model
<i>corrects the physics residual</i> instead of regressing from scratch — anchoring scale
and phase while cleaning up the sub-critical conditioning and rotor-anisotropy bias.</p>
<div id="pipeline"></div>

<h2>Method comparison</h2>
<p>Leak-free evaluation (Group<i>K</i>-Fold on condition — every repeat/mount of a
condition shares a fold), scored on identical held-out conditions. Toggle a metric:</p>
<div class="controls" id="metric-controls"></div>
<div class="grid2">
  <div class="card"><svg id="bars" viewBox="0 0 560 300"></svg></div>
  <div class="card"><div id="method-table"></div></div>
</div>

<h2>Estimated vs true — interactive</h2>
<div class="controls" id="scatter-controls"></div>
<div class="card"><svg id="scatter" viewBox="0 0 560 520"></svg>
  <div class="cap" id="scatter-cap"></div></div>

<h2>1X response-orbit explorer</h2>
<p>The baseline-subtracted 1X orbit is the added-mass signature the twin inverts.
Pick a condition and speed — note how the disk-2 plane often whirls as hard as the
loaded plane, the physical reason localization is the hard part.</p>
<div class="controls">
  <label>Condition <select id="orbit-cond"></select></label>
  <label>Speed <select id="orbit-speed"></select></label>
</div>
<div class="grid2">
  <div class="card"><svg id="orbit" viewBox="-1.1 -1.1 2.2 2.2"></svg></div>
  <div class="card"><div id="orbit-readout"></div></div>
</div>

<h2>Validation figures</h2>
<div class="figs">
  {figblock("metrics", "Headline metrics across the three estimators.")}
  {figblock("scatter", "Estimated vs true added unbalance (loaded disk-points), per method.")}
  {figblock("confusion", "Per-disk detection: cross-plane false-positives collapse with the hybrid.")}
  {figblock("bland_altman", "Bland–Altman agreement — hybrid bias/limits vs physics.")}
  {figblock("phase", "Keyphasor phase, true vs estimated (U ≥ 24 g·mm).")}
  {figblock("cross_speed", "Cross-speed consistency — a physical unbalance is speed-independent.")}
  {figblock("uncertainty", "Ensemble uncertainty tracks the actual error (hybrid).")}
  {figblock("importance", "What the hybrid leans on: physics prior first, then orbit/anisotropy cues.")}
  {figblock("orbits", "Response-orbit severity ladder (disk-1 unbalance, N1).")}
</div>

<h2>Blind validation — 10 unsealed conditions, unbiased</h2>
<p>All ten acquired blind conditions were unsealed after prediction: <b>five balanced</b>
(empty disks) and <b>five loaded two-plane</b> cases (some off-grid, some out of the mostly
single-disk training range). The models never saw them. This is the honest accuracy figure —
and it shows a real trade-off.</p>
<div class="figs">
  {figblock("blind_scatter", "Estimated vs true on all 10 unseen conditions (grey = truly balanced). Physics tracks large two-plane magnitude but false-alarms on empties; the trees reject empties cleanly but underestimate out-of-distribution two-plane loads.")}
  {figblock("blind", "Balanced-only specificity: on the 5 empty-disk conditions the trees keep every reading below the 6 g·mm threshold; physics does not.")}
</div>
<div class="grid2">
  <div class="card"><div id="blind-val"></div></div>
  <div class="card"><div id="blind-spec"></div>
    <div id="blind-table" style="margin-top:14px"></div></div>
</div>
<p class="pill">Take-away: the learned twins win <b>detection &amp; specificity</b> (no false
alarms on balanced rotors); the linear physics wins <b>magnitude on large out-of-distribution
two-plane loads</b> (the trees don't extrapolate). Adding two-plane conditions to training
is the concrete next step — the hybrid frame already carries the physics scale as a prior.</p>

<h2>Method &amp; rigor</h2>
<div class="card"><table class="rigor">
  <tr><td>Slow-roll runout</td><td class="ok">corrected</td>
    <td>60–70% of low-speed 1X is runout; subtracted a 251-rpm complex reference per sensor.</td></tr>
  <tr><td>Whirl-sense</td><td class="ok">calibrated</td>
    <td>Measured phase conjugated to the mounting-angle sense (fixes ±90° flips).</td></tr>
  <tr><td>Leakage-free CV</td><td class="ok">enforced</td>
    <td>GroupKFold on condition_id — repeats/mounts never straddle folds; the three
    calibration conditions are excluded from every method's score.</td></tr>
  <tr><td>2-plane conditioning</td><td class="warn">characterized</td>
    <td>cond(A) ≈ {B["cond_A"].get("N1","?")} (N1) / {B["cond_A"].get("N2","?")} (N2):
    sub-critical, both disks drive near-rigid motion. This is the localization limiter the hybrid attacks.</td></tr>
  <tr><td>Uncertainty</td><td class="ok">quantified</td>
    <td>Tree-ensemble spread → per-estimate σ (g·mm), calibrated against held-out error.</td></tr>
</table></div>

<footer>
Generated by <code>scripts/dashboard.py</code> from <code>analysis/benchmark.json</code> +
figures. Estimators: physics ICM · Extra-Trees ML · physics-informed hybrid.
RK-4: RU 30 mm, 0.8 kg disks, N<sub>c</sub> ≈ 1648 rpm, sub-critical dwells 1000 / 1200 rpm.
</footer>
</div>
<script>const DATA={data};{JS}</script>
"""


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--sec:#52514e;--muted:#898781;
--grid:#e1e0d9;--line:rgba(11,11,11,.10);--physics:#2a78d6;--ml:#eb6834;--hybrid:#1baf7a;
--ok:#0ca30c;--warn:#c98500;--shadow:0 1px 2px rgba(20,30,45,.05),0 6px 20px rgba(20,30,45,.06);
--mono:ui-monospace,"Cascadia Code",Consolas,monospace;--ui:system-ui,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--sec:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--line:rgba(255,255,255,.10);--physics:#3987e5;--ml:#d95926;--hybrid:#199e70;--ok:#0ca30c;--warn:#e0a030;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.4);}}
:root[data-theme=dark]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--sec:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--line:rgba(255,255,255,.10);--physics:#3987e5;--ml:#d95926;--hybrid:#199e70;--warn:#e0a030;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--ui);line-height:1.55;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:34px 22px 72px}
header{border-bottom:2px solid var(--hybrid);padding-bottom:18px;margin-bottom:24px}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--hybrid)}
h1{font-size:29px;font-weight:700;margin:.35em 0 .3em;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--sec);font-size:15px;max-width:74ch}
.lead{font-size:15.5px;max-width:80ch;color:var(--ink)}
h2{font-size:16px;font-weight:670;margin:40px 0 12px;display:flex;align-items:center;gap:9px}
h2::before{content:"";width:8px;height:8px;border-radius:2px;background:var(--hybrid)}
p{max-width:80ch;color:var(--sec)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:6px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:15px 16px;box-shadow:var(--shadow)}
.tile.hero{border-color:var(--hybrid);box-shadow:0 0 0 1px var(--hybrid),var(--shadow)}
.tile .k{font:600 10.5px/1.2 var(--ui);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tile .v{font:700 30px/1.05 var(--ui);margin-top:9px;color:var(--ink)}
.tile.hero .v{color:var(--hybrid)}
.tile .v .pct{font-size:18px;font-weight:600}
.tile .u{font-size:11.5px;color:var(--muted);margin-top:4px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:var(--shadow);margin:8px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.controls label{font-size:12.5px;color:var(--sec);display:flex;gap:6px;align-items:center}
select{font-family:var(--ui);font-size:13px;padding:5px 8px;border-radius:8px;border:1px solid var(--line);
background:var(--surface);color:var(--ink)}
.chip{font:600 12px/1 var(--ui);padding:7px 12px;border-radius:20px;border:1px solid var(--line);
background:var(--surface);color:var(--sec);cursor:pointer;transition:.12s}
.chip[aria-pressed=true]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
svg{width:100%;height:auto;display:block;overflow:visible}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 9px;border-top:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;border-top:0}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.rigor td:first-child{color:var(--sec);font-weight:600;white-space:nowrap}
.rigor td.ok{color:var(--ok);font-weight:600}.rigor td.warn{color:var(--warn);font-weight:600}
.figs{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:820px){.figs{grid-template-columns:1fr}}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;box-shadow:var(--shadow)}
figure img{width:100%;height:auto;border-radius:6px;background:#fff}
figcaption{font-size:12px;color:var(--muted);margin-top:8px}
.cap{font-size:12px;color:var(--muted);margin-top:6px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;margin:2px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:6px;color:var(--sec)}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
code{font-family:var(--mono);background:var(--bg);padding:1px 5px;border-radius:4px;font-size:.9em}
.pill{border-left:3px solid var(--hybrid);padding:8px 0 8px 14px;color:var(--sec);font-size:13.5px;margin:14px 0;background:linear-gradient(90deg,color-mix(in srgb,var(--hybrid) 6%,transparent),transparent)}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
.tt{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font:12px var(--ui);
padding:6px 9px;border-radius:7px;opacity:0;transition:opacity .1s;z-index:9;white-space:nowrap}
"""


JS = r"""
const M=DATA.bench.methods, MK=['physics','ml','hybrid'];
const COL=k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();
const NAME={physics:'Physics (ICM)',ml:'ML (Extra-Trees)',hybrid:'Hybrid'};
const SVGNS='http://www.w3.org/2000/svg';
function el(t,a={},kids=[]){const e=document.createElementNS(SVGNS,t);
 for(const k in a)e.setAttribute(k,a[k]); (Array.isArray(kids)?kids:[kids]).forEach(c=>c&&e.appendChild(c));return e;}
function txt(t){return document.createTextNode(t);}
let tip=document.createElement('div');tip.className='tt';document.body.appendChild(tip);
function showTip(x,y,h){tip.innerHTML=h;tip.style.left=(x+12)+'px';tip.style.top=(y+12)+'px';tip.style.opacity=1;}
function hideTip(){tip.style.opacity=0;}

/* ---- pipeline diagram ---- */
(function(){const box=document.getElementById('pipeline');
 const stages=[['Proximity probes','4× X/Y + keyphasor'],['Order tracking','keyphasor 1X/2X/3X'],
  ['Features','runout-comp · orbits'],['3 inverse models','physics · ML · hybrid'],['Unbalance','g·mm ∠° per disk']];
 const s=el('svg',{viewBox:'0 0 980 96'});const w=176,g=24;
 stages.forEach((st,i)=>{const x=i*(w+g);const c=i==3?'hybrid':'muted';
  s.appendChild(el('rect',{x,y:20,width:w,height:56,rx:11,fill:'var(--surface)',
   stroke:i==3?'var(--hybrid)':'var(--line)','stroke-width':i==3?2:1}));
  const t1=el('text',{x:x+w/2,y:44,'text-anchor':'middle','font-size':13,'font-weight':650,fill:'var(--ink)'});t1.appendChild(txt(st[0]));s.appendChild(t1);
  const t2=el('text',{x:x+w/2,y:62,'text-anchor':'middle','font-size':11,fill:'var(--muted)'});t2.appendChild(txt(st[1]));s.appendChild(t2);
  if(i<stages.length-1)s.appendChild(el('path',{d:`M${x+w+2} 48 L${x+w+g-2} 48`,stroke:'var(--muted)','stroke-width':1.6,'marker-end':'url(#ar)'}));});
 const defs=el('defs');const mk=el('marker',{id:'ar',markerWidth:8,markerHeight:8,refX:6,refY:3,orient:'auto'});
 mk.appendChild(el('path',{d:'M0 0 L6 3 L0 6 z',fill:'var(--muted)'}));defs.appendChild(mk);s.appendChild(defs);
 box.appendChild(s);})();

/* ---- metric bars ---- */
const METRICS=[['r2','Magnitude R²',false,v=>v.toFixed(2)],['localization','Localization %',false,v=>v.toFixed(0)],
 ['phase_hi','Phase err ° (U≥24)',true,v=>v.toFixed(1)],['rmse','RMSE g·mm',true,v=>v.toFixed(1)],
 ['crosstalk','Cross-plane leakage',true,v=>v.toFixed(1)]];
let curMetric=0;
function metricVal(m,key){return key=='localization'?M[m].localization_acc*100:M[m].overall[key];}
function drawBars(){const svg=document.getElementById('bars');svg.innerHTML='';
 const [key,label,lower,fmt]=METRICS[curMetric];const W=560,H=300,pad=54;
 const vals=MK.map(m=>metricVal(m,key));const mx=Math.max(...vals)*1.16;
 const bw=90,gap=48,x0=90;
 const best=lower?vals.indexOf(Math.min(...vals)):vals.indexOf(Math.max(...vals));
 svg.appendChild(el('line',{x1:pad,y1:H-pad,x2:W-16,y2:H-pad,stroke:'var(--grid)'}));
 MK.forEach((m,i)=>{const v=vals[i];const h=(v/mx)*(H-pad-30);const x=x0+i*(bw+gap);const y=H-pad-h;
  const bar=el('rect',{x,y,width:bw,height:h,rx:5,fill:'var(--'+m+')'});
  bar.style.cursor='pointer';
  bar.addEventListener('mousemove',e=>showTip(e.clientX,e.clientY,`<b>${NAME[m]}</b><br>${label}: ${fmt(v)}`));
  bar.addEventListener('mouseleave',hideTip);svg.appendChild(bar);
  const lt=el('text',{x:x+bw/2,y:y-8,'text-anchor':'middle','font-size':15,'font-weight':i==best?700:500,fill:'var(--ink)'});lt.appendChild(txt(fmt(v)));svg.appendChild(lt);
  const nm=el('text',{x:x+bw/2,y:H-pad+20,'text-anchor':'middle','font-size':12.5,fill:'var(--sec)'});nm.appendChild(txt(NAME[m].split(' (')[0]));svg.appendChild(nm);
  if(i==best){const badge=el('text',{x:x+bw/2,y:H-pad+36,'text-anchor':'middle','font-size':10.5,'font-weight':600,fill:'var(--hybrid)'});badge.appendChild(txt('★ best'));svg.appendChild(badge);}});
 const ti=el('text',{x:pad,y:26,'font-size':13,'font-weight':650,fill:'var(--ink)'});ti.appendChild(txt(label+(lower?'  (lower better)':'  (higher better)')));svg.appendChild(ti);}
(function(){const c=document.getElementById('metric-controls');
 METRICS.forEach((mt,i)=>{const b=document.createElement('button');b.className='chip';b.textContent=mt[1];
  b.setAttribute('aria-pressed',i==0);b.onclick=()=>{curMetric=i;[...c.children].forEach((x,j)=>x.setAttribute('aria-pressed',j==i));drawBars();};c.appendChild(b);});})();

/* ---- method table ---- */
(function(){const rows=[['Magnitude slope',m=>M[m].overall.slope.toFixed(2)],
 ['Magnitude R²',m=>M[m].overall.r2.toFixed(2)],['RMSE (g·mm)',m=>M[m].overall.rmse.toFixed(1)],
 ['Phase err ° (U≥24)',m=>M[m].overall.phase_hi.toFixed(1)],['Cross-plane leak (g·mm)',m=>M[m].overall.crosstalk.toFixed(1)],
 ['Localization %',m=>(M[m].localization_acc*100).toFixed(0)],['Specificity %',m=>{const c=M[m].confusion;return (100*c.tn/(c.tn+c.fp)).toFixed(0);}],
 ['N1↔N2 agree (g·mm)',m=>M[m].cross_speed.mean_abs_gmm.toFixed(1)],['Trial run needed?',m=>m=='physics'?'yes / speed':'no']];
 let h='<table><tr><th>metric</th>'+MK.map(m=>`<th class=num style="color:var(--${m})">${NAME[m].split(' (')[0]}</th>`).join('')+'</tr>';
 rows.forEach(r=>{h+='<tr><td>'+r[0]+'</td>'+MK.map(m=>`<td class=num>${r[1](m)}</td>`).join('')+'</tr>';});
 document.getElementById('method-table').innerHTML=h+'</table>';})();

/* ---- scatter ---- */
let scMethod='hybrid';
function drawScatter(){const svg=document.getElementById('scatter');svg.innerHTML='';
 const W=560,H=520,pad=56,lim=66;const P=DATA.scatter[scMethod];
 const sx=v=>pad+v/lim*(W-pad-20),sy=v=>H-pad-v/lim*(H-pad-24);
 for(let g=0;g<=60;g+=15){svg.appendChild(el('line',{x1:sx(g),y1:sy(0),x2:sx(g),y2:sy(lim),stroke:'var(--grid)','stroke-width':.7}));
  svg.appendChild(el('line',{x1:sx(0),y1:sy(g),x2:sx(lim),y2:sy(g),stroke:'var(--grid)','stroke-width':.7}));
  const xl=el('text',{x:sx(g),y:H-pad+16,'text-anchor':'middle','font-size':11,fill:'var(--muted)'});xl.appendChild(txt(g));svg.appendChild(xl);
  const yl=el('text',{x:pad-8,y:sy(g)+4,'text-anchor':'end','font-size':11,fill:'var(--muted)'});yl.appendChild(txt(g));svg.appendChild(yl);}
 svg.appendChild(el('line',{x1:sx(0),y1:sy(0),x2:sx(lim),y2:sy(lim),stroke:'var(--muted)','stroke-dasharray':'5 4','stroke-width':1}));
 P.forEach(([t,e,d])=>{const mk=d==1?el('circle',{cx:sx(t),cy:sy(e),r:4.2}):el('rect',{x:sx(t)-3.6,y:sy(e)-3.6,width:7.2,height:7.2});
  mk.setAttribute('fill','var(--'+scMethod+')');mk.setAttribute('fill-opacity',.6);mk.setAttribute('stroke','var(--surface)');mk.setAttribute('stroke-width',.6);
  mk.addEventListener('mousemove',ev=>showTip(ev.clientX,ev.clientY,`disk ${d}<br>true ${t} → est ${e} g·mm`));mk.addEventListener('mouseleave',hideTip);svg.appendChild(mk);});
 const xt=el('text',{x:W/2,y:H-12,'text-anchor':'middle','font-size':12.5,fill:'var(--sec)'});xt.appendChild(txt('true added unbalance (g·mm)'));svg.appendChild(xt);
 const yt=el('text',{x:16,y:H/2,'text-anchor':'middle','font-size':12.5,fill:'var(--sec)',transform:`rotate(-90 16 ${H/2})`});yt.appendChild(txt('estimated (g·mm)'));svg.appendChild(yt);
 const o=M[scMethod].overall;
 document.getElementById('scatter-cap').innerHTML=`<b style="color:var(--${scMethod})">${NAME[scMethod]}</b> — R² ${o.r2.toFixed(2)}, slope ${o.slope.toFixed(2)}, RMSE ${o.rmse.toFixed(1)} g·mm. ● disk 1  ■ disk 2. Dashed = perfect 1:1.`;}
(function(){const c=document.getElementById('scatter-controls');
 MK.forEach((m,i)=>{const b=document.createElement('button');b.className='chip';b.textContent=NAME[m];
  b.style.borderColor='var(--'+m+')';b.setAttribute('aria-pressed',m==scMethod);
  b.onclick=()=>{scMethod=m;[...c.children].forEach(x=>x.setAttribute('aria-pressed',x===b));drawScatter();};c.appendChild(b);});})();

/* ---- orbit explorer ---- */
const OKEYS=Object.keys(DATA.orbits).sort();
(function(){const cs=document.getElementById('orbit-cond'),sp=document.getElementById('orbit-speed');
 const conds=[...new Set(OKEYS.map(k=>k.split('|')[0]))];
 conds.forEach(c=>{const o=document.createElement('option');o.value=c;
  const r=DATA.orbits[c+'|N1']||DATA.orbits[c+'|N2'];o.textContent=c+(r?'  · '+r.config:'');cs.appendChild(o);});
 ['N1','N2'].forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s+(s=='N1'?' · 1000 rpm':' · 1200 rpm');sp.appendChild(o);});
 cs.value=conds.find(c=>c=='A007')||conds[0];sp.value='N1';
 cs.onchange=sp.onchange=drawOrbit;drawOrbit();})();
function drawOrbit(){const cid=document.getElementById('orbit-cond').value,spd=document.getElementById('orbit-speed').value;
 const r=DATA.orbits[cid+'|'+spd];const svg=document.getElementById('orbit');svg.innerHTML='';
 if(!r){svg.appendChild(el('text',{x:0,y:0,'text-anchor':'middle',fill:'var(--muted)','font-size':.12},txt('no data')));return;}
 // scale to fit
 let mx=.05;[['plane1_x','plane1_y'],['plane2_x','plane2_y']].forEach(([px,py])=>{const X=r[px],Y=r[py];
  for(let a=0;a<6.3;a+=.3){mx=Math.max(mx,Math.abs(X[0]*Math.cos(a)-X[1]*Math.sin(a)),Math.abs(Y[0]*Math.cos(a)-Y[1]*Math.sin(a)));}});
 const S=0.9/mx;
 svg.appendChild(el('line',{x1:-1,y1:0,x2:1,y2:0,stroke:'var(--grid)','stroke-width':.006}));
 svg.appendChild(el('line',{x1:0,y1:-1,x2:0,y2:1,stroke:'var(--grid)','stroke-width':.006}));
 [['plane1_x','plane1_y','physics','plane 1 (disk 1)'],['plane2_x','plane2_y','hybrid','plane 2 (disk 2)']].forEach(([px,py,col,lab])=>{
  const X=r[px],Y=r[py];let d='';for(let a=0;a<=6.30;a+=.05){const x=(X[0]*Math.cos(a)-X[1]*Math.sin(a))*S;const y=-(Y[0]*Math.cos(a)-Y[1]*Math.sin(a))*S;d+=(a==0?'M':'L')+x.toFixed(3)+' '+y.toFixed(3)+' ';}
  svg.appendChild(el('path',{d,fill:'none',stroke:'var(--'+col+')','stroke-width':.02}));});
 // readout
 const est=DATA.ests[cid+'|'+spd]||{};
 const trueTxt=`true: disk1 ${r.U1[0].toFixed(0)} g·mm ∠${r.U1[1].toFixed(0)}° · disk2 ${r.U2[0].toFixed(0)} g·mm ∠${r.U2[1].toFixed(0)}°`;
 let h=`<div class="legend"><span><i class="sw" style="background:var(--physics)"></i>plane 1 (disk 1)</span><span><i class="sw" style="background:var(--hybrid)"></i>plane 2 (disk 2)</span></div>`;
 h+=`<p style="margin:6px 0 10px;color:var(--sec)"><b>${cid}</b> · ${r.config} · ${spd}<br>${trueTxt}</p>`;
 h+='<table><tr><th>estimate</th><th class=num>disk 1</th><th class=num>disk 2</th></tr>';
 MK.forEach(m=>{const e=est[m]||{};const u1=(e.U1==null?'–':e.U1),u2=(e.U2==null?'–':e.U2);h+=`<tr><td style="color:var(--${m});font-weight:600">${NAME[m].split(' (')[0]}</td><td class=num>${u1} g·mm</td><td class=num>${u2} g·mm</td></tr>`;});
 h+='</table>';document.getElementById('orbit-readout').innerHTML=h;}

/* ---- full blind validation ---- */
(function(){const bv=DATA.bench.blind_validation;const el=document.getElementById('blind-val');
 if(!bv||!el)return;
 const rows=[['Magnitude MAE (g·mm)',m=>bv.methods[m].mae.toFixed(1),true],
  ['Loaded-disk MAE (g·mm)',m=>bv.methods[m].mae_loaded.toFixed(0),true],
  ['Detection accuracy',m=>(bv.methods[m].detect_acc*100).toFixed(0)+'%',false],
  ['Specificity (reject empty)',m=>(bv.methods[m].specificity*100).toFixed(0)+'%',false],
  ['Sensitivity (find loaded)',m=>(bv.methods[m].sensitivity*100).toFixed(0)+'%',false]];
 let h=`<div style="font-weight:650;margin-bottom:6px">Blind accuracy — ${bv.n_loaded_cond} loaded + ${bv.n_balanced_cond} balanced conditions (${bv.n_points} disk-points)</div>`;
 h+='<table><tr><th>metric</th>'+MK.map(m=>`<th class=num style="color:var(--${m})">${NAME[m].split(' (')[0]}</th>`).join('')+'</tr>';
 rows.forEach(r=>{h+='<tr><td>'+r[0]+'</td>'+MK.map(m=>`<td class=num>${r[1](m)}</td>`).join('')+'</tr>';});
 el.innerHTML=h+'</table>';})();

/* ---- blind specificity ---- */
(function(){const bb=DATA.bench.blind_balanced;const el=document.getElementById('blind-spec');
 if(!bb||!el)return;
 let h=`<div style="font-weight:650;margin-bottom:6px">Specificity on ${bb.n_conditions} unseen balanced conditions (${bb.methods.hybrid.n} disk-points)</div>`;
 h+='<table><tr><th>method</th><th class=num>called empty</th><th class=num>specificity</th><th class=num>worst reading</th></tr>';
 MK.forEach(m=>{const s=bb.methods[m];const good=s.specificity>=0.99;
  h+=`<tr><td style="color:var(--${m});font-weight:600">${NAME[m].split(' (')[0]}</td><td class=num>${s.correct}/${s.n}</td>`
   +`<td class=num style="font-weight:700;color:${good?'var(--ok)':'var(--ink)'}">${(s.specificity*100).toFixed(0)}%</td>`
   +`<td class=num>${s.max_gmm.toFixed(1)} g·mm</td></tr>`;});
 el.innerHTML=h+'</table>';})();

/* ---- blind table ---- */
(function(){const B=DATA.blind;if(!B.length){document.getElementById('blind-table').innerHTML='';return;}
 const balSet=new Set((DATA.bench.blind_balanced||{}).conditions||[]);
 const agg={};B.forEach(r=>{const k=r.condition_id;(agg[k]=agg[k]||[]).push(r);});
 const mean=(a,f)=>a.reduce((s,r)=>s+(+r[f]),0)/a.length;
 let h='<div style="font-weight:650;margin:2px 0 6px">Per-condition hybrid estimate (mean over acqs)</div>';
 h+='<table><tr><th>cond</th><th class=num>hybrid D1</th><th class=num>hybrid D2</th><th>truth</th></tr>';
 Object.keys(agg).sort().forEach(k=>{const a=agg[k];const bal=balSet.has(k);
  h+=`<tr><td>${k}</td><td class=num>${mean(a,'hybrid_U1_gmm').toFixed(1)}</td><td class=num>${mean(a,'hybrid_U2_gmm').toFixed(1)}</td>`
   +`<td>${bal?'<span style="color:var(--ok);font-weight:600">balanced</span>':'<span style="color:var(--muted)">sealed</span>'}</td></tr>`;});
 document.getElementById('blind-table').innerHTML=h+'</table>';})();

drawBars();drawScatter();
"""


def main():
    payload, figs = build_payload()
    content = render(payload, figs)

    art = SCRATCH / "twin_dashboard.html"
    art.write_text(content, encoding="utf-8")

    standalone = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
                  "<meta name=viewport content='width=device-width,initial-scale=1'>"
                  + content + "</head></html>")
    (ANALYSIS / "dashboard.html").write_text(standalone, encoding="utf-8")
    kb = len(content) // 1024
    print(f"wrote {ANALYSIS/'dashboard.html'} (standalone) and {art} (artifact body) — {kb} KB")


if __name__ == "__main__":
    main()
