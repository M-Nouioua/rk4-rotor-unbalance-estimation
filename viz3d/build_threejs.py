"""
Build the self-contained three.js 3D digital-twin viewer.

Merges viz3d/twin_state.json into an HTML template -> viz3d/rotor_twin_threejs.html.
Open that file in a browser — it is FULLY OFFLINE: three.js + OrbitControls are vendored
in viz3d/lib/ and referenced locally, and the data is embedded, so it works with no
internet and no server (just double-click). Re-download the lib only if it's missing:
  curl -sL -o viz3d/lib/three.min.js https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
  curl -sL -o viz3d/lib/OrbitControls.js https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js

    python viz3d/make_twin_data.py       # refresh the data first
    python viz3d/build_threejs.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

TEMPLATE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>RK-4 Rotor 3D Twin (three.js)</title>
<style>
  :root{--bg:#0c1118;--panel:#141c26;--ink:#e8eef5;--mut:#8b98a8;--acc:#2bb7c6;
    --d1:#4f9dff;--d2:#2fc39a;--probe:#f2a24b;--heavy:#ff5b52;--line:#26313d;
    --mono:"Cascadia Code",Consolas,monospace;--ui:system-ui,"Segoe UI",sans-serif;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--ui);overflow:hidden}
  #app{position:fixed;inset:0}
  canvas{display:block}
  .panel{position:fixed;top:14px;left:14px;width:290px;background:rgba(20,28,38,.92);
    border:1px solid var(--line);border-radius:12px;padding:14px 15px;backdrop-filter:blur(6px)}
  .eyebrow{font:600 10.5px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--acc)}
  h1{font-size:16px;margin:6px 0 2px;font-weight:650}
  .sub{font-size:11.5px;color:var(--mut);margin-bottom:12px}
  label{display:block;font-size:11px;color:var(--mut);margin:11px 0 4px;letter-spacing:.02em;text-transform:uppercase}
  select,input[type=range]{width:100%}
  select{background:#0f1620;color:var(--ink);border:1px solid var(--line);border-radius:7px;
    padding:6px 8px;font-family:var(--ui);font-size:13px}
  .row{display:flex;gap:8px;align-items:center}
  .row>*{flex:1}
  button{background:#0f1620;color:var(--ink);border:1px solid var(--line);border-radius:7px;
    padding:7px;font-family:var(--mono);font-size:12px;cursor:pointer}
  button:hover{border-color:var(--acc);color:var(--acc)}
  button.on{background:var(--acc);color:#00252a;border-color:var(--acc);font-weight:700}
  .val{font-family:var(--mono);font-size:11px;color:var(--ink);float:right}
  .hud{position:fixed;bottom:14px;left:14px;background:rgba(20,28,38,.92);border:1px solid var(--line);
    border-radius:12px;padding:12px 15px;font-family:var(--mono);font-size:12px;min-width:290px}
  .hud .k{color:var(--mut)}
  .hud b{color:var(--ink);font-weight:600}
  .legend{position:fixed;top:14px;right:14px;background:rgba(20,28,38,.92);border:1px solid var(--line);
    border-radius:12px;padding:12px 14px;font-size:11.5px}
  .legend div{display:flex;align-items:center;gap:8px;margin:3px 0;color:var(--mut)}
  .sw{width:11px;height:11px;border-radius:3px;display:inline-block}
  .note{font-size:10.5px;color:var(--mut);margin-top:10px;line-height:1.4}
</style></head><body>
<div id="app"></div>

<div class="panel">
  <div class="eyebrow">KFUPM · IMR — digital twin</div>
  <h1>RK-4 Rotor · 3D Twin</h1>
  <div class="sub">Motion driven by measured 1X/2X/3X vectors — a faithful whirl, not an animation.</div>

  <label>Condition</label>
  <select id="cond"></select>
  <label>Speed</label>
  <select id="speed"></select>

  <label>Whirl amplification <span class="val" id="ampV"></span></label>
  <input type="range" id="amp" min="200" max="8000" step="100" value="2500">
  <label>Animation speed <span class="val" id="rpsV"></span></label>
  <input type="range" id="rps" min="0" max="3" step="0.05" value="0.6">
  <div class="row" style="margin-top:12px">
    <button id="play" class="on">⏸ Pause</button>
    <button id="trail">Orbit trail</button>
    <button id="reset">Reset view</button>
  </div>
  <div class="note">Drag to orbit · scroll to zoom. Deflection is shown ×amplification
  (real orbits are ~0.1–0.7 mil). Heavy-spot marker = mounted screw.</div>
</div>

<div class="legend">
  <div><span class="sw" style="background:var(--d1)"></span>disk 1</div>
  <div><span class="sw" style="background:var(--d2)"></span>disk 2</div>
  <div><span class="sw" style="background:var(--probe)"></span>probe plane</div>
  <div><span class="sw" style="background:var(--heavy)"></span>heavy spot</div>
</div>

<div class="hud" id="hud"></div>

<script src="lib/three.min.js"></script>
<script src="lib/OrbitControls.js"></script>
<script>
const DATA = __TWIN_DATA__;
const MIL_MM = 0.0254, ZC = [DATA.geom.bearings[0], DATA.geom.planes[0].z,
                             DATA.geom.planes[1].z, DATA.geom.bearings[1]];
const L = DATA.geom.shaft.length, HALF = L/2;
const COL = k=>getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();

// ---------- physics ----------
function deflAt(pl, th){                 // -> {y:vertical(mils), z:horizontal(mils)}
  let Z=0, Y=0;
  for(let k=1;k<=3;k++){const vx=pl.x[k-1], vy=pl.y[k-1];
    Z += vx[0]*Math.cos(k*th) - vx[1]*Math.sin(k*th);
    Y += vy[0]*Math.cos(k*th) - vy[1]*Math.sin(k*th);}
  return {y:Y, z:Z};
}
function lag(v, u){                       // Lagrange cubic through the 4 ZC control points
  let s=0; for(let i=0;i<4;i++){let t=v[i]; for(let j=0;j<4;j++) if(j!==i) t*=(u-ZC[j])/(ZC[i]-ZC[j]); s+=t;} return s;
}
function centerlineVecs(cond, th, amp){
  const d1=deflAt(cond.planes.plane1,th), d2=deflAt(cond.planes.plane2,th), sc=MIL_MM*amp;
  return {vY:[0,d1.y*sc,d2.y*sc,0], vZ:[0,d1.z*sc,d2.z*sc,0]};
}

// ---------- scene ----------
const app=document.getElementById('app');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio); app.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(COL('bg'));
const camera=new THREE.PerspectiveCamera(42,1,1,5000);
const CAM0=new THREE.Vector3(150,190,560);
camera.position.copy(CAM0);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,0.65));
const dir=new THREE.DirectionalLight(0xffffff,0.8); dir.position.set(200,400,300); scene.add(dir);
const dir2=new THREE.DirectionalLight(0x88aaff,0.3); dir2.position.set(-200,-100,-200); scene.add(dir2);
// ground grid
const grid=new THREE.GridHelper(900,18,0x223040,0x1a2530); grid.position.y=-90; scene.add(grid);

const mat=(c,metal=0.4,rough=0.45,op=1)=>new THREE.MeshStandardMaterial(
  {color:c,metalness:metal,roughness:rough,transparent:op<1,opacity:op});
const steel=mat(0x9fb0c0,0.7,0.35);

// shaft (tube rebuilt each frame)
let shaft=new THREE.Mesh(new THREE.BufferGeometry(), steel); scene.add(shaft);
const shaftR=Math.max(4, DATA.geom.shaft.od/2);

// disks + holes
const diskColors=[COL('d1'),COL('d2')];
const disks=DATA.geom.disks.map((d,i)=>{
  const g=new THREE.Group();
  const cyl=new THREE.Mesh(new THREE.CylinderGeometry(d.od/2,d.od/2,d.thick,48),
                           mat(new THREE.Color(diskColors[i]),0.35,0.5,0.92));
  cyl.rotation.z=Math.PI/2;              // cylinder axis -> local X (shaft axis)
  g.add(cyl);
  // rim
  const rim=new THREE.Mesh(new THREE.TorusGeometry(d.od/2,1.1,8,48),mat(0xffffff,0.2,0.6));
  rim.rotation.y=Math.PI/2; g.add(rim);
  // 8 holes on the face at balance radius
  const holes=[];
  for(let h=0;h<DATA.geom.n_holes;h++){
    const a=h*2*Math.PI/DATA.geom.n_holes;
    const m=new THREE.Mesh(new THREE.SphereGeometry(3.4,12,12),mat(0x2a3644,0.2,0.7));
    m.position.set(d.thick/2+1, DATA.geom.balance_radius*Math.cos(a), DATA.geom.balance_radius*Math.sin(a));
    m.userData.a=a; g.add(m); holes.push(m);
  }
  g.userData={holes, node:d};
  scene.add(g); return g;
});

// bearings (fixed pillow blocks)
DATA.geom.bearings.forEach(z=>{
  const b=new THREE.Mesh(new THREE.BoxGeometry(26,60,60),mat(0x2b3644,0.3,0.7));
  b.position.set(z-HALF,-30,0); scene.add(b);
  const cap=new THREE.Mesh(new THREE.CylinderGeometry(16,16,28,24),mat(0x3a4757,0.4,0.5));
  cap.rotation.z=Math.PI/2; cap.position.set(z-HALF,0,0); scene.add(cap);
});

// probes (x=horizontal->+Z, y=vertical->+Y) at each plane
DATA.geom.planes.forEach(p=>{
  [[0,0,46],[0,46,0]].forEach(off=>{
    const pr=new THREE.Mesh(new THREE.CylinderGeometry(3,3,26,16),mat(new THREE.Color(COL('probe')),0.5,0.4));
    const v=new THREE.Vector3(off[0],off[1],off[2]);
    pr.position.set(p.z-HALF, v.y?v.y-13:0, v.z?v.z-13:0);
    if(v.z) pr.rotation.x=Math.PI/2;
    scene.add(pr);
  });
});

// orbit trail rings (per plane)
const trails=DATA.geom.planes.map(()=> {
  const l=new THREE.Line(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({color:COL('acc'),transparent:true,opacity:0.55}));
  l.visible=false; scene.add(l); return l;
});

// ---------- state ----------
let cond=null, th=0, playing=true, amp=2500, rps=0.6, showTrail=false, last=performance.now();

function rebuildShaft(vY,vZ){
  const pts=[]; for(let i=0;i<=64;i++){const u=ZC[0]+(ZC[3]-ZC[0])*i/64; pts.push(new THREE.Vector3(u-HALF, lag(vY,u), lag(vZ,u)));}
  const curve=new THREE.CatmullRomCurve3(pts);
  shaft.geometry.dispose();
  shaft.geometry=new THREE.TubeGeometry(curve,64,shaftR,12,false);
}
function tangent(vY,vZ,u){const e=2; return new THREE.Vector3(1,(lag(vY,u+e)-lag(vY,u-e))/(2*e),(lag(vZ,u+e)-lag(vZ,u-e))/(2*e)).normalize();}

function updateTrails(){
  DATA.geom.planes.forEach((p,i)=>{
    trails[i].visible=showTrail; if(!showTrail) return;
    const pl=cond.planes[p.name==='plane1'?'plane1':'plane2'], sc=MIL_MM*amp, pts=[];
    for(let a=0;a<=64;a++){const t=a/64*2*Math.PI, d=deflAt(pl,t);
      pts.push(new THREE.Vector3(p.z-HALF, d.y*sc, d.z*sc));}
    trails[i].geometry.dispose(); trails[i].geometry=new THREE.BufferGeometry().setFromPoints(pts);
  });
}

function frame(now){
  const dt=Math.min(0.05,(now-last)/1000); last=now;
  if(playing) th=(th+rps*2*Math.PI*dt)%(2*Math.PI);
  if(cond){
    const {vY,vZ}=centerlineVecs(cond,th,amp);
    rebuildShaft(vY,vZ);
    disks.forEach(g=>{
      const u=g.userData.node.z, c=new THREE.Vector3(u-HALF,lag(vY,u),lag(vZ,u));
      g.position.copy(c);
      const T=tangent(vY,vZ,u);
      const q=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1,0,0),T);
      const qs=new THREE.Quaternion().setFromAxisAngle(T,th);
      g.quaternion.copy(qs.multiply(q));
    });
  }
  controls.update(); renderer.render(scene,camera);
  requestAnimationFrame(frame);
}

function setHeavySpots(){
  const U=[cond.U1,cond.U2];
  disks.forEach((g,i)=>{
    const loaded=U[i].mag_gmm>0.1, a=U[i].ang_deg*Math.PI/180;
    g.userData.holes.forEach(h=>{
      const isHeavy=loaded && Math.abs(((h.userData.a-a+Math.PI)%(2*Math.PI))-Math.PI)<0.2;
      h.material.color.set(isHeavy?COL('heavy'):0x2a3644);
      h.scale.setScalar(isHeavy?1.8:1);
    });
  });
}
function hud(){
  const p1=cond.planes.plane1, p2=cond.planes.plane2;
  const amp1x=(pl)=>Math.hypot(Math.hypot(pl.x[0][0],pl.x[0][1]),Math.hypot(pl.y[0][0],pl.y[0][1]));
  document.getElementById('hud').innerHTML=
    `<span class="k">condition</span> <b>${cond.id}</b> · ${cond.config} &nbsp; `+
    `<span class="k">rpm</span> <b>${cond.rpm.toFixed(0)}</b> (${cond.speed})<br>`+
    `<span class="k">unbalance</span> D1 <b>${cond.U1.mag_gmm} g·mm∠${cond.U1.ang_deg}°</b> · `+
    `D2 <b>${cond.U2.mag_gmm} g·mm∠${cond.U2.ang_deg}°</b><br>`+
    `<span class="k">1X orbit</span> plane1 <b>${amp1x(p1).toFixed(3)}</b> · `+
    `plane2 <b>${amp1x(p2).toFixed(3)}</b> mil (pk)`;
}
function load(){
  const key=document.getElementById('cond').value+'|'+document.getElementById('speed').value;
  cond=DATA.conditions[key]||DATA.conditions[DATA.order[0]];
  setHeavySpots(); updateTrails(); hud();
}

// ---------- UI ----------
(function(){
  const cs=document.getElementById('cond'), sp=document.getElementById('speed');
  const ids=[...new Set(DATA.order.map(k=>k.split('|')[0]))];
  const cfgOf=id=>{const k=DATA.order.find(x=>x.startsWith(id+'|')); return DATA.conditions[k].config;};
  ids.forEach(id=>{const o=document.createElement('option');o.value=id;o.textContent=id+' · '+cfgOf(id);cs.appendChild(o);});
  [...new Set(DATA.order.map(k=>k.split('|')[1]))].forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;sp.appendChild(o);});
  cs.value=ids.includes('A008')?'A008':ids[0];
  cs.onchange=sp.onchange=load;
  const ampEl=document.getElementById('amp'), rpsEl=document.getElementById('rps');
  const ampV=document.getElementById('ampV'), rpsV=document.getElementById('rpsV');
  const syncLabels=()=>{ampV.textContent='×'+amp; rpsV.textContent=rps.toFixed(2)+' rev/s';};
  ampEl.oninput=()=>{amp=+ampEl.value; syncLabels(); updateTrails(); hud();};
  rpsEl.oninput=()=>{rps=+rpsEl.value; syncLabels();};
  syncLabels();
  const play=document.getElementById('play');
  play.onclick=()=>{playing=!playing; play.textContent=playing?'⏸ Pause':'▶ Play'; play.classList.toggle('on',playing);};
  const trail=document.getElementById('trail');
  trail.onclick=()=>{showTrail=!showTrail; trail.classList.toggle('on',showTrail); updateTrails();};
  document.getElementById('reset').onclick=()=>{camera.position.copy(CAM0); controls.target.set(0,0,0);};
})();

function resize(){const w=innerWidth,h=innerHeight; renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();}
addEventListener('resize',resize); resize(); load();
requestAnimationFrame(frame);
</script></body></html>
"""


def main():
    data = json.loads((HERE / "twin_state.json").read_text())
    html = TEMPLATE.replace("__TWIN_DATA__", json.dumps(data, separators=(",", ":")))
    out = HERE / "rotor_twin_threejs.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
