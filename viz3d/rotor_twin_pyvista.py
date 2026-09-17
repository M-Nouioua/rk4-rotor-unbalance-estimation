"""
RK-4 rotor 3D digital twin — native PyVista viewer (runs on the lab PC).

Same physics as the three.js viewer: the shaft bends and whirls from the measured
1X/2X/3X vectors at the two probe planes, interpolated along the shaft (zero at the
bearings); disks spin at the keyphasor phase with the heavy-spot marked.

Sources:
    python viz3d/rotor_twin_pyvista.py                 # playback viz3d/twin_state.json (n/p to cycle)
    python viz3d/rotor_twin_pyvista.py --file datasets/A008_N1_m1_a1_1784813244.npz
    python viz3d/rotor_twin_pyvista.py --live          # live from the NI DAQ (on the rig)
    python viz3d/rotor_twin_pyvista.py --sim            # synthetic source (no hardware)

Controls: drag to orbit · scroll to zoom · SPACE play/pause · n/p next/prev condition
· sliders for amplification & animation speed · q or close window to quit.

Needs: pip install pyvista   (live/file also need the project's acquisition deps)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

try:
    import pyvista as pv
except ImportError:
    sys.exit("PyVista not installed:  pip install pyvista")

MIL_MM = 0.0254
STATE = HERE / "twin_state.json"


# ---------------- physics (shared with the three.js viewer) ----------------
def defl_at(plane, th):
    """plane={'x':[[re,im]x3],'y':[[re,im]x3]} -> (y_vertical, z_horizontal) in mils."""
    Z = Y = 0.0
    for k in (1, 2, 3):
        vx, vy = plane["x"][k - 1], plane["y"][k - 1]
        Z += vx[0] * np.cos(k * th) - vx[1] * np.sin(k * th)
        Y += vy[0] * np.cos(k * th) - vy[1] * np.sin(k * th)
    return Y, Z


def lag_factory(zc):
    def lag(v, u):
        s = 0.0
        for i in range(4):
            t = v[i]
            for j in range(4):
                if j != i:
                    t *= (u - zc[j]) / (zc[i] - zc[j])
            s += t
        return s
    return lag


def rot_from_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c > 0.99999:
        return np.eye(3)
    if c < -0.99999:
        return -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_axis(axis, ang):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis; c, s = np.cos(ang), np.sin(ang)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


# ---------------- data sources ----------------
def load_state():
    return json.loads(STATE.read_text())


def phasors_from_npz(path):
    """Compute the measured, runout-compensated phasor structure from one recording."""
    from config import load_config
    from processing.order_tracking import harmonic_vectors, detect_pulses, instantaneous_rpm
    cfg = load_config()
    to_mils = 1000.0 / (cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"] * 25.4)
    kc = cfg["daq"]["keyphasor"]
    ro = np.load(ROOT / "datasets" / "runout_slowroll.npz", allow_pickle=True)
    rz = {p: {k: harmonic_vectors(ro[p], float(ro["fs"]), ro["keyphasor"], cfg)[k] * to_mils
              for k in (1, 2, 3)} for p in ["plane1_x", "plane1_y", "plane2_x", "plane2_y"]}
    d = np.load(path, allow_pickle=True); fs, kp = float(d["fs"]), d["keyphasor"]
    pu = detect_pulses(kp, fs, kc["threshold_v"], kc["edge"])
    rpm = float(np.median(instantaneous_rpm(pu))) if len(pu) > 2 else float("nan")

    def vec(p):
        h = harmonic_vectors(d[p], fs, kp, cfg)
        return [[float((h[k] * to_mils - rz[p][k]).real), float((h[k] * to_mils - rz[p][k]).imag)]
                for k in (1, 2, 3)]
    return {"id": pathlib.Path(path).stem[:10], "config": "recording", "rpm": rpm,
            "U1": {"mag_gmm": 0, "ang_deg": 0}, "U2": {"mag_gmm": 0, "ang_deg": 0},
            "planes": {"plane1": {"x": vec("plane1_x"), "y": vec("plane1_y")},
                       "plane2": {"x": vec("plane2_x"), "y": vec("plane2_y")}}}


class LiveSource:
    """Background NI-DAQ reader -> latest phasor state (order-tracked each block)."""
    def __init__(self, sim=False, dur=1.0):
        from gui.session import Session
        self.s = Session(); self.sim = sim; self.dur = dur
        self.state = load_state()["conditions"].get("BASE|N1")   # placeholder until first block
        self.lock = threading.Lock(); self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                rec = self.s.acquire(self.dur, simulate=self.sim, rpm=1000)
                f = self.s.features(rec)
                st = self._to_state(rec, f)
                with self.lock:
                    self.state = st
            except Exception as e:                       # keep the viewer alive on a DAQ hiccup
                print("live acquire error:", e); time.sleep(0.5)

    def _to_state(self, rec, f):
        from config import load_config
        from processing.order_tracking import harmonic_vectors
        cfg = load_config()
        to_mils = 1000.0 / (cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"] * 25.4)
        def vec(p):
            h = harmonic_vectors(rec[p], rec["fs"], rec["keyphasor"], cfg)
            return [[float((h[k] * to_mils).real), float((h[k] * to_mils).imag)] for k in (1, 2, 3)]
        return {"id": "LIVE", "config": "rig", "rpm": f["rpm"],
                "U1": {"mag_gmm": 0, "ang_deg": 0}, "U2": {"mag_gmm": 0, "ang_deg": 0},
                "planes": {"plane1": {"x": vec("plane1_x"), "y": vec("plane1_y")},
                           "plane2": {"x": vec("plane2_x"), "y": vec("plane2_y")}}}

    def get(self):
        with self.lock:
            return self.state


# ---------------- viewer ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="animate a single recording (.npz)")
    ap.add_argument("--live", action="store_true", help="live from the NI DAQ")
    ap.add_argument("--sim", action="store_true", help="synthetic live source")
    a = ap.parse_args()

    data = load_state()
    geom = data["geom"]
    L = geom["shaft"]["length"]; HALF = L / 2
    zc = [geom["bearings"][0], geom["planes"][0]["z"], geom["planes"][1]["z"], geom["bearings"][1]]
    lag = lag_factory(zc)
    Rbal = geom["balance_radius"]; nH = geom["n_holes"]

    # source selection
    live = None
    order = data["order"]
    idx = 0
    if a.file:
        cond = phasors_from_npz(a.file); mode = "file"
    elif a.live or a.sim:
        live = LiveSource(sim=a.sim); cond = live.get(); mode = "live"
    else:
        keys = list(order); idx = keys.index("A008|N1") if "A008|N1" in keys else 0
        cond = data["conditions"][keys[idx]]; mode = "playback"

    st = {"th": 0.0, "amp": 2500.0, "rps": 0.6, "play": True, "idx": (idx if mode == "playback" else 0)}

    pl = pv.Plotter(title="RK-4 Rotor 3D Twin (PyVista)")
    pl.set_background("#0c1118")
    pl.enable_anti_aliasing()

    # --- geometry ---
    def centerline(th):
        d1y, d1z = defl_at(cond["planes"]["plane1"], th)
        d2y, d2z = defl_at(cond["planes"]["plane2"], th)
        sc = MIL_MM * st["amp"]
        return [0, d1y * sc, d2y * sc, 0], [0, d1z * sc, d2z * sc, 0]

    def shaft_pts(th):
        vY, vZ = centerline(th)
        us = np.linspace(zc[0], zc[3], 60)      # between the bearings (no extrapolation)
        return np.array([[u - HALF, lag(vY, u), lag(vZ, u)] for u in us]), vY, vZ

    p0, vY, vZ = shaft_pts(0)
    shaft = pv.lines_from_points(p0)
    pl.add_mesh(shaft, color="#9fb0c0", render_lines_as_tubes=True, line_width=12,
                specular=0.6, name="shaft")

    disk_col = ["#4f9dff", "#2fc39a"]
    disk_actors, hole_pd, hole_local = [], [], []
    for i, d in enumerate(geom["disks"]):
        cyl = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0),
                          radius=d["od"] / 2, height=d["thick"], resolution=48)
        act = pl.add_mesh(cyl, color=disk_col[i], opacity=0.9, specular=0.3, name=f"disk{i}")
        disk_actors.append(act)
        loc = np.array([[d["thick"] / 2 + 1,
                         Rbal * np.cos(h * 2 * np.pi / nH),
                         Rbal * np.sin(h * 2 * np.pi / nH)] for h in range(nH)])
        hole_local.append(loc)
        pdh = pv.PolyData(loc.copy())
        pdh["heavy"] = np.zeros(nH)
        pl.add_mesh(pdh, scalars="heavy", cmap=["#2a3644", "#ff5b52"], clim=[0, 1],
                    render_points_as_spheres=True, point_size=14, show_scalar_bar=False,
                    name=f"holes{i}")
        hole_pd.append(pdh)

    for z in geom["bearings"]:
        pl.add_mesh(pv.Box(bounds=(z - HALF - 13, z - HALF + 13, -60, 0, -30, 30)),
                    color="#2b3644", name=f"brg{z}")
    for p in geom["planes"]:
        pl.add_mesh(pv.Cylinder(center=(p["z"] - HALF, 0, 30), direction=(0, 0, 1),
                                radius=3, height=26), color="#f2a24b", name=f"prz{p['z']}")
        pl.add_mesh(pv.Cylinder(center=(p["z"] - HALF, 30, 0), direction=(0, 1, 0),
                                radius=3, height=26), color="#f2a24b", name=f"pry{p['z']}")

    def set_heavy():
        for i, pd in enumerate(hole_pd):
            U = cond["U1"] if i == 0 else cond["U2"]
            hv = np.zeros(nH)
            if U["mag_gmm"] > 0.1:
                a_ = np.radians(U["ang_deg"])
                for h in range(nH):
                    if abs(((h * 2 * np.pi / nH - a_ + np.pi) % (2 * np.pi)) - np.pi) < 0.4:
                        hv[h] = 1
            pd["heavy"] = hv
    set_heavy()

    # --- controls ---
    def toggle():
        st["play"] = not st["play"]
    def nxt(step):
        if mode != "playback":
            return
        st["idx"] = (st["idx"] + step) % len(order)
        nonlocal_set(order[st["idx"]])
    def nonlocal_set(key):
        nonlocal cond
        cond = data["conditions"][key]; set_heavy()
    pl.add_key_event("space", toggle)
    pl.add_key_event("n", lambda: nxt(1))
    pl.add_key_event("p", lambda: nxt(-1))
    pl.add_slider_widget(lambda v: st.update(amp=v), [200, 8000], value=st["amp"],
                         title="amplification", pointa=(0.02, 0.92), pointb=(0.30, 0.92))
    pl.add_slider_widget(lambda v: st.update(rps=v), [0, 3], value=st["rps"],
                         title="anim rev/s", pointa=(0.02, 0.80), pointb=(0.30, 0.80))
    pl.add_axes()

    # --- animation loop ---
    pl.show(interactive_update=True, auto_close=False)
    last = time.time()
    try:
        while True:
            now = time.time(); dt = min(0.05, now - last); last = now
            if live is not None:
                cond = live.get()
            if st["play"]:
                st["th"] = (st["th"] + st["rps"] * 2 * np.pi * dt) % (2 * np.pi)
            th = st["th"]
            pts, vY, vZ = shaft_pts(th)
            shaft.points = pts
            for i, (act, d) in enumerate(zip(disk_actors, geom["disks"])):
                u = d["z"]
                c = np.array([u - HALF, lag(vY, u), lag(vZ, u)])
                e = 2.0
                T = np.array([1.0, (lag(vY, u + e) - lag(vY, u - e)) / (2 * e),
                              (lag(vZ, u + e) - lag(vZ, u - e)) / (2 * e)])
                R = rot_axis(T, th) @ rot_from_to(np.array([1.0, 0, 0]), T)
                M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c
                act.user_matrix = M
                hole_pd[i].points = (hole_local[i] @ R.T) + c
            pl.add_text(f"{cond['id']} · {cond.get('config','')}  |  "
                        f"{cond['rpm']:.0f} rpm  |  U1 {cond['U1']['mag_gmm']} g.mm  "
                        f"U2 {cond['U2']['mag_gmm']} g.mm   [{mode}]  space=play n/p=cond",
                        position="lower_left", font_size=10, color="#e8eef5", name="hud")
            pl.update()
            time.sleep(max(0.0, 1 / 60 - (time.time() - now)))
    except (RuntimeError, AttributeError):
        pass          # window closed
    if live is not None:
        live.run = False


if __name__ == "__main__":
    main()
