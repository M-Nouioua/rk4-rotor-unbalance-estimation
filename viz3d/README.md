# 3D Digital-Twin Viewers

3D visual twins of the RK-4 rotor. The shaft bends and whirls from the runout-compensated
1X/2X/3X vectors at the two probe planes, interpolated along the shaft (zero at the
bearings); the disks spin at the keyphasor phase with the mounted heavy-spot marked — a
*faithful reconstruction of the measured motion*, not a scripted animation.

**► `rotor_twin_threejs.html` is the chosen viewer.** It is fully offline and
self-contained: `three.js` + `OrbitControls` are vendored in `lib/` and the data is
embedded — just double-click, no internet or server needed. `rotor_twin_pyvista.py` is
kept as a native alternative (it can read the live NI DAQ).

| | `rotor_twin_threejs.html` ► | `rotor_twin_pyvista.py` |
|---|---|---|
| Runs in | any web browser (WebGL) | Python window on the lab PC |
| Best for | demos, sharing, remote, handoff | live on the rig |
| Live DAQ feed | not yet (playback of recorded states) | yes (`--live`) |
| Dependencies | none (offline, vendored in `lib/`) | `pip install pyvista` |

## Shared data

`twin_state.json` holds the rig **geometry** (from `config/rig.yaml`) and, for ~10
representative conditions × 2 speeds, the measured complex 1X/2X/3X vectors at each plane,
the true mounted unbalance, and rpm. Rebuild it from the recordings anytime:

```bash
python viz3d/make_twin_data.py        # -> viz3d/twin_state.json
python viz3d/build_threejs.py         # embeds the data into the html
```

## A) three.js (web) ► chosen viewer

Open **`viz3d/rotor_twin_threejs.html`** in a browser (double-click). Controls: pick
condition/speed, whirl amplification, animation speed, play/pause, orbit trail; drag to
orbit, scroll to zoom.

- **Fully offline** — `lib/three.min.js` + `lib/OrbitControls.js` are vendored and the data
  is embedded, so it needs no internet and no server. Keep the `lib/` folder next to the html.
- Rebuild after refreshing the data: `python viz3d/build_threejs.py`.

## B) PyVista (native / live on rig)

```bash
pip install pyvista
python viz3d/rotor_twin_pyvista.py                       # playback (n/p to cycle conditions)
python viz3d/rotor_twin_pyvista.py --file datasets/A008_N1_m1_a1_1784813244.npz   # one recording
python viz3d/rotor_twin_pyvista.py --sim                 # synthetic source, no hardware
python viz3d/rotor_twin_pyvista.py --live                # LIVE from the NI DAQ, on the rig
```
Controls: drag to orbit · scroll to zoom · **SPACE** play/pause · **n/p** next/prev
condition · sliders for amplification & animation speed · **q**/close to quit.

`--live` runs the project's acquisition + order tracking in a background thread and updates
the twin each block — this is the synchronized on-rig twin. (Needs the NI-DAQmx runtime and
`config/rig.yaml` pointing at the real device names.)

## Notes on faithfulness

- **Amplification**: real orbits are ~0.1–0.7 mil on a 560 mm shaft — invisible at true
  scale. The slider exaggerates lateral motion (default ×2500); rpm and orbit amplitudes in
  the readout are the *true* measured values.
- **Animation speed** is decoupled from real rpm (1000–1200 rpm is too fast to watch); the
  whirl *shape*, phase and direction are the measured ones.
- Currently uses **1X + 2X + 3X**; 1X (the unbalance response) dominates. Adding a live DAQ
  feed to the three.js viewer, or shaft mode-shape from ROSS for the interpolation, are the
  natural next steps once you pick a viewer.
