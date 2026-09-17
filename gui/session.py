"""
Acquisition session backend (toolkit-agnostic, headless-testable).

Holds ALL campaign logic so the GUI can be a thin view:
  - load the test matrix (datasets/conditions.csv) and speeds (datasets/speeds.csv)
  - track progress from the acquisition log (datasets/index.csv)
  - auto-compute the next (mount, acq) indices for a condition/speed
  - acquire a record (real NI hardware via acquisition.daq, or a synthetic source)
  - compute live features: measured rpm, 1X/2X/3X per plane, synchronous orbits,
    and a peak-orbit safety value (mils) for the trip indicator
  - save the .npz and append one fully-labeled row to index.csv

No hardware or GUI imports here -> importable and unit-testable anywhere.
"""
from __future__ import annotations

import csv
import time
import pathlib
import numpy as np

from config import load_config, ROOT
from processing.order_tracking import detect_pulses, instantaneous_rpm, harmonic_vectors

DATASETS = ROOT / "datasets"
CHANNELS = ["keyphasor", "plane1_x", "plane1_y", "plane2_x", "plane2_y"]
PLANES = {"plane1": ("plane1_x", "plane1_y"), "plane2": ("plane2_x", "plane2_y")}

INDEX_FIELDS = ["run_id", "timestamp", "condition_id", "speed_id", "rpm_commanded",
                "rpm_measured", "mount_idx", "acq_idx", "file", "T_ambient_C", "notes"]


class Session:
    def __init__(self, cfg: dict | None = None, n_acq: int = 3):
        self.cfg = cfg or load_config()
        self.n_acq = n_acq
        self.conditions = self._read_csv(DATASETS / "conditions.csv")
        self.speeds = self._read_csv(DATASETS / "speeds.csv")
        self._by_id = {c["condition_id"]: c for c in self.conditions}

    # ---- loading -----------------------------------------------------
    @staticmethod
    def _read_csv(path: pathlib.Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def condition(self, cid: str) -> dict:
        return self._by_id[cid]

    def condition_ids(self) -> list[str]:
        return [c["condition_id"] for c in self.conditions]

    def speed_ids(self) -> list[str]:
        return [s["speed_id"] for s in self.speeds]

    def speed_rpm(self, speed_id: str) -> float | None:
        for s in self.speeds:
            if s["speed_id"] == speed_id and str(s.get("rpm", "")).strip():
                return float(s["rpm"])
        return None

    def applies_at(self, cid: str, speed_id: str) -> bool:
        """Whether a condition is scheduled at this speed (run_at_speeds column).

        'all' or blank -> every speed; otherwise a '|'/','-separated list of speed ids.
        Lets a subset (e.g. the N3 multi-speed study) run without cluttering the
        other speeds' progress.
        """
        ras = (self._by_id.get(cid, {}).get("run_at_speeds") or "all").strip()
        if ras in ("", "all"):
            return True
        return speed_id in [s.strip() for s in ras.replace(",", "|").split("|")]

    def target_acqs(self, cid: str, default: int) -> int:
        """Acquisitions to log for a condition at one speed. Uncertainty anchors need
        n_mount independent re-mounts (n_mount x n_acq); everything else uses `default`."""
        c = self._by_id.get(cid, {})
        if c.get("block") == "U_anchor":
            return int(c.get("n_mount") or 1) * self.n_acq
        return default

    def label_text(self, cid: str) -> str:
        """Human reminder of what to physically mount for this condition."""
        c = self.condition(cid)
        if c["block"] == "blind":
            return "SEALED — mount the colleague-prepared unknown; do not read labels."
        def part(disk):
            m, a = c[f"{disk}_mass_g"], c[f"{disk}_angle_deg"]
            return f"{disk}: {m} g @ {a}°" if m not in ("", "0.0", "0") else f"{disk}: —"
        return f"{c['config']}  |  {part('disk1')}   {part('disk2')}   (r={c['radius_mm']} mm)"

    # ---- progress / indexing ----------------------------------------
    def _log_rows(self) -> list[dict]:
        return self._read_csv(DATASETS / "index.csv")

    def next_indices(self, cid: str, speed_id: str) -> tuple[int, int]:
        """Next (mount, acq): fill n_acq acquisitions per mount, then new mount."""
        rows = [r for r in self._log_rows()
                if r["condition_id"] == cid and r["speed_id"] == speed_id]
        if not rows:
            return 1, 1
        last_mount = max(int(r["mount_idx"]) for r in rows)
        acqs = sum(1 for r in rows if int(r["mount_idx"]) == last_mount)
        return (last_mount, acqs + 1) if acqs < self.n_acq else (last_mount + 1, 1)

    def progress(self) -> dict:
        rows = self._log_rows()
        done = len(rows)
        planned = 0
        for c in self.conditions:
            nm = int(c.get("n_mount", 3) or 3)
            planned += len(self.speeds) * nm * self.n_acq
        per_cond: dict[str, int] = {}
        for r in rows:
            per_cond[r["condition_id"]] = per_cond.get(r["condition_id"], 0) + 1
        return {"done": done, "planned": planned, "per_condition": per_cond}

    # ---- acquisition -------------------------------------------------
    def acquire(self, duration_s: float, simulate: bool = False,
                rpm: float = 3000.0, seed: int = 0) -> dict:
        if simulate:
            return self._synthetic(duration_s, rpm, seed)
        from acquisition.daq import acquire_record
        return acquire_record(self.cfg, duration_s)

    def _synthetic(self, duration_s, rpm, seed) -> dict:
        rng = np.random.default_rng(seed)
        fs = self.cfg["daq"]["sample_rate_hz"]
        t = np.arange(int(duration_s * fs)) / fs
        f = rpm / 60.0
        kp = -5.0 * ((f * t) % 1.0 < 0.02).astype(float)   # negative-going, like the real rig
        def ch(a1, p1, a2=0.0, p2=0.0):
            return (a1 * np.sin(2 * np.pi * f * t + p1)
                    + a2 * np.sin(2 * np.pi * 2 * f * t + p2)
                    + 0.02 * rng.standard_normal(t.size))
        # Modest amplitudes (~1.5-2 mils) so Simulate mode is realistic and does
        # not constantly fire the orbit safety trip.
        return {"keyphasor": kp, "fs": fs,
                "plane1_x": ch(0.35, 0.2, 0.06, 1.0), "plane1_y": ch(0.32, 0.2 - np.pi / 2, 0.055, 0.5),
                "plane2_x": ch(0.28, 1.4, 0.05, 0.3), "plane2_y": ch(0.26, 1.4 - np.pi / 2, 0.045, 0.1)}

    # ---- feature extraction -----------------------------------------
    def _orbit_from_harmonics(self, ox: dict, oy: dict, npts: int = 720):
        """Reconstruct one synchronous revolution from complex harmonic vectors."""
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        x = sum(np.real(v * np.exp(1j * k * th)) for k, v in ox.items())
        y = sum(np.real(v * np.exp(1j * k * th)) for k, v in oy.items())
        return x, y

    def _volts_to_mils(self, v: float) -> float:
        # sensitivity mV/µm -> mV/mil = *25.4 ; mil = V*1000 / (mV/mil)
        mv_per_mil = self.cfg["daq"]["channels"]["plane1_x"]["sensitivity_mV_per_um"] * 25.4
        return v * 1000.0 / mv_per_mil

    def features(self, rec: dict) -> dict:
        fs = rec["fs"]
        kp = rec["keyphasor"]
        kc = self.cfg["daq"]["keyphasor"]
        pulses = detect_pulses(kp, fs, kc["threshold_v"], kc["edge"])
        rpm = float(np.median(instantaneous_rpm(pulses))) if len(pulses) > 2 else float("nan")

        out = {"rpm": rpm, "planes": {}, "max_orbit_mils": 0.0}
        for plane, (cx, cy) in PLANES.items():
            ox = harmonic_vectors(rec[cx], fs, kp, self.cfg)
            oy = harmonic_vectors(rec[cy], fs, kp, self.cfg)
            x, y = self._orbit_from_harmonics(ox, oy)
            pk_mils = self._volts_to_mils(float(np.max(np.hypot(x, y))))
            out["planes"][plane] = {
                "orders": {k: (abs(ox[k]), np.degrees(np.angle(ox[k]))) for k in ox},
                "orbit_x": x, "orbit_y": y, "peak_mils": pk_mils,
            }
            out["max_orbit_mils"] = max(out["max_orbit_mils"], pk_mils)
        return out

    def orbit_trip(self, feats: dict) -> bool:
        """True if peak orbit exceeds the safety limit (default half-clearance)."""
        limit = float(self.cfg.get("safety", {}).get("orbit_trip_mils", 5.0))
        return feats["max_orbit_mils"] > limit

    # ---- persistence -------------------------------------------------
    def save(self, rec: dict, cid: str, speed_id: str, rpm_cmd: float,
             mount: int, acq: int, rpm_meas: float, T_ambient: str = "",
             notes: str = "") -> dict:
        DATASETS.mkdir(exist_ok=True)
        stamp = int(time.time())
        run_id = f"{cid}_{speed_id}_m{mount}_a{acq}_{stamp}"
        fname = f"{run_id}.npz"
        arrays = {k: rec[k] for k in CHANNELS if k in rec}
        np.savez(DATASETS / fname, fs=rec["fs"], condition_id=cid,
                 speed_id=speed_id, rpm_commanded=rpm_cmd, rpm_measured=rpm_meas,
                 mount_idx=mount, acq_idx=acq, **arrays)
        row = {
            "run_id": run_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "condition_id": cid, "speed_id": speed_id, "rpm_commanded": rpm_cmd,
            "rpm_measured": round(rpm_meas, 1), "mount_idx": mount, "acq_idx": acq,
            "file": fname, "T_ambient_C": T_ambient, "notes": notes,
        }
        self._append_index(row)
        return {"path": str(DATASETS / fname), "row": row}

    def _append_index(self, row: dict) -> None:
        path = DATASETS / "index.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def delete_condition(self, cid: str) -> int:
        """Remove all logged runs (index rows + .npz files) for a condition so it
        can be redone. Returns how many runs were removed."""
        rows = self._log_rows()
        removed = [r for r in rows if r["condition_id"] == cid]
        for r in removed:
            f = DATASETS / r["file"]
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass
        keep = [r for r in rows if r["condition_id"] != cid]
        with open(DATASETS / "index.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
            w.writeheader(); w.writerows(keep)
        return len(removed)
