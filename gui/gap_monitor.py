"""
Live probe health / gap monitor (simple).

Shows every channel's LIVE level against its target range so you can adjust a
probe and watch it settle into the green band.

IMPORTANT: the NI-9234 is AC-coupled, so this reads the *dynamic* signal level,
NOT the DC gap voltage (set the actual gap with a DMM on the Proximitor DC/GAP
output, target ~-10 V). But a correctly gapped, on-target proximity probe reads
LOW and quiet here; an out-of-range / mis-aimed probe reads HIGH and oscillates.
Adjust until the bar is GREEN and the oscillation line disappears.

Run:
    python -m gui.gap_monitor        # real hardware
    python -m scripts.monitor        # same
    (add --sim for a no-hardware demo)

Needs: pip install pyqt5
"""
from __future__ import annotations

import sys
import numpy as np

try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    sys.exit("Monitor needs PyQt5:  pip install pyqt5")

from config import load_config
from acquisition.daq import CHANNEL_ORDER
from processing.spectra import fft_spectrum
from processing.order_tracking import detect_pulses, instantaneous_rpm

# thresholds on AC RMS (V) — from observed good (~0.01-0.05) vs bad (~0.4) probes
GOOD_V = 0.06
WATCH_V = 0.15
BAR_MAX_MV = 500          # progress-bar full scale (mV rms)
TONE_MIN_HZ = 200         # look for oscillation tones above this
TONE_FLAG_V = 0.05        # tone bigger than this = oscillating
BLOCK_S = 0.25            # acquisition block per update

GREEN, AMBER, RED, GREY = "#27ae60", "#e69500", "#e5343a", "#8a94a2"


def classify(rms):
    if rms <= GOOD_V:
        return "IN RANGE", GREEN
    if rms <= WATCH_V:
        return "MARGINAL", AMBER
    return "OUT OF RANGE", RED


class MonitorWorker(QtCore.QObject):
    metrics = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, cfg, simulate):
        super().__init__()
        self.cfg = cfg
        self.simulate = simulate
        self.timer = None
        self.acq = None

    @QtCore.pyqtSlot()
    def on_started(self):
        try:
            if not self.simulate:
                from acquisition.daq import RK4Acquisition
                self.acq = RK4Acquisition(self.cfg)
                self.acq.block = int(self.acq.fs * BLOCK_S)
                self.acq.open()
                self.acq.task.start()
        except Exception as e:
            self.error.emit(f"DAQ open failed: {e}")
            return
        self.timer = QtCore.QTimer()
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _read(self):
        fs = self.cfg["daq"]["sample_rate_hz"]
        if self.simulate:
            n = int(fs * BLOCK_S)
            t = np.arange(n) / fs
            rng = np.random.default_rng()
            def quiet():
                return 0.02 * rng.standard_normal(n)
            # plane1_y simulated as out-of-range: big 1634 Hz oscillation
            return {"fs": fs, "keyphasor": quiet(),
                    "plane1_x": quiet() + 0.03 * np.sin(2 * np.pi * 1634 * t),
                    "plane1_y": quiet() + 0.55 * np.sin(2 * np.pi * 1634 * t),
                    "plane2_x": quiet(), "plane2_y": quiet()}
        data = self.acq.read_block()
        return {name: data[i] for i, name in enumerate(CHANNEL_ORDER)} | \
               {"fs": self.acq.fs}

    def _tick(self):
        try:
            rec = self._read()
            fs = rec["fs"]
            out = {}
            for name in CHANNEL_ORDER:
                x = np.asarray(rec[name], dtype=float)
                rms = float(np.sqrt(np.mean(x ** 2)))
                freqs, amp = fft_spectrum(x, fs)
                m = freqs >= TONE_MIN_HZ
                if m.any():
                    i = int(np.argmax(amp[m]))
                    tone_hz, tone_v = float(freqs[m][i]), float(amp[m][i])
                else:
                    tone_hz = tone_v = 0.0
                out[name] = {"rms": rms, "tone_hz": tone_hz, "tone_v": tone_v}
            kc = self.cfg["daq"]["keyphasor"]
            pulses = detect_pulses(rec["keyphasor"], fs, kc["threshold_v"], kc["edge"])
            out["keyphasor"]["pulses"] = len(pulses)
            out["keyphasor"]["rpm"] = (float(np.median(instantaneous_rpm(pulses)))
                                       if len(pulses) > 2 else 0.0)
            self.metrics.emit(out)
        except Exception as e:
            self.error.emit(str(e))

    @QtCore.pyqtSlot()
    def stop(self):
        if self.timer:
            self.timer.stop()
        if self.acq is not None:
            try:
                self.acq.task.stop()
            except Exception:
                pass
            self.acq.close()
            self.acq = None


class MonitorGUI(QtWidgets.QMainWindow):
    sigStop = QtCore.pyqtSignal()

    LABELS = {
        "keyphasor": "Keyphasor  (Mod1/ai0)",
        "plane1_x": "Plane 1 · X  (Mod1/ai1)",
        "plane1_y": "Plane 1 · Y  (Mod1/ai2)",
        "plane2_x": "Plane 2 · X  (Mod1/ai3)",
        "plane2_y": "Plane 2 · Y  (Mod2/ai0)",
    }

    def __init__(self, simulate=False):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle("RK4 — Probe Gap / Health Monitor")
        self.resize(720, 430)
        self.rows = {}
        self._build()
        self._start_worker(simulate)

    def _build(self):
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root); v.setSpacing(10)

        note = QtWidgets.QLabel(
            "AC signal level (the 9234 can't read DC gap). A correctly gapped, "
            "on-target probe reads LOW & quiet → GREEN. Adjust until in range.\n"
            f"Target:  IN RANGE ≤ {GOOD_V*1000:.0f} mV   ·   MARGINAL ≤ "
            f"{WATCH_V*1000:.0f} mV   ·   OUT OF RANGE > {WATCH_V*1000:.0f} mV")
        note.setWordWrap(True); note.setStyleSheet("color:#555; padding:2px;")
        v.addWidget(note)

        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(9)
        hdr = ["Channel", "Level", "", "Status", "Oscillation"]
        for c, h in enumerate(hdr):
            lab = QtWidgets.QLabel(h); lab.setStyleSheet("font-weight:600;color:#888;font-size:11px;")
            grid.addWidget(lab, 0, c)

        for r, name in enumerate(CHANNEL_ORDER, start=1):
            nm = QtWidgets.QLabel(self.LABELS[name]); nm.setStyleSheet("font-weight:600;")
            val = QtWidgets.QLabel("—"); val.setMinimumWidth(90)
            val.setStyleSheet("font-family:monospace;font-size:14px;")
            bar = QtWidgets.QProgressBar(); bar.setRange(0, BAR_MAX_MV); bar.setTextVisible(False)
            bar.setFixedWidth(220); bar.setFixedHeight(18)
            st = QtWidgets.QLabel("—"); st.setMinimumWidth(110); st.setStyleSheet("font-weight:600;")
            osc = QtWidgets.QLabel(""); osc.setStyleSheet("font-family:monospace;color:#777;font-size:12px;")
            for c, w in enumerate((nm, val, bar, st, osc)):
                grid.addWidget(w, r, c)
            self.rows[name] = {"val": val, "bar": bar, "st": st, "osc": osc}
        v.addLayout(grid)
        v.addStretch(1)
        self.status = QtWidgets.QLabel("starting…"); self.status.setStyleSheet("color:#888;")
        v.addWidget(self.status)

    def _start_worker(self, simulate):
        self.thread = QtCore.QThread(self)
        self.worker = MonitorWorker(self.cfg, simulate)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.on_started)
        self.worker.metrics.connect(self._update)
        self.worker.error.connect(lambda m: self.status.setText(f"error: {m}"))
        self.sigStop.connect(self.worker.stop)
        self.thread.start()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)
        self.status.setText("monitoring… (simulate)" if simulate else "monitoring live DAQ…")

    @QtCore.pyqtSlot(dict)
    def _update(self, m):
        for name, d in m.items():
            row = self.rows[name]
            rms = d["rms"]; mv = rms * 1000.0
            if name == "keyphasor":
                rpm = d.get("rpm", 0.0); pulses = d.get("pulses", 0)
                row["val"].setText(f"{mv:6.1f} mV")
                row["bar"].setValue(int(min(mv, BAR_MAX_MV)))
                row["bar"].setStyleSheet(self._bar_css(GREY))
                row["st"].setText(f"{pulses} pulse/s"); row["st"].setStyleSheet(f"color:{GREY};font-weight:600;")
                row["osc"].setText(f"{rpm:.0f} rpm" if rpm else "rotor stopped")
                continue
            label, color = classify(rms)
            row["val"].setText(f"{mv:6.1f} mV")
            row["bar"].setValue(int(min(mv, BAR_MAX_MV)))
            row["bar"].setStyleSheet(self._bar_css(color))
            row["st"].setText(label); row["st"].setStyleSheet(f"color:{color};font-weight:700;")
            if d["tone_v"] > TONE_FLAG_V:
                row["osc"].setText(f"⚠ {d['tone_hz']:.0f} Hz  {d['tone_v']*1000:.0f} mV")
            else:
                row["osc"].setText("—")

    @staticmethod
    def _bar_css(color):
        return (f"QProgressBar{{background:#eee;border:1px solid #ccc;border-radius:4px;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:3px;}}")

    def _shutdown(self):
        if getattr(self, "_down", False):
            return
        self._down = True
        self.sigStop.emit()
        self.thread.quit(); self.thread.wait(3000)

    def closeEvent(self, e):
        self._shutdown(); super().closeEvent(e)


def main(simulate=False):
    app = QtWidgets.QApplication(sys.argv)
    w = MonitorGUI(simulate=simulate or ("--sim" in sys.argv)); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
