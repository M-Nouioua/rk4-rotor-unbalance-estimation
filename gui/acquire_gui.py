"""
RK4 imbalance acquisition GUI (thin view over gui.session.Session).

Pick a condition -> the panel tells you what mass to mount -> LIVE shows real-time
rpm / 1X-2X / orbits with a red safety trip -> CAPTURE records the full block, saves
the .npz, and appends a labelled row to datasets/index.csv (mount/acq auto-advance).

Acquisition (the slow, blocking part) runs on a background QThread worker, so the
window stays responsive during real-hardware reads. Live frames are driven by a timer
ON THE WORKER THREAD (so Stop/Capture commands are still delivered between blocks);
results are marshaled back to the UI via signals. CAPTURE auto-pauses then resumes live.

Run:
    python -m gui.acquire_gui          # or:  python -m scripts.acquire

Needs:  pip install pyqt5 pyqtgraph    (plus nidaqmx for real hardware)
Tick "Simulate" to drive it with synthetic data when no DAQ is connected.
"""
from __future__ import annotations

import sys

try:
    from PyQt5 import QtWidgets, QtCore
    import pyqtgraph as pg
except ImportError:
    sys.exit("GUI needs PyQt5 + pyqtgraph:  pip install pyqt5 pyqtgraph")

from gui.session import Session

TRIP_RED = "#c0392b"
OK_GREEN = "#27ae60"
LIVE_BLOCK_S = 0.5          # short block for live frames (keeps stop/capture latency low)


class AcquisitionWorker(QtCore.QObject):
    """Runs blocking acquisition on a background thread. Owns a worker-thread timer."""
    frameReady = QtCore.pyqtSignal(dict)
    captureDone = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, session: Session):
        super().__init__()
        self.s = session
        self.timer: QtCore.QTimer | None = None
        self._live = {"simulate": True, "rpm": 1500.0}

    @QtCore.pyqtSlot()
    def on_started(self):
        # Created here so the timer has worker-thread affinity.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)

    @QtCore.pyqtSlot(dict)
    def start_live(self, params):
        self._live = params
        if self.timer and not self.timer.isActive():
            self.timer.start()

    @QtCore.pyqtSlot()
    def stop_live(self):
        if self.timer:
            self.timer.stop()

    def _tick(self):
        try:
            rec = self.s.acquire(LIVE_BLOCK_S, simulate=self._live["simulate"],
                                 rpm=self._live["rpm"])
            self.frameReady.emit(self.s.features(rec))
        except Exception as e:
            self.error.emit(str(e))

    @QtCore.pyqtSlot(dict)
    def capture(self, p):
        was_live = bool(self.timer and self.timer.isActive())
        if self.timer:
            self.timer.stop()
        try:
            rec = self.s.acquire(p["dur"], simulate=p["simulate"], rpm=p["rpm"])
            feats = self.s.features(rec)
            self.captureDone.emit({"rec": rec, "feats": feats,
                                   "trip": self.s.orbit_trip(feats), "params": p})
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if was_live and self.timer:
                self.timer.start()


class AcquireGUI(QtWidgets.QMainWindow):
    sigStartLive = QtCore.pyqtSignal(dict)
    sigStopLive = QtCore.pyqtSignal()
    sigCapture = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.s = Session()
        self.setWindowTitle("RK4 — Imbalance Acquisition")
        self.resize(1150, 640)
        self._build()
        self._on_condition_changed()
        self._start_worker()

    # ---- worker/thread ----------------------------------------------
    def _start_worker(self):
        self.thread = QtCore.QThread(self)
        self.worker = AcquisitionWorker(self.s)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.on_started)
        self.sigStartLive.connect(self.worker.start_live)
        self.sigStopLive.connect(self.worker.stop_live)
        self.sigCapture.connect(self.worker.capture)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.captureDone.connect(self._on_capture_done)
        self.worker.error.connect(self._on_error)
        self.thread.start()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)   # clean stop on any quit path

    def _live_params(self):
        return {"simulate": self.chk_sim.isChecked(), "rpm": self.spn_rpm.value()}

    # ---- layout ------------------------------------------------------
    def _build(self):
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        h = QtWidgets.QHBoxLayout(root)

        form = QtWidgets.QFormLayout()
        self.cbo_cond = QtWidgets.QComboBox(); self.cbo_cond.addItems(self.s.condition_ids())
        self.cbo_cond.currentTextChanged.connect(self._on_condition_changed)
        self.lbl_mount = QtWidgets.QLabel(); self.lbl_mount.setWordWrap(True)
        self.lbl_mount.setStyleSheet("font-weight:bold; padding:6px; background:#f4f4f4;")

        self.cbo_speed = QtWidgets.QComboBox(); self.cbo_speed.addItems(self.s.speed_ids() or ["N1"])
        self.cbo_speed.currentTextChanged.connect(self._on_speed_changed)
        self.spn_rpm = QtWidgets.QDoubleSpinBox(); self.spn_rpm.setRange(0, 15000); self.spn_rpm.setValue(1500)
        self.spn_mount = QtWidgets.QSpinBox(); self.spn_mount.setRange(1, 999)
        self.spn_acq = QtWidgets.QSpinBox(); self.spn_acq.setRange(1, 99)
        self.spn_dur = QtWidgets.QDoubleSpinBox(); self.spn_dur.setRange(1, 120); self.spn_dur.setValue(10)
        self.chk_sim = QtWidgets.QCheckBox("Simulate (no DAQ)"); self.chk_sim.setChecked(True)
        self.txt_notes = QtWidgets.QLineEdit()

        form.addRow("Condition", self.cbo_cond)
        form.addRow("Mount", self.lbl_mount)
        form.addRow("Speed", self.cbo_speed)
        form.addRow("Commanded rpm", self.spn_rpm)
        form.addRow("Mount idx", self.spn_mount)
        form.addRow("Acq idx", self.spn_acq)
        form.addRow("Duration (s)", self.spn_dur)
        form.addRow("", self.chk_sim)
        form.addRow("Notes", self.txt_notes)

        self.btn_live = QtWidgets.QPushButton("Start LIVE"); self.btn_live.setCheckable(True)
        self.btn_live.toggled.connect(self._toggle_live)
        self.btn_cap = QtWidgets.QPushButton("CAPTURE ⏺")
        self.btn_cap.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn_cap.clicked.connect(self._capture)
        self.lbl_prog = QtWidgets.QLabel()

        left = QtWidgets.QVBoxLayout()
        left.addLayout(form); left.addWidget(self.btn_live); left.addWidget(self.btn_cap)
        left.addWidget(self.lbl_prog); left.addStretch(1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setFixedWidth(330)
        h.addWidget(lw)

        right = QtWidgets.QVBoxLayout()
        self.lbl_rpm = QtWidgets.QLabel("rpm: —"); self.lbl_rpm.setStyleSheet("font-size:20px;")
        self.lbl_safe = QtWidgets.QLabel("orbit: — mils")
        self.lbl_safe.setStyleSheet(f"font-size:18px; color:{OK_GREEN};")
        topline = QtWidgets.QHBoxLayout(); topline.addWidget(self.lbl_rpm); topline.addStretch(1); topline.addWidget(self.lbl_safe)
        right.addLayout(topline)

        self.lbl_orders = QtWidgets.QLabel("1X / 2X: —"); self.lbl_orders.setStyleSheet("font-family:monospace;")
        right.addWidget(self.lbl_orders)

        plots = QtWidgets.QHBoxLayout()
        self.p1 = self._orbit_plot("Plane 1 (before disk 1)")
        self.p2 = self._orbit_plot("Plane 2 (after disk 2)")
        self.c1 = self.p1.plot(pen=pg.mkPen("#2980b9", width=2))
        self.c2 = self.p2.plot(pen=pg.mkPen("#8e44ad", width=2))
        plots.addWidget(self.p1); plots.addWidget(self.p2)
        right.addLayout(plots)
        h.addLayout(right, 1)

        self._refresh_progress()

    def _orbit_plot(self, title):
        w = pg.PlotWidget(title=title)
        w.setAspectLocked(True); w.showGrid(x=True, y=True)
        w.setLabel("bottom", "X"); w.setLabel("left", "Y")
        return w

    # ---- widget events ----------------------------------------------
    def _on_condition_changed(self, *_):
        cid = self.cbo_cond.currentText()
        self.lbl_mount.setText(self.s.label_text(cid))
        self._sync_indices()

    def _on_speed_changed(self, *_):
        rpm = self.s.speed_rpm(self.cbo_speed.currentText())
        if rpm:
            self.spn_rpm.setValue(rpm)
        self._sync_indices()

    def _sync_indices(self):
        m, a = self.s.next_indices(self.cbo_cond.currentText(), self.cbo_speed.currentText())
        self.spn_mount.setValue(m); self.spn_acq.setValue(a)

    def _refresh_progress(self):
        p = self.s.progress()
        self.lbl_prog.setText(f"progress: {p['done']} / {p['planned']} acquisitions")

    def _toggle_live(self, on):
        self.btn_live.setText("Stop LIVE" if on else "Start LIVE")
        (self.sigStartLive.emit(self._live_params()) if on else self.sigStopLive.emit())

    def _capture(self):
        self.btn_cap.setEnabled(False)
        self.statusBar().showMessage("acquiring…")
        self.sigCapture.emit({
            "simulate": self.chk_sim.isChecked(), "rpm": self.spn_rpm.value(),
            "dur": self.spn_dur.value(), "cid": self.cbo_cond.currentText(),
            "sid": self.cbo_speed.currentText(), "mount": self.spn_mount.value(),
            "acq": self.spn_acq.value(), "notes": self.txt_notes.text(),
        })

    # ---- worker callbacks (UI thread) -------------------------------
    def _update_view(self, feats):
        self.lbl_rpm.setText(f"rpm: {feats['rpm']:.0f}")
        trip = self.s.orbit_trip(feats)
        self.lbl_safe.setText(f"orbit: {feats['max_orbit_mils']:.2f} mils" + ("  ⚠ TRIP" if trip else ""))
        self.lbl_safe.setStyleSheet(f"font-size:18px; color:{TRIP_RED if trip else OK_GREEN};")
        d1 = feats["planes"]["plane1"]["orders"]; d2 = feats["planes"]["plane2"]["orders"]
        self.lbl_orders.setText(
            f"P1  1X {d1[1][0]:.3f}∠{d1[1][1]:+6.1f}°   2X {d1[2][0]:.3f}∠{d1[2][1]:+6.1f}°\n"
            f"P2  1X {d2[1][0]:.3f}∠{d2[1][1]:+6.1f}°   2X {d2[2][0]:.3f}∠{d2[2][1]:+6.1f}°")
        self.c1.setData(feats["planes"]["plane1"]["orbit_x"], feats["planes"]["plane1"]["orbit_y"])
        self.c2.setData(feats["planes"]["plane2"]["orbit_x"], feats["planes"]["plane2"]["orbit_y"])

    @QtCore.pyqtSlot(dict)
    def _on_frame(self, feats):
        self._update_view(feats)

    @QtCore.pyqtSlot(dict)
    def _on_capture_done(self, payload):
        feats, p = payload["feats"], payload["params"]
        self._update_view(feats)
        if payload["trip"] and QtWidgets.QMessageBox.warning(
                self, "Orbit over safety limit",
                f"Peak orbit {feats['max_orbit_mils']:.2f} mils exceeds the trip limit.\nSave anyway?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel) != QtWidgets.QMessageBox.Save:
            self.btn_cap.setEnabled(True); return
        res = self.s.save(payload["rec"], p["cid"], p["sid"], p["rpm"],
                          p["mount"], p["acq"], feats["rpm"], notes=p["notes"])
        self.statusBar().showMessage(f"saved {res['row']['file']}", 5000)
        self.btn_cap.setEnabled(True)
        self._refresh_progress(); self._sync_indices()

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg):
        self.btn_cap.setEnabled(True)
        self.statusBar().showMessage(f"error: {msg}", 5000)

    # ---- shutdown ----------------------------------------------------
    def _shutdown(self):
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        self.sigStopLive.emit()
        self.thread.quit()
        self.thread.wait(3000)

    def closeEvent(self, e):
        self._shutdown()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = AcquireGUI(); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
