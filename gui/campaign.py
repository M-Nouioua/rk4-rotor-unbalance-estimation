"""
Campaign runner GUI — the full test matrix as a click-through checklist.

Every condition from datasets/conditions.csv is a row showing WHAT TO MOUNT
(mass + hole angle + disk). Pick a speed, press the row's Run button; it captures
the acquisitions, logs them to datasets/index.csv, shows live rpm/1X-2X/orbit with
the safety trip, and turns the row GREEN when that condition is complete at that speed.

ANGLE CONVENTION (mount this way):
  Align BOTH disks' engraved "0" hole to the keyphasor notch (same direction).
  Angle increases with rotation; holes at 0/45/90/135/180/225/270/315 deg.
  In-phase = same hole on both disks; Anti-phase = disk-2 mass 180 deg opposite.

Run:  python -m gui.campaign      (add --sim for no hardware)
Needs: pip install pyqt5 pyqtgraph
"""
from __future__ import annotations

import sys

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg
except ImportError:
    sys.exit("Campaign GUI needs PyQt5 + pyqtgraph:  pip install pyqt5 pyqtgraph")

from gui.session import Session

GREEN, AMBER, GREY, RED = "#1f9d57", "#e69500", "#8a94a2", "#e5343a"


class CampaignWorker(QtCore.QObject):
    acqDone = QtCore.pyqtSignal(dict)     # per acquisition
    jobDone = QtCore.pyqtSignal(dict)     # per Run click
    error = QtCore.pyqtSignal(str)

    def __init__(self, session):
        super().__init__()
        self.s = session

    @QtCore.pyqtSlot(dict)
    def run_job(self, job):
        cond, speed = job["condition"], job["speed"]
        rpm, dur, sim = job["rpm"], job["duration"], job["simulate"]
        captured, tripped, speed_fail = 0, False, False
        for _ in range(job["n"]):
            m, a = self.s.next_indices(cond, speed)
            try:
                rec = self.s.acquire(dur, simulate=sim, rpm=rpm or 1000)
                feats = self.s.features(rec)
            except Exception as e:
                self.error.emit(str(e)); break
            # guard: refuse to log if the measured speed doesn't match the selected one
            if rpm and not sim and abs(feats["rpm"] - rpm) / rpm > 0.07:
                speed_fail = True
                self.error.emit(
                    f"{cond} @ {speed}: measured {feats['rpm']:.0f} rpm != {rpm:.0f} rpm — "
                    f"set the motor to {rpm:.0f} rpm, then Run again (not saved)")
                break
            if self.s.orbit_trip(feats):
                tripped = True
                self.acqDone.emit({"feats": feats, "trip": True})
                break
            self.s.save(rec, cond, speed, rpm, m, a, feats["rpm"], notes=job.get("notes", ""))
            captured += 1
            self.acqDone.emit({"feats": feats, "trip": False, "mount": m, "acq": a})
        self.jobDone.emit({"condition": cond, "speed": speed, "captured": captured,
                           "tripped": tripped, "speed_fail": speed_fail})


class CampaignGUI(QtWidgets.QMainWindow):
    sigRunJob = QtCore.pyqtSignal(dict)

    COLS = ["ID", "Block", "Mount (mass @ hole, per disk)", "Done", "", "Run"]

    def __init__(self, simulate=False):
        super().__init__()
        self.s = Session()
        self.simulate = simulate
        self.setWindowTitle("RK4 — Experiment Campaign Runner")
        self.resize(1180, 760)
        self.buttons = {}          # condition_id -> QPushButton
        self.row_of = {}           # condition_id -> table row
        self.failed = set()        # (condition_id, speed) that failed the speed check
        self._build()
        self._start_worker()
        self._populate()

    # ---- layout ------------------------------------------------------
    def _build(self):
        root = QtWidgets.QWidget(); self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)

        help_ = QtWidgets.QLabel(
            "Mount convention:  align BOTH disks' “0” hole to the keyphasor notch "
            "(same direction).  Holes at 0/45/90/135/180/225/270/315°, angle increases "
            "with rotation.  In-phase = same hole both disks;  Anti-phase = 180° apart.  "
            "One screw per condition; empty all other holes.")
        help_.setWordWrap(True); help_.setStyleSheet("background:#f4f4f4;padding:7px;border-radius:6px;")
        v.addWidget(help_)

        # control bar
        bar = QtWidgets.QHBoxLayout()
        self.cbo_speed = QtWidgets.QComboBox(); self.cbo_speed.addItems(self.s.speed_ids() or ["N1"])
        self.cbo_speed.currentTextChanged.connect(self._recolor)
        self.lbl_rpm_cmd = QtWidgets.QLabel()
        self.cbo_speed.currentTextChanged.connect(self._show_cmd_rpm)
        self.spn_target = QtWidgets.QSpinBox(); self.spn_target.setRange(1, 30); self.spn_target.setValue(3)
        self.spn_target.setToolTip("acquisitions to log per condition per speed (green when reached)")
        self.spn_target.valueChanged.connect(self._recolor)
        self.spn_dur = QtWidgets.QDoubleSpinBox(); self.spn_dur.setRange(1, 60); self.spn_dur.setValue(8)
        self.cbo_block = QtWidgets.QComboBox()
        self.cbo_block.addItems(["all", "baseline", "ICM", "A_severity", "B_phase",
                                 "C_fine_phase", "LoD", "T_twoplane", "U_anchor", "blind"])
        self.cbo_block.currentTextChanged.connect(self._populate)
        self.cbo_sort = QtWidgets.QComboBox(); self.cbo_sort.addItems(["by mass", "by block"])
        self.cbo_sort.setToolTip("'by mass' groups all conditions sharing a mass+angle, "
                                 "so you finish one screw placement before changing it")
        self.cbo_sort.currentTextChanged.connect(self._populate)
        self.chk_sim = QtWidgets.QCheckBox("Simulate"); self.chk_sim.setChecked(self.simulate)
        self.btn_redo = QtWidgets.QPushButton("Redo selected")
        self.btn_redo.setToolTip("Delete the selected row's logged data (both speeds) "
                                 "and its .npz files, so the condition can be re-run")
        self.btn_redo.clicked.connect(self._redo_selected)
        for w in (QtWidgets.QLabel("Speed"), self.cbo_speed, self.lbl_rpm_cmd,
                  QtWidgets.QLabel("  Acqs/cond"), self.spn_target,
                  QtWidgets.QLabel("  Dur s"), self.spn_dur,
                  QtWidgets.QLabel("  Block"), self.cbo_block,
                  QtWidgets.QLabel("  Sort"), self.cbo_sort, self.chk_sim, self.btn_redo):
            bar.addWidget(w)
        bar.addStretch(1)
        v.addLayout(bar)

        # live panel
        live = QtWidgets.QHBoxLayout()
        self.lbl_rpm = QtWidgets.QLabel("rpm: —"); self.lbl_rpm.setStyleSheet("font-size:18px;")
        self.lbl_orders = QtWidgets.QLabel("1X / 2X: —")
        self.lbl_orders.setStyleSheet("font-family:monospace;")
        self.lbl_safe = QtWidgets.QLabel("orbit —"); self.lbl_safe.setStyleSheet(f"color:{GREEN};font-weight:600;")
        self.p1 = self._orbit("Plane 1"); self.p2 = self._orbit("Plane 2")
        self.c1 = self.p1.plot(pen=pg.mkPen("#2980b9", width=2))
        self.c2 = self.p2.plot(pen=pg.mkPen("#8e44ad", width=2))
        col = QtWidgets.QVBoxLayout(); col.addWidget(self.lbl_rpm); col.addWidget(self.lbl_orders); col.addWidget(self.lbl_safe); col.addStretch(1)
        live.addLayout(col); live.addWidget(self.p1); live.addWidget(self.p2)
        v.addLayout(live)

        # progress + table
        self.bar = QtWidgets.QProgressBar(); v.addWidget(self.bar)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.table, 1)
        self.status = QtWidgets.QLabel(); v.addWidget(self.status)
        self._show_cmd_rpm()

    def _orbit(self, title):
        w = pg.PlotWidget(title=title); w.setAspectLocked(True); w.showGrid(x=True, y=True)
        w.setMaximumHeight(180); w.setMaximumWidth(220)
        return w

    # ---- worker ------------------------------------------------------
    def _start_worker(self):
        self.thread = QtCore.QThread(self)
        self.worker = CampaignWorker(self.s)
        self.worker.moveToThread(self.thread)
        self.sigRunJob.connect(self.worker.run_job)
        self.worker.acqDone.connect(self._on_acq)
        self.worker.jobDone.connect(self._on_job_done)
        self.worker.error.connect(self._on_error)
        self.thread.start()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)

    # ---- table -------------------------------------------------------
    def _show_cmd_rpm(self):
        r = self.s.speed_rpm(self.cbo_speed.currentText())
        self.lbl_rpm_cmd.setText(f"= {r:.0f} rpm" if r else "(set rpm in speeds.csv)")

    def _sort_key(self, c):
        """Group by (mass, angle, config) so one screw placement is finished before it moves."""
        d1 = float(c["disk1_mass_g"] or 0); d2 = float(c["disk2_mass_g"] or 0)
        mass = max(d1, d2)
        if mass == 0 and c["config"] != "baseline":
            mass = 1e9                                   # blind / ICM (unknown) -> last
        ang = float(c["disk1_angle_deg"] or 0) if d1 > 0 else float(c["disk2_angle_deg"] or 0)
        rank = {"baseline": -1, "D1": 0, "D2": 1, "inphase": 2, "antiphase": 3}.get(c["config"], 8)
        return (mass, ang, rank, c["condition_id"])

    def _conditions(self):
        f = self.cbo_block.currentText()
        conds = [c for c in self.s.conditions if f == "all" or c["block"] == f]
        if self.cbo_sort.currentText() == "by mass":
            conds = sorted(conds, key=self._sort_key)
        return conds

    def _populate(self):
        conds = self._conditions()
        self.table.setRowCount(len(conds)); self.buttons.clear(); self.row_of.clear()
        for r, c in enumerate(conds):
            cid = c["condition_id"]; self.row_of[cid] = r
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(cid))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(c["block"]))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(self.s.label_text(cid)))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(""))
            btn = QtWidgets.QPushButton("Run")
            btn.clicked.connect(lambda _=False, x=cid: self.run_condition(x))
            self.buttons[cid] = btn
            self.table.setCellWidget(r, 5, btn)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self._recolor()

    def _recolor(self):
        speed = self.cbo_speed.currentText()
        default = self.spn_target.value()
        rows = self.s._log_rows()
        done_map = {}
        for row in rows:
            k = (row["condition_id"], row["speed_id"])
            done_map[k] = done_map.get(k, 0) + 1
        total_done = total_target = n_applicable = complete = 0
        for cid, r in self.row_of.items():
            done = done_map.get((cid, speed), 0)
            if not self.s.applies_at(cid, speed):
                self.table.item(r, 3).setText("—")
                it = self.table.item(r, 4); it.setText(f"n/a at {speed}")
                it.setBackground(QtGui.QColor(GREY)); it.setForeground(QtGui.QColor("#fff"))
                self.buttons[cid].setEnabled(False)
                continue
            target = self.s.target_acqs(cid, default)
            n_applicable += 1; total_target += target; total_done += min(done, target)
            self.table.item(r, 3).setText(f"{done}/{target}")
            if (cid, speed) in self.failed and done < target:
                col, txt = RED, "FAILED — wrong rpm"
            elif done == 0:
                col, txt = GREY, "pending"
            elif done < target:
                col, txt = AMBER, (f"partial · mount {done//self.s.n_acq + 1}"
                                   if target > self.s.n_acq else "partial")
            else:
                col, txt = GREEN, "DONE"; complete += 1
            it = self.table.item(r, 4); it.setText(txt)
            it.setBackground(QtGui.QColor(col)); it.setForeground(QtGui.QColor("#fff"))
            self.buttons[cid].setEnabled(done < target)
        self.bar.setMaximum(max(1, total_target)); self.bar.setValue(total_done)
        self.status.setText(f"{complete} / {n_applicable} conditions complete at {speed}")

    # ---- run ---------------------------------------------------------
    def run_condition(self, cid):
        speed = self.cbo_speed.currentText()
        if not self.s.applies_at(cid, speed):
            self.status.setText(f"{cid} is not scheduled at {speed} (run_at_speeds)")
            return
        target = self.s.target_acqs(cid, self.spn_target.value())
        done = sum(1 for row in self.s._log_rows()
                   if row["condition_id"] == cid and row["speed_id"] == speed)
        # cap each Run to ONE physical mount (<= n_acq) so multi-mount anchors get
        # correct mount labels — the operator re-mounts between clicks.
        rem_in_mount = self.s.n_acq - (done % self.s.n_acq)
        n = max(1, min(target - done, rem_in_mount))
        for b in self.buttons.values():
            b.setEnabled(False)
        self.status.setText(f"running {cid} @ {speed}: capturing {n} acquisition(s)…")
        self.sigRunJob.emit({
            "condition": cid, "speed": speed, "rpm": self.s.speed_rpm(speed),
            "duration": self.spn_dur.value(), "simulate": self.chk_sim.isChecked(),
            "n": n, "notes": self.s.label_text(cid),
        })

    def _redo_selected(self):
        r = self.table.currentRow()
        if r < 0:
            self.status.setText("select a row first, then press Redo"); return
        cid = self.table.item(r, 0).text()
        if QtWidgets.QMessageBox.question(
                self, "Redo condition",
                f"Delete ALL logged data for {cid} (both speeds, incl. .npz files) "
                f"and mark it pending?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        n = self.s.delete_condition(cid)
        for s in self.s.speed_ids():
            self.failed.discard((cid, s))
        self._recolor()
        self.status.setText(f"cleared {cid}: removed {n} run(s) — now pending, ready to re-run")

    @QtCore.pyqtSlot(dict)
    def _on_acq(self, d):
        f = d["feats"]
        self.lbl_rpm.setText(f"rpm: {f['rpm']:.0f}")
        trip = d["trip"]
        self.lbl_safe.setText(f"orbit {f['max_orbit_mils']:.2f} mils" + ("  ⚠ TRIP" if trip else ""))
        self.lbl_safe.setStyleSheet(f"color:{RED if trip else GREEN};font-weight:700;")
        d1 = f["planes"]["plane1"]["orders"]; d2 = f["planes"]["plane2"]["orders"]
        self.lbl_orders.setText(f"P1 1X {d1[1][0]:.3f}∠{d1[1][1]:+6.1f}  2X {d1[2][0]:.3f}\n"
                                f"P2 1X {d2[1][0]:.3f}∠{d2[1][1]:+6.1f}  2X {d2[2][0]:.3f}")
        self.c1.setData(f["planes"]["plane1"]["orbit_x"], f["planes"]["plane1"]["orbit_y"])
        self.c2.setData(f["planes"]["plane2"]["orbit_x"], f["planes"]["plane2"]["orbit_y"])

    @QtCore.pyqtSlot(dict)
    def _on_job_done(self, d):
        key = (d["condition"], d["speed"])
        if d.get("speed_fail"):
            self.failed.add(key)
        if d["captured"]:
            self.failed.discard(key)            # a good capture clears the failed flag
        if d["tripped"]:
            QtWidgets.QMessageBox.warning(self, "Safety trip",
                f"{d['condition']} @ {d['speed']}: orbit exceeded the trip limit — "
                f"captured {d['captured']} before stopping. Reduce mass/speed.")
        self._recolor()
        if d["captured"]:
            cid, speed = d["condition"], d["speed"]
            self.status.setText(f"{cid} @ {speed}: +{d['captured']} logged.")
            # multi-mount (anchor) prompt: physically re-mount before the next repeat
            target = self.s.target_acqs(cid, self.spn_target.value())
            done = sum(1 for row in self.s._log_rows()
                       if row["condition_id"] == cid and row["speed_id"] == speed)
            if target > self.s.n_acq and done < target:
                QtWidgets.QMessageBox.information(
                    self, "Re-mount for the next repeat",
                    f"{cid}: mount {done // self.s.n_acq} done ({done}/{target}).\n\n"
                    f"UNSCREW and RE-MOUNT the mass (re-weigh, re-seat, re-torque), then "
                    f"press Run again for mount {done // self.s.n_acq + 1}. "
                    f"This is what makes the setup-variance (σ_mount) estimate valid.")

    @QtCore.pyqtSlot(str)
    def _on_error(self, m):
        for b in self.buttons.values():
            b.setEnabled(True)
        self.status.setText(f"error: {m}")
        self._recolor()

    # ---- shutdown ----------------------------------------------------
    def _shutdown(self):
        if getattr(self, "_down", False):
            return
        self._down = True
        self.thread.quit(); self.thread.wait(4000)

    def closeEvent(self, e):
        self._shutdown(); super().closeEvent(e)


def main(simulate=False):
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = CampaignGUI(simulate=simulate or ("--sim" in sys.argv)); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
