"""
Continuous, hardware-timed acquisition from two NI-9234 modules.

Proximity probes are AC-coupled (IEPE OFF) so the dynamic displacement fits the
9234 +/-5 V range after the negative DC gap bias is removed. The keyphasor is
read on its own channel as a raw voltage; pulse detection happens downstream.

Requires the NI-DAQmx runtime + the `nidaqmx` Python package. All hardware calls
are isolated here so the rest of the pipeline can run on recorded data offline.
"""
from __future__ import annotations

import numpy as np

try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        Coupling,
        TerminalConfiguration,
    )
    _HAS_NIDAQMX = True
except ImportError:  # allows offline development without NI drivers
    _HAS_NIDAQMX = False


# Order matters: this is the row order of the returned data array.
CHANNEL_ORDER = ["keyphasor", "plane1_x", "plane1_y", "plane2_x", "plane2_y"]


class RK4Acquisition:
    """Thin wrapper around a DAQmx analog-input task for the RK4 rig."""

    def __init__(self, cfg: dict):
        if not _HAS_NIDAQMX:
            raise RuntimeError(
                "nidaqmx not available. Install NI-DAQmx runtime + `pip install nidaqmx`."
            )
        self.cfg = cfg["daq"]
        self.fs = int(self.cfg["sample_rate_hz"])
        self.block = int(self.cfg["samples_per_read"])
        self.task: "nidaqmx.Task | None" = None

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "RK4Acquisition":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        self.task = nidaqmx.Task()
        vr = float(self.cfg["voltage_range_v"])
        coupling = Coupling.AC if self.cfg["coupling"].upper() == "AC" else Coupling.DC

        for name in CHANNEL_ORDER:
            ch = self.cfg["channels"][name]
            aic = self.task.ai_channels.add_ai_voltage_chan(
                ch["device"],
                min_val=-vr,
                max_val=vr,
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,  # 9234
            )
            # AC coupling removes the proximity-probe DC gap bias. A plain voltage
            # channel on the 9234 has IEPE excitation OFF by default (correct for
            # the Proximitor buffered outputs).
            aic.ai_coupling = coupling

        self.task.timing.cfg_samp_clk_timing(
            rate=self.fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.block * 4,
        )

    def read_block(self) -> np.ndarray:
        """Read one block. Returns array shaped (n_channels, samples)."""
        assert self.task is not None, "call open() first"
        data = self.task.read(number_of_samples_per_channel=self.block)
        return np.asarray(data, dtype=float)

    def stream(self, n_blocks: int | None = None):
        """Yield consecutive blocks. n_blocks=None -> run until stopped."""
        assert self.task is not None, "call open() first"
        self.task.start()
        i = 0
        try:
            while n_blocks is None or i < n_blocks:
                yield self.read_block()
                i += 1
        finally:
            self.task.stop()

    def close(self) -> None:
        if self.task is not None:
            self.task.close()
            self.task = None


def acquire_record(cfg: dict, duration_s: float) -> dict:
    """Acquire a fixed-duration record and return a labeled channel dict."""
    acq = RK4Acquisition(cfg)
    fs = acq.fs
    n_blocks = int(np.ceil(duration_s * fs / acq.block))
    chunks: list[np.ndarray] = []
    with acq:
        for blk in acq.stream(n_blocks=n_blocks):
            chunks.append(blk)
    data = np.concatenate(chunks, axis=1)
    return {name: data[i] for i, name in enumerate(CHANNEL_ORDER)} | {"fs": fs}
