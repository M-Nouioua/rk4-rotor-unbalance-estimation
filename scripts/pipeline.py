"""
One-shot analysis pipeline for the digital twin.

Runs the full offline chain end-to-end so the whole result set regenerates from the
raw .npz acquisitions with a single command:

    features -> benchmark -> controlled experiments -> physics-informed operator
             -> figure data -> diagnostic figures -> dashboard -> provenance

    python -m scripts.pipeline                 # everything
    python -m scripts.pipeline --no-features   # reuse the cached feature table
    python -m scripts.pipeline --fast          # skip the slow operator stages

Manuscript figures are NOT produced here. They are rendered in MATLAB from the CSVs
that `scripts.export_fig_data` writes:

    matlab -batch "addpath('matlab'); make_all_figures"

`scripts.figures` still writes matplotlib PNGs, but those are working diagnostics
and inputs to the interactive dashboard, not manuscript figures.

Each stage is runnable on its own; see the module docstrings for what each one
establishes and which claims depend on it.
"""
from __future__ import annotations

import argparse
import sys
import time

from config import ROOT

# Windows consoles default to cp1252; keep sub-stage unicode from crashing the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def _run(label, fn):
    t = time.time()
    print(f"\n{'='*66}\n>> {label}\n{'='*66}")
    fn()
    print(f"  [OK] {label} ({time.time()-t:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-features", action="store_true",
                    help="reuse analysis/features.csv instead of re-extracting")
    ap.add_argument("--fast", action="store_true",
                    help="skip the operator stages (gate, validation, mode study)")
    args = ap.parse_args()

    if not args.no_features:
        from scripts import build_features
        _run("Extract features from all acquisitions", build_features.main)
    elif not (ROOT / "analysis" / "features.csv").exists():
        from scripts import build_features
        _run("Extract features (cache missing)", build_features.main)

    from scripts import benchmark
    _run("Benchmark: ICM vs ML vs physics-augmented, factor-aware", benchmark.main)

    from scripts import representation_test, angle_novelty_control
    _run("Controlled experiment: representation vs capacity", representation_test.main)
    _run("Control: angle novelty vs training-set size", angle_novelty_control.main)

    if not args.fast:
        from scripts import pinn_gate, pinn_validate, pinn_modes
        _run("Physics-informed operator: factor-aware gate", pinn_gate.main)
        _run("Operator: nested selection, blind scoring, forward validation",
             pinn_validate.main)
        _run("Operator: mode-count study", pinn_modes.main)

    from scripts import export_fig_data, figures, dashboard, freeze_blind
    _run("Export plot-ready data for the MATLAB figures", export_fig_data.main)
    _run("Render diagnostic figures (PNG, not manuscript)", figures.main)
    _run("Build interactive dashboard", dashboard.main)
    _run("Freeze blind-evidence provenance", freeze_blind.main)

    print(f"\n{'='*66}")
    print("DONE.")
    print("  results      analysis/RESULTS.md  (generated; cite this, never prose)")
    print("  manuscript   matlab -batch \"addpath('matlab'); make_all_figures\"")
    print("               -> analysis/figures/*.pdf (vector, Times New Roman)")
    print("  provenance   python -m scripts.freeze_blind --verify")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
