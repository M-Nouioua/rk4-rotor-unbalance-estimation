"""
Check that the headline numbers in the manuscript still match the generated
artefacts.

The manuscript quotes values in prose and in hand-written tables, so a change to
the analysis code can leave the text stating a number the pipeline no longer
produces. This script asserts the specific figures the argument rests on, and
fails loudly rather than reporting a clean run.

    python -m scripts.audit_numbers

Exit codes: 0 all checks pass, 1 at least one mismatch.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
PAPER = ROOT / "paper"


def load(name):
    p = ANALYSIS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tex() -> str:
    return "\n".join(f.read_text(encoding="utf-8")
                     for f in sorted(PAPER.glob("*.tex"))
                     if not f.name.endswith(".unused"))


def main() -> int:
    body = tex()
    bench = load("benchmark.json")
    gate = load("pinn_gate.json")
    val = load("pinn_validate.json")
    rep = load("representation_test.json")
    ang = load("angle_novelty_control.json")
    modes = load("pinn_modes.json")
    repeat = load("repeatability.json")

    checks = []

    def chk(label, value, fmt="{:.3f}"):
        """Assert that `value`, formatted, appears somewhere in the manuscript."""
        s = fmt.format(value) if not isinstance(value, str) else value
        checks.append((label, s, s in body))

    # --- condition-level baselines -------------------------------------------
    # Localisation and the threshold-free areas are checked as well as the
    # predictive scores. An earlier version of this script asserted only the
    # r2_pred values, so a change to the load-threshold comparison moved every
    # localisation and ROC figure in Table 2 while this audit still reported a
    # clean run.
    if bench:
        for m, name in (("physics", "ICM"), ("ml", "trees"), ("hybrid", "physics-augmented")):
            c = bench["methods"][m]["condition"]
            o = c["overall"]
            chk(f"{name} r2_pred", o["r2_pred"])
            chk(f"{name} localisation %", 100.0 * c["localization_acc"], "{:.1f}")
            det = c.get("detection") or {}
            if det.get("roc_auc") is not None:
                chk(f"{name} ROC-AUC", det["roc_auc"])
            if det.get("trivial_acc") is not None:
                chk("all-loaded baseline %", 100.0 * det["trivial_acc"], "{:.1f}")
        fm = bench.get("factor_matrix", {}).get("methods", {})
        h = (fm.get("hybrid") or {})
        for p in ("condition", "angle_sector", "magnitude_extrapolating"):
            if h.get(p):
                chk(f"hybrid {p} r2_pred", h[p]["r2_pred"])

    # --- operator, the rows the manuscript quotes ----------------------------
    if gate:
        iso = (gate["variants"].get("iso") or {})
        for p in ("condition", "angle_sector", "magnitude_extrapolating"):
            if iso.get(p):
                chk(f"operator iso {p} r2_pred", iso[p]["r2_pred"])
        an = (gate["variants"].get("aniso") or {})
        if an.get("angle_sector"):
            chk("operator aniso angle_sector r2_pred", an["angle_sector"]["r2_pred"])

    # --- controls and mechanism ---------------------------------------------
    if ang:
        v = ang["verdict"]
        chk("angle-novelty delta r2_pred", v["delta_r2_pred_from_angle_novelty"])
    if rep:
        a = rep["protocols"]["angle_sector"]
        chk("cartesian angle r2_pred", a["cartesian"]["r2_pred"])
        chk("equivariant angle r2_pred", a["equivariant"]["r2_pred"])

    # --- measurement repeatability, the benchmark for the forward residual ---
    if repeat:
        a, r = repeat["absolute"], repeat["relative"]
        chk("repeatability abs median um", a["median"])
        chk("repeatability abs p95 um", a["p95"])
        chk("repeatability rel median", r["median"])
        chk("repeatability n records", str(r["n_records"]))

    # --- forward and modes ---------------------------------------------------
    if val and "forward" in val:
        chk("forward response r2_pred", val["forward"]["response_r2_pred"])
    if modes:
        m3 = modes["modes"].get("3")
        if m3:
            chk("3-mode inverse r2_pred", m3["inverse"]["r2_pred"])
            chk("3-mode forward r2_pred", m3["forward"]["response_r2_pred"])

    width = max(len(c[0]) for c in checks) if checks else 10
    bad = 0
    for label, s, ok in checks:
        print(f"  {'ok ' if ok else 'MISSING'}  {label:<{width}}  {s}")
        if not ok:
            bad += 1
    print(f"\n{len(checks) - bad} of {len(checks)} headline numbers found in the manuscript")
    if bad:
        print("A MISSING row means the artefact value does not appear in the text.")
        print("Either the text is stale, or that value is simply not quoted; check each.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
