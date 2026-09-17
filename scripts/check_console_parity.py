"""
Verify the browser runs the same model sklearn scored.

    python -m scripts.check_console_parity

The console does not call Python. It multiplies out the exported weights in
TypeScript (`lib/rotor/faults.ts`), which means there are two implementations of
this model and they can drift: a sign convention, a feature order, a modulo
that behaves differently in the two languages. This script compares them on the
exact rounded feature matrix the console holds and fails on any disagreement.

It needs the TypeScript side's output, produced by bundling and running the
console's own classifier:

    npx esbuild <entry>.ts --bundle --format=esm --platform=node \\
      --outfile=out.mjs --alias:@=<repo root> && node out.mjs > ts_out.json

Pass that file with --ts. Without it the script prints the sklearn side and the
command to generate the other half.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from config import ROOT
from estimation.faults import (
    FaultClassifier,
    add_derived,
    deployable_columns,
    select_C,
)
from scripts.train_fault_classifier import EXPORT_DECIMALS

TOL_PROB = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", help="JSON emitted by the bundled TypeScript classifier")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "analysis" / "features.csv")
    deploy = deployable_columns()
    lab = add_derived(df[df["labeled"] == 1]).reset_index(drop=True)
    # The console stores the matrix rounded, so score sklearn on the rounded
    # matrix too — otherwise this measures rounding, not agreement.
    X = np.round(lab[deploy].to_numpy(float), EXPORT_DECIMALS)

    chosen, _ = select_C(df, feats=deploy)
    pipe = FaultClassifier("logreg", C=chosen, feats=deploy).fit(df).pipe
    py_pred = pipe.predict(X)
    py_prob = pipe.predict_proba(X)

    print(f"sklearn: {len(py_pred)} rows, C={chosen:g}, "
          f"classes={list(pipe.named_steps['model'].classes_)}")

    if not args.ts:
        print("\nno --ts given; nothing compared. Generate the other half with the\n"
              "esbuild command in this module's docstring, then re-run with --ts.")
        return 0

    ts = json.loads(open(args.ts).read())
    ts_pred = np.array(ts["pred"])
    ts_prob = np.array(ts["probs"])

    fails = []
    if list(ts["classes"]) != [str(c) for c in pipe.named_steps["model"].classes_]:
        fails.append(f"class order differs: {ts['classes']}")
    if len(ts_pred) != len(py_pred):
        fails.append(f"row count differs: {len(ts_pred)} vs {len(py_pred)}")
    else:
        disagree = np.flatnonzero(ts_pred != py_pred)
        if disagree.size:
            fails.append(f"{disagree.size} label disagreements, first at row "
                         f"{disagree[0]}: py={py_pred[disagree[0]]} ts={ts_pred[disagree[0]]}")
        worst = float(np.abs(py_prob - ts_prob).max())
        print(f"max |probability difference| = {worst:.3e}  (tolerance {TOL_PROB:.0e})")
        if worst > TOL_PROB:
            fails.append(f"probabilities differ by {worst:.3e}")
        print(f"label agreement = {(ts_pred == py_pred).sum()} / {len(py_pred)}")

    if fails:
        print("\nPARITY FAILED")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPARITY OK — the console evaluates the model sklearn scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
