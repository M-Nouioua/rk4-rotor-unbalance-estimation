"""
Train and deploy the rotor fault-state classifier.

    python -m scripts.train_fault_classifier            # train, score, export
    python -m scripts.train_fault_classifier --no-write # print the report only
    python -m scripts.train_fault_classifier --compare  # + candidate table

Writes three artifacts:

  analysis/fault_classifier.json        metrics, confusion matrix, coefficients,
                                        and the fitted model as plain numbers
  analysis/fault_predictions.csv        out-of-fold prediction per acquisition
  ../lib/rotor/faultModel.json          the same model, where the TwinOps rotor
                                        console imports it from

The console runs the exported coefficients directly — there is no Python at
serving time — so the model that ships is byte-for-byte the model scored here.

The headline metric is nested-CV: the regulariser is tuned, so a flat CV over
the same folds that chose it would be reporting its own hyper-parameter search
back to itself. `--compare` prints the flat-CV candidate table too, which is
where the model choice came from and is where you can see what the tree
ensembles do on this campaign.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import ROOT
from estimation.faults import (
    CLASSES,
    FaultClassifier,
    add_derived,
    cross_validate,
    deployable_columns,
    full_columns,
    label,
    nested_cross_validate,
    select_C,
)

ANALYSIS = ROOT / "analysis"
CONSOLE_MODEL = ROOT.parent / "lib" / "rotor" / "faultModel.json"
CONSOLE_HOLDOUT = ROOT.parent / "lib" / "rotor" / "faultHoldout.json"

#: Decimals kept in the exported feature matrix. The console runs the model on
#: exactly these rounded numbers, and a parity check scores sklearn on the same
#: rounded matrix, so it compares the two implementations rather than comparing
#: one of them against a rounding difference.
#:
#: 4 is measured, not guessed: at 4 decimals none of the 510 deployed-model
#: predictions move off their full-precision answer, at 3 six of them do. It
#: costs 88 kB against 112 kB at 6.
EXPORT_DECIMALS = 4


def _bar(v: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * n + "." * (width - n)


def report(res: dict, title: str) -> None:
    print(f"\n=== {title}")
    print(f"    {res['n']} labelled acquisitions / {res['n_conditions']} conditions, "
          f"{res['n_features']} features, GroupKFold({res['n_splits']}) on condition_id")
    if res.get("selection") == "nested":
        print(f"    C chosen per outer fold from {res['c_grid']}: {res['chosen_C']}")
    for k in ("accuracy", "macro_f1", "balanced_accuracy"):
        print(f"    {k:<20} {res[k]:.3f}  {_bar(res[k])}")
    print(f"    {'majority baseline':<20} {res['majority_baseline']:.3f}  "
          f"{_bar(res['majority_baseline'])}   (4-class chance {res['chance']:.3f})")
    print(f"    detection  recall {res['detection_recall']:.3f}   "
          f"specificity {res['detection_specificity']:.3f}")
    print(f"    {'class':<11}{'support':>8}{'recall':>9}{'precision':>11}")
    for c in CLASSES:
        d = res["per_class"][c]
        print(f"    {c:<11}{d['support']:>8}{d['recall']:>9.3f}{d['precision']:>11.3f}")
    print(f"    confusion (rows = true, cols = predicted, order {CLASSES}):")
    for c, row in zip(CLASSES, res["confusion"]):
        print(f"      {c:<11}{''.join(f'{v:>6}' for v in row)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true", help="print only")
    ap.add_argument("--compare", action="store_true", help="candidate table")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(ANALYSIS / "features.csv")
    deploy = deployable_columns()

    if args.compare:
        full = full_columns(df[df["labeled"] == 1])
        print(f"\n=== candidate comparison (flat leak-free CV, model selection only)")
        print(f"    {'features':<10}{'model':<8}{'acc':>7}{'macroF1':>9}"
              f"{'bal.acc':>9}{'det.rec':>9}{'det.spec':>10}")
        for name, feats in (("FULL", full), ("DEPLOY", deploy)):
            for kind, kw in (("logreg", {"C": 3.0}), ("rf", {"compact": True}),
                             ("et", {"compact": True}), ("rf", {}), ("et", {})):
                r = cross_validate(df, kind, feats=feats, n_splits=args.splits, **kw)
                tag = kind + ("*" if kw.get("compact") else "")
                print(f"    {name:<10}{tag:<8}{r['accuracy']:>7.3f}{r['macro_f1']:>9.3f}"
                      f"{r['balanced_accuracy']:>9.3f}{r['detection_recall']:>9.3f}"
                      f"{r['detection_specificity']:>10.3f}")
        print("    * = depth/tree-capped so the ensemble would be exportable")

    nested = nested_cross_validate(df, feats=deploy, n_splits=args.splits)
    report(nested, "DEPLOYED MODEL — nested leak-free CV (headline)")

    full_res = nested_cross_validate(df, feats=full_columns(df[df["labeled"] == 1]),
                                     n_splits=args.splits)
    report(full_res, "same model on the FULL feature set (offline analyst, y-probe)")

    # The nested run scores the procedure; the shipped model is that same
    # procedure applied once to everything.
    chosen, curve = select_C(df, feats=deploy, n_splits=args.splits)
    clf = FaultClassifier("logreg", C=chosen, feats=deploy).fit(df)
    coefs = clf.coefficients()
    print(f"\n=== deployed fit: C={chosen} on all {nested['n']} labelled acquisitions")
    top = max(curve, key=lambda r: r["macro_f1"])
    print(f"    1-SE rule over the whole labelled set: best C={top['C']:g} at "
          f"{top['macro_f1']:.3f} +/- {top['se']:.3f}, "
          f"so anything >= {top['macro_f1'] - top['se']:.3f} is a tie")
    for r in curve:
        mark = " <- deployed" if r["C"] == chosen else ""
        print(f"      C={r['C']:<7g} macroF1 {r['macro_f1']:.3f} +/- {r['se']:.3f}{mark}")
    strongest = coefs.abs().max(axis=1).sort_values(ascending=False).head(8)
    print("    strongest features (max |coef| across classes):")
    for f, v in strongest.items():
        print(f"      {f:<24}{v:>7.3f}")

    if args.no_write:
        print("\n--no-write: nothing written")
        return

    lab = add_derived(df[df["labeled"] == 1]).reset_index(drop=True)
    preds = pd.DataFrame({
        "file": lab["file"], "condition_id": lab["condition_id"],
        "block": lab["block"], "config": lab["config"], "speed_id": lab["speed_id"],
        "U1_mag": lab["U1_mag"], "U2_mag": lab["U2_mag"],
        "true_class": label(lab), "pred_class": nested["pred"],
        "fold": nested["fold"],
    })
    preds.to_csv(ANALYSIS / "fault_predictions.csv", index=False)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "analysis/features.csv",
        "classes": CLASSES,
        "headline": {k: v for k, v in nested.items()
                     if k not in ("fold", "pred", "y_true", "features")},
        "full_feature_set": {k: v for k, v in full_res.items()
                             if k not in ("fold", "pred", "y_true", "features")},
        "selection": {"rule": "one-standard-error", "chosen_C": chosen, "curve": curve},
        "model": clf.export(),
        "coefficients": {c: coefs[c].to_dict() for c in coefs.columns},
    }
    (ANALYSIS / "fault_classifier.json").write_text(json.dumps(payload, indent=1))
    CONSOLE_MODEL.write_text(json.dumps(
        {
            "generated": payload["generated"],
            "trainedOn": {
                "source": "ROTOR/analysis/features.csv",
                "acquisitions": nested["n"],
                "conditions": nested["n_conditions"],
                "loadThreshGmm": nested["load_thresh_gmm"],
            },
            "metrics": {
                "accuracy": nested["accuracy"],
                "macroF1": nested["macro_f1"],
                "balancedAccuracy": nested["balanced_accuracy"],
                "majorityBaseline": nested["majority_baseline"],
                "chance": nested["chance"],
                "detectionRecall": nested["detection_recall"],
                "detectionSpecificity": nested["detection_specificity"],
                "perClass": nested["per_class"],
                "confusion": nested["confusion"],
                "validation": (f"nested GroupKFold({nested['n_splits']}"
                               f"x{nested['inner_splits']}) on condition_id"),
            },
            **clf.export(),
        }, indent=1))
    # Real acquisitions the console replays: the feature vector it feeds the
    # model, the campaign's true class, and the *out-of-fold* prediction. The
    # deployed model has seen every one of these rows, so its own answer on them
    # is in-sample and is not what the console reports — the out-of-fold column
    # is the honest one and is what gets displayed.
    Xr = np.round(lab[deploy].to_numpy(float), EXPORT_DECIMALS)
    CONSOLE_HOLDOUT.write_text(json.dumps({
        "generated": payload["generated"],
        "source": "ROTOR/analysis/features.csv",
        "note": ("Real RK-4 acquisitions. `oof` is the leak-free out-of-fold "
                 "prediction (GroupKFold on condition_id); the deployed model was "
                 "fitted on all of these rows, so its in-sample answer on them is "
                 "not a performance claim."),
        "features": deploy,
        "classes": CLASSES,
        "decimals": EXPORT_DECIMALS,
        "n": int(len(lab)),
        # Column-major-free flat layout: row i occupies x[i*nf : (i+1)*nf].
        # Repeating 24 JSON keys 510 times cost 64 kB and bought nothing.
        "x": [float(v) for v in Xr.ravel()],
        "truth": [CLASSES.index(c) for c in label(lab)],
        "oof": [CLASSES.index(c) for c in nested["pred"]],
        "fold": [int(f) for f in nested["fold"]],
        "condition": [str(c) for c in lab["condition_id"]],
        "speed": [str(s) for s in lab["speed_id"]],
        "u1": [round(float(v), 2) for v in lab["U1_mag"]],
        "u2": [round(float(v), 2) for v in lab["U2_mag"]],
    }, separators=(",", ":")))
    print(f"\nwrote {ANALYSIS/'fault_classifier.json'}")
    print(f"wrote {ANALYSIS/'fault_predictions.csv'}")
    print(f"wrote {CONSOLE_MODEL}")
    print(f"wrote {CONSOLE_HOLDOUT}  ({CONSOLE_HOLDOUT.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
