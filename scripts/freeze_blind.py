"""
Freeze the blind-validation evidence into an immutable, auditable record.

WHY THIS EXISTS. The blind set is the only part of the campaign that can support
an unbiased accuracy claim, and it can only do that ONCE. Its value rests on three
facts that must be provable after the fact, not asserted in prose:

  1. the predictions were made before the labels were revealed;
  2. the code that made them is identifiable;
  3. the raw recordings have not changed since.

This script records all three. Run it NOW (the BLND01-10 labels are already
revealed, so the record is retrospective and is marked as such), and run it again
immediately after predicting any future blind block -- before unsealing.

    python -m scripts.freeze_blind                  # write/refresh the manifest
    python -m scripts.freeze_blind --verify         # re-check hashes, change nothing

Output: analysis/blind_freeze.json

A retrospective freeze is weaker evidence than a prospective one and the manifest
says so explicitly. For the confirmatory claim in the manuscript, the honest
statement is: BLND01-10 were predicted and then unsealed during the campaign; this
manifest fixes the artifacts as they stand and is the baseline against which any
re-analysis must be compared. Anything tuned after unsealing (for example the
direct-|U| target ablation) must be reported as exploratory and validated on a
NEW untouched block -- never on BLND01-10 again.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from config import ROOT

ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "blind_freeze.json"

# artifacts whose content defines the blind claim
CODE = ["estimation/icm.py", "estimation/ml.py", "estimation/twin.py",
        "estimation/stats.py", "processing/order_tracking.py",
        "processing/features.py", "scripts/build_features.py",
        "scripts/benchmark.py"]
ARTIFACTS = ["analysis/features.csv", "analysis/benchmark.json",
             "analysis/blind_predictions.csv", "datasets/conditions.csv",
             "datasets/index.csv", "datasets/runout_slowroll.npz"]


def sha256(path) -> str | None:
    p = ROOT / path
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "pandas", "sklearn"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def blind_inventory() -> dict:
    """Per-blind-condition record counts, file hashes and reveal status."""
    conds = pd.read_csv(ROOT / "datasets" / "conditions.csv")
    blind = conds[conds.block == "blind"].copy()
    idx_path = ROOT / "datasets" / "index.csv"
    idx = pd.read_csv(idx_path) if idx_path.exists() else pd.DataFrame()
    inv = {}
    for _, c in blind.iterrows():
        cid = c.condition_id
        rows = idx[idx.condition_id == cid] if len(idx) else pd.DataFrame()
        sealed = str(c.get("config", "")) == "sealed"
        files = sorted(rows.file.tolist()) if len(rows) else []
        inv[cid] = {
            "acquired_records": int(len(rows)),
            "files": files,
            "file_sha256": {f: sha256(f"datasets/{f}") for f in files},
            "revealed": not sealed,
            "truth": None if sealed else {
                "config": c.get("config"),
                "disk1_gmm": float(c.get("U1_gmm") or 0),
                "disk1_angle_deg": (None if pd.isna(c.get("disk1_angle_deg"))
                                    else float(c.get("disk1_angle_deg"))),
                "disk2_gmm": float(c.get("U2_gmm") or 0),
                "disk2_angle_deg": (None if pd.isna(c.get("disk2_angle_deg"))
                                    else float(c.get("disk2_angle_deg"))),
            },
            "first_timestamp": (None if not len(rows)
                                else str(rows.timestamp.min())),
            "last_timestamp": (None if not len(rows)
                               else str(rows.timestamp.max())),
        }
    return inv


def build() -> dict:
    inv = blind_inventory()
    revealed = [k for k, v in inv.items() if v["revealed"] and v["acquired_records"]]
    sealed = [k for k, v in inv.items() if not v["revealed"]]
    acquired_no_pred = [k for k, v in inv.items() if not v["acquired_records"]]
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": "RETROSPECTIVE -- BLND01-10 were unsealed during the campaign "
                      "(2026-07-26) before this manifest existed. This record fixes "
                      "the artifacts as they stand; it does not by itself prove "
                      "prediction preceded revelation. Treat BLND01-10 as SPENT: "
                      "usable as reported evidence, not re-usable to confirm any "
                      "model chosen after 2026-07-26.",
        "canonical_root": str(ROOT),
        "git_rev": git_rev(),
        "versions": pkg_versions(),
        "code_sha256": {p: sha256(p) for p in CODE},
        "artifact_sha256": {p: sha256(p) for p in ARTIFACTS},
        "blind_conditions": inv,
        "summary": {
            "revealed_with_data": sorted(revealed),
            "sealed": sorted(sealed),
            "defined_but_not_acquired": sorted(acquired_no_pred),
            "n_revealed": len(revealed),
            "n_sealed": len(sealed),
        },
        "rules": [
            "Never re-tune on BLND01-10 and then report them as confirmatory.",
            "Any post-hoc model change is exploratory until validated on a new block.",
            "Re-run with --verify before submission; any hash change invalidates the "
            "reported blind numbers until re-audited.",
        ],
    }


def verify() -> int:
    if not OUT.exists():
        print(f"no manifest at {OUT} -- run without --verify first")
        return 1
    old = json.loads(OUT.read_text())
    bad = []
    for section in ("code_sha256", "artifact_sha256"):
        for p, h in old.get(section, {}).items():
            now = sha256(p)
            if h != now:
                bad.append((p, h, now))
    for cid, rec in old.get("blind_conditions", {}).items():
        for f, h in (rec.get("file_sha256") or {}).items():
            now = sha256(f"datasets/{f}")
            if h != now:
                bad.append((f"{cid}:{f}", h, now))
    if not bad:
        n = len(old.get("code_sha256", {})) + len(old.get("artifact_sha256", {}))
        print(f"OK -- all hashes match the manifest frozen at {old['frozen_at']} "
              f"({n} tracked files + blind recordings)")
        return 0
    print(f"CHANGED since {old['frozen_at']}:")
    for p, h, now in bad:
        print(f"  {p}\n    was {h}\n    now {now}")
    print("\nBlind numbers reported from the old state are not valid for the new "
          "state until re-audited.")
    return 2


def main():
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    rec = build()
    OUT.write_text(json.dumps(rec, indent=2))
    s = rec["summary"]
    print(f"wrote {OUT}")
    print(f"  revealed (spent): {s['n_revealed']}  -> {', '.join(s['revealed_with_data'])}")
    print(f"  still sealed:     {s['n_sealed']}")
    print(f"  defined, no data: {len(s['defined_but_not_acquired'])}")
    rev = rec["git_rev"] or "(not a git repo -- consider git init for real provenance)"
    print(f"  git rev: {rev}")
    print("\n  " + rec["provenance"])


if __name__ == "__main__":
    main()
