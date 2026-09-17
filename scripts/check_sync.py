"""
Verify the canonical tree and the AdminPanel mirror hold identical sources.

WHY THIS EXISTS. This check was previously written as

    diff -rq --include="*.tex" TREE_A TREE_B 2>/dev/null | grep -v ...

`--include` is not a GNU diff option. diff rejected it, exited immediately, the
error went to /dev/null, and the grep found nothing, so the check reported "in
sync" no matter what the trees contained. It was vacuous for its whole working
life. Content hashing is used here instead, and a difference sets the exit code
so the failure cannot be read as success.

    python -m scripts.check_sync            # report
    python -m scripts.check_sync --stale     # also flag artefacts older than the
                                             # code that generates them

Exit codes: 0 identical, 1 differences found, 2 a tree is missing.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

CANON = pathlib.Path(r"C:\Users\HP\Desktop\ROTOR")
MIRROR = pathlib.Path(
    r"C:\Users\HP\Desktop\Journal papers\Rotor_Kit\AdminPanel\AdminPanel\ROTOR")

PATTERNS = ("*.md", "paper/*.tex", "paper/*.bib", "paper/*.md",
            "manuscript/*.md", "estimation/*.py", "scripts/*.py",
            "processing/*.py", "model/*.py", "matlab/*.m")

# generated artefact -> the sources whose change should invalidate it
DEPENDS = {
    "analysis/benchmark.json":            ["scripts/benchmark.py", "estimation/stats.py",
                                           "estimation/twin.py", "estimation/ml.py"],
    "analysis/RESULTS.md":                ["scripts/benchmark.py"],
    "analysis/representation_test.json":  ["scripts/representation_test.py",
                                           "estimation/stats.py"],
    "analysis/angle_novelty_control.json": ["scripts/angle_novelty_control.py"],
    "analysis/pinn_gate.json":            ["scripts/pinn_gate.py", "estimation/pinn.py"],
    "analysis/pinn_validate.json":        ["scripts/pinn_validate.py", "estimation/pinn.py"],
    "analysis/pinn_modes.json":           ["scripts/pinn_modes.py", "estimation/pinn.py"],
    "analysis/repeatability.json":        ["scripts/repeatability.py"],
}


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    for t in (CANON, MIRROR):
        if not t.is_dir():
            print(f"MISSING TREE: {t}")
            return 2

    same, differ, only_canon, only_mirror = 0, [], [], []
    for pat in PATTERNS:
        for f in sorted(CANON.glob(pat)):
            rel = f.relative_to(CANON).as_posix()
            g = MIRROR / rel
            if not g.exists():
                only_canon.append(rel)
            elif sha(f) != sha(g):
                differ.append(rel)
            else:
                same += 1
        for g in sorted(MIRROR.glob(pat)):
            rel = g.relative_to(MIRROR).as_posix()
            if not (CANON / rel).exists():
                only_mirror.append(rel)

    print(f"identical        : {same}")
    print(f"DIFFERENT        : {len(differ)}")
    for d in differ:
        print(f"   {d}")
    if only_canon:
        print(f"only in canonical: {len(only_canon)}")
        for d in only_canon:
            print(f"   {d}")
    if only_mirror:
        print(f"only in mirror   : {len(only_mirror)}")
        for d in only_mirror:
            print(f"   {d}")

    stale = []
    if "--stale" in sys.argv:
        print("\nartefact freshness (canonical):")
        for art, srcs in DEPENDS.items():
            a = CANON / art
            if not a.exists():
                print(f"   MISSING   {art}")
                continue
            at = a.stat().st_mtime
            newer = [s for s in srcs
                     if (CANON / s).exists() and (CANON / s).stat().st_mtime > at]
            if newer:
                stale.append(art)
                print(f"   STALE     {art}  (older than {', '.join(newer)})")
            else:
                print(f"   ok        {art}")

    bad = bool(differ or only_canon or only_mirror or stale)
    print("\n" + ("DIFFERENCES FOUND" if bad else "trees identical and artefacts current"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
