"""
Build the handoff package(s) for the digital-twin group.

    python -m scripts.make_handoff            # both: lite (~15 MB) + full (~2.2 GB)
    python -m scripts.make_handoff --lite     # code + tables + analysis, no raw .npz
    python -m scripts.make_handoff --full     # everything incl. all raw recordings

Writes next to the project folder (not inside it). Raw .npz are stored uncompressed
(they don't deflate) so the FULL build stays fast; everything else is compressed.
"""
from __future__ import annotations

import argparse
import os
import time
import zipfile

from config import ROOT

OUT_DIR = ROOT.parent
EXCLUDE_DIRS = {"__pycache__", "_superseded_20260726", ".git", ".venv", "node_modules"}
# reference recordings kept even in the lite build (small, needed by the pipeline)
KEEP_NPZ = {"runout_slowroll.npz", "runup1.npz"}


def excluded_file(rel: str) -> bool:
    name = os.path.basename(rel)
    if name.startswith("Gmail") and name.endswith(".zip"):
        return True
    if name.startswith("index_backup_"):
        return True
    if name.endswith(".pyc"):
        return True
    if name.startswith("ROTOR_handoff"):
        return True
    if name == "unnamed.jpg":       # re-added under a descriptive name
        return True
    return False


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            if not excluded_file(rel):
                yield full, rel.replace(os.sep, "/")


def build(zip_path, include_raw: bool):
    n = raw = skipped = 0
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as z:
        arcroot = "ROTOR"
        for full, rel in iter_files():
            is_npz = rel.endswith(".npz")
            if is_npz and not (include_raw or os.path.basename(rel) in KEEP_NPZ):
                skipped += 1
                continue
            # store (no deflate) for raw signals — they don't compress
            ctype = zipfile.ZIP_STORED if is_npz else zipfile.ZIP_DEFLATED
            z.write(full, f"{arcroot}/{rel}", compress_type=ctype)
            n += 1
            raw += is_npz
        # add the rig photo under a descriptive name
        photo = ROOT / "unnamed.jpg"
        if photo.exists():
            z.write(photo, f"{arcroot}/rig_balance_disk.jpg")
            n += 1
    mb = os.path.getsize(zip_path) / 1e6
    print(f"  {os.path.basename(zip_path)}: {n} files ({raw} raw .npz), "
          f"{mb:,.1f} MB, {time.time()-t0:.1f}s  (skipped {skipped} raw)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    do_lite = a.lite or not (a.lite or a.full)
    do_full = a.full or not (a.lite or a.full)

    print(f"packaging from {ROOT} -> {OUT_DIR}")
    if do_lite:
        build(OUT_DIR / "ROTOR_handoff.zip", include_raw=False)
    if do_full:
        build(OUT_DIR / "ROTOR_handoff_FULL.zip", include_raw=True)
    print("done.")


if __name__ == "__main__":
    main()
