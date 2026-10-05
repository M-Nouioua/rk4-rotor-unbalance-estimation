"""
Package the raw acquisitions for deposition.

Groups the 563 recordings into one archive per campaign block, so a reader who
wants only the blind set downloads 0.19 GB instead of 2.2 GB. The archives are
uncompressed tar, because .npz is already a compressed container and gzip buys
under 5 per cent on this data.

Superseded recordings and the index backup are excluded: they are not referenced
by any reported result, and shipping them invites the wrong file being used.

    python -m scripts.package_dataset --out D:/zenodo_upload

Writes one tar per block, the four metadata CSVs, DATA_README.md, and a
SHA256SUMS file covering everything, so a downloader can verify the transfer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
import shutil
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
METADATA = ("conditions.csv", "index.csv", "speeds.csv", "runup_1x.csv")
EXCLUDE_DIRS = ("_superseded_20260726",)
EXCLUDE_FILES = ("index_backup_20260726.csv",)
REFERENCE = ("runup1.npz", "runout_slowroll.npz")


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blocks() -> dict[str, str]:
    with open(DATASETS / "conditions.csv", newline="", encoding="utf-8") as f:
        return {r["condition_id"]: r["block"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for the upload set")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    cid_block = blocks()

    groups: dict[str, list[pathlib.Path]] = {}
    unassigned = []
    for p in sorted(DATASETS.glob("*.npz")):
        if p.name in REFERENCE:
            groups.setdefault("reference", []).append(p)
            continue
        cid = p.name.split("_")[0]
        b = cid_block.get(cid)
        if b is None:
            unassigned.append(p.name)
            continue
        groups.setdefault(b, []).append(p)

    if unassigned:
        print(f"  {len(unassigned)} file(s) match no condition; not packaged:")
        for n in unassigned[:5]:
            print(f"     {n}")
        print("  resolve these before depositing")
        return 1

    total = sum(p.stat().st_size for ps in groups.values() for p in ps)
    print(f"  {sum(len(v) for v in groups.values())} recordings, "
          f"{total / 2**30:.2f} GB, in {len(groups)} archives\n")
    for b in sorted(groups):
        n = len(groups[b])
        sz = sum(p.stat().st_size for p in groups[b]) / 2**30
        print(f"    {b:14s} {n:4d} files  {sz:5.2f} GB  ->  {b}.tar")

    if args.dry_run:
        print("\n  dry run, nothing written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for b in sorted(groups):
        dest = out / f"{b}.tar"
        print(f"\n  writing {dest.name} ...", end="", flush=True)
        with tarfile.open(dest, "w") as t:
            for p in groups[b]:
                t.add(p, arcname=p.name)
        print(f" {dest.stat().st_size / 2**30:.2f} GB")

    for name in METADATA:
        src = DATASETS / name
        if src.exists():
            shutil.copy2(src, out / name)
            print(f"  copied {name}")
    readme = ROOT / "DATA_README.md"
    if readme.exists():
        shutil.copy2(readme, out / "DATA_README.md")
        print("  copied DATA_README.md")

    print("\n  hashing the upload set ...")
    lines = []
    for p in sorted(out.iterdir()):
        if p.name == "SHA256SUMS" or p.is_dir():
            continue
        lines.append(f"{sha256(p)}  {p.name}")
        print(f"    {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n  upload set ready in {out}")
    print(f"  {len(lines)} files, {sum(p.stat().st_size for p in out.iterdir() if p.is_file()) / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
