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
    ap.add_argument("--max-part-mb", type=int, default=0,
                    help="split a block into parts of at most this size. Each part "
                         "is a standalone tar, so a dropped upload costs one part "
                         "rather than the whole block. 0 means one tar per block.")
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

    cap = args.max_part_mb * 2**20 if args.max_part_mb else 0

    # Decide the archive layout before reporting it, so a dry run previews
    # exactly what a real run would write.
    plan: list[tuple[str, list[pathlib.Path]]] = []
    for b in sorted(groups):
        parts: list[list[pathlib.Path]] = [[]]
        running = 0
        for p in groups[b]:
            sz = p.stat().st_size
            if cap and parts[-1] and running + sz > cap:
                parts.append([])
                running = 0
            parts[-1].append(p)
            running += sz
        for i, members in enumerate(parts, 1):
            name = f"{b}.tar" if len(parts) == 1 else f"{b}.part{i:02d}.tar"
            plan.append((name, members))

    total = sum(p.stat().st_size for ps in groups.values() for p in ps)
    print(f"  {sum(len(v) for v in groups.values())} recordings, "
          f"{total / 2**30:.2f} GB, in {len(plan)} archives"
          + (f" capped at {args.max_part_mb} MB\n" if cap else "\n"))
    for name, members in plan:
        sz = sum(p.stat().st_size for p in members) / 2**20
        print(f"    {name:26s} {len(members):4d} files  {sz:7.0f} MB")

    if args.dry_run:
        print("\n  dry run, nothing written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for name, members in plan:
        dest = out / name
        print(f"\n  writing {name} ({len(members)} files) ...", end="", flush=True)
        with tarfile.open(dest, "w") as t:
            for p in members:
                t.add(p, arcname=p.name)
        print(f" {dest.stat().st_size / 2**20:.0f} MB")

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
