"""
Upload the packaged raw acquisitions to Zenodo as a draft deposition.

This script never publishes. It creates a draft, uploads the files, attaches the
metadata and the link to the code record, then prints the draft URL for review.
Publishing mints a permanent DOI that cannot be withdrawn, so that step is left
to a person looking at the finished record.

The token is read from the ZENODO_TOKEN environment variable and is never
written to disk, echoed, or included in any error message. Set it in the shell
that runs this, not in a file that might be committed:

    $env:ZENODO_TOKEN = "..."       # PowerShell  <- the usual shell here
    set ZENODO_TOKEN=...            # Windows cmd
    export ZENODO_TOKEN=...         # bash

Usage:

    python -m scripts.zenodo_upload --dir D:/zenodo_upload            # dry run
    python -m scripts.zenodo_upload --dir D:/zenodo_upload --create

Uploads resume: a file already present in the draft with a matching size is
skipped, so a run interrupted partway can simply be repeated with the same
--deposition id.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import requests

API = "https://zenodo.org/api"
CODE_DOI = "10.5281/zenodo.23152815"
TIMEOUT = (30, 1800)   # connect, read

METADATA = {
    "title": "Raw acquisitions: Bently Nevada RK-4 rotor-kit unbalance campaign",
    "upload_type": "dataset",
    "description": (
        "<p>Raw proximity-probe and keyphasor recordings for a two-plane rotor "
        "balancing campaign on a Bently Nevada RK-4 rotor kit, with the condition "
        "design and acquisition log needed to interpret them.</p>"
        "<p>563 recordings, about 2.2 GB, grouped into one archive per campaign "
        "block: severity ladder, angular coverage, fine angular steps, balanced "
        "baseline and trial-mass calibration, ten withheld conditions whose labels "
        "were sealed until after prediction, and the run-up and slow-roll "
        "references.</p>"
        "<p>Each record is 8.0 s on five channels at 12.8 kS/s: a once-per-revolution "
        "keyphasor pulse and two orthogonal proximity-probe pairs at two axial "
        "planes. Channels are stored as raw volts at the Proximitor output; the "
        "probes are calibrated at 7.874 mV per micrometre, so 1 V is 127.0 "
        "micrometres. The channels are AC-coupled, so the DC gap voltage and with "
        "it the shaft centreline are not recorded.</p>"
        "<p>See DATA_README.md in this record for the full format description, the "
        "file naming scheme, the rig parameters, and the two steps that must not be "
        "skipped when deriving first-order vectors: angular resampling and "
        "slow-roll runout subtraction.</p>"
        "<p>The analysis code that consumes these recordings is archived at "
        f"doi:{CODE_DOI}.</p>"
    ),
    "access_right": "open",
    "license": "cc-by-4.0",
    "related_identifiers": [
        {"identifier": CODE_DOI,
         "relation": "isSupplementTo",
         "resource_type": "software",
         "scheme": "doi"},
    ],
    "keywords": ["rotor balancing", "rotordynamics", "unbalance",
                 "condition monitoring", "influence coefficient method",
                 "proximity probe", "vibration", "benchmark dataset"],
}


def token() -> str:
    t = os.environ.get("ZENODO_TOKEN", "").strip()
    if not t:
        sys.exit(
            'ZENODO_TOKEN is not set. In PowerShell:\n'
            '    $env:ZENODO_TOKEN = "<token>"\n'
            'Then re-run. Do not pass it as an argument, where it would land in '
            'shell history.')
    return t


def scrub(e: Exception, tok: str) -> str:
    """Never let the token reach a log line."""
    return str(e).replace(tok, "<token>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the upload set from package_dataset")
    ap.add_argument("--create", action="store_true",
                    help="actually create the draft and upload; otherwise dry run")
    ap.add_argument("--deposition", type=int,
                    help="resume into an existing draft id instead of creating one")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    files = sorted(p for p in d.iterdir() if p.is_file())
    if not files:
        sys.exit(f"no files in {d}; run scripts/package_dataset.py first")

    total = sum(p.stat().st_size for p in files)
    print(f"  {len(files)} files, {total / 2**30:.2f} GB from {d}\n")
    for p in files:
        print(f"    {p.stat().st_size / 2**20:9.1f} MB  {p.name}")

    if not args.create:
        print("\n  dry run. Re-run with --create to upload.")
        print("  Nothing is published either way; this only ever makes a draft.")
        return 0

    tok = token()
    s = requests.Session()
    s.params = {"access_token": tok}

    try:
        if args.deposition:
            dep_id = args.deposition
            r = s.get(f"{API}/deposit/depositions/{dep_id}", timeout=TIMEOUT)
            r.raise_for_status()
            dep = r.json()
            print(f"\n  resuming draft {dep_id}")
        else:
            r = s.post(f"{API}/deposit/depositions", json={}, timeout=TIMEOUT)
            r.raise_for_status()
            dep = r.json()
            dep_id = dep["id"]
            print(f"\n  created draft {dep_id}")

        bucket = dep["links"]["bucket"]
        existing = {f["filename"]: f["filesize"] for f in dep.get("files", [])}

        for p in files:
            size = p.stat().st_size
            if existing.get(p.name) == size:
                print(f"    skip (already uploaded) {p.name}")
                continue
            print(f"    uploading {p.name} ({size / 2**20:.1f} MB) ...",
                  end="", flush=True)
            with open(p, "rb") as fh:
                r = s.put(f"{bucket}/{p.name}", data=fh, timeout=TIMEOUT)
            r.raise_for_status()
            print(" ok")

        r = s.put(f"{API}/deposit/depositions/{dep_id}",
                  json={"metadata": METADATA}, timeout=TIMEOUT)
        r.raise_for_status()
        print("\n  metadata attached")

        url = f"https://zenodo.org/uploads/{dep_id}"
        print(f"\n  DRAFT READY, NOT PUBLISHED")
        print(f"  review it at {url}")
        print("\n  Check the author list, the licence, and the related identifier,")
        print("  then publish from that page. Publishing mints a permanent DOI")
        print("  that cannot be withdrawn, so this script will not do it for you.")
    except requests.HTTPError as e:
        body = ""
        try:
            body = json.dumps(e.response.json())[:400]
        except Exception:
            body = e.response.text[:400] if e.response is not None else ""
        sys.exit(f"\n  Zenodo refused the request: {scrub(e, tok)}\n  {body}")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"\n  upload failed: {scrub(e, tok)}\n"
                 f"  re-run with --deposition {locals().get('dep_id', '<id>')} to resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
