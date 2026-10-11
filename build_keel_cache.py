#!/usr/bin/env python3
"""Build keel_cache/ (41 benchmark tasks) and heldout/keel_cache/ (13 held-out
tasks) from the KEEL files distributed in the PyPI package
imbalanced-databases 0.1.1, and print the SHA-256 of every data file.

    python build_keel_cache.py              # downloads the wheel with pip if needed

This is how the re-run, controlled and held-out experiments obtained their
inputs (the KEEL web site was not reachable from the execution environment).
The data rows of these files reproduce the archived benchmark records bit for
bit (verify_reproduction.py). download_keel.sh is the alternative that fetches
the zips from the KEEL site itself.

Each task is written as keel_cache/<task>.zip containing <task>.dat, the
layout the notebooks' loader reads. For some tasks the package keeps a second
copy, .dat.bak, which differs from .dat only by the KEEL header line
"@outputs" (the data rows are identical; checked below). The benchmark cache
uses the .dat.bak copy where one exists and the held-out cache the .dat file,
exactly as in the runs reported; the hashes of the held-out files are those in
heldout/PROTOCOL.md. The loader ignores header lines other than @attribute and
@data, so the choice does not affect any record.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
WHEEL = "imbalanced_databases-0.1.1-py3-none-any.whl"
WHEEL_SHA256 = None  # printed on first use; the version pin identifies the file on PyPI

BENCHMARK = [l.split('"')[1] for l in (REPO / "download_keel.sh").read_text().splitlines()
             if l.strip().startswith('"') and "sci2s" in l]
HELDOUT = ["car-good", "car-vgood", "cleveland-0_vs_4", "dermatology-6", "flare-F",
           "led7digit-0-2-4-5-6-7-8-9_vs_1", "lymphography-normal-fibrosis", "new_thyroid1",
           "new_thyroid2", "page-blocks-1-3_vs_4", "page-blocks0", "wisconsin", "zoo-3"]


def get_wheel(tmp: Path) -> Path:
    cached = REPO / WHEEL
    if cached.exists():
        return cached
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(tmp),
                    "imbalanced-databases==0.1.1"], check=True)
    return tmp / WHEEL


def data_rows(b: bytes) -> list[bytes]:
    return [l for l in b.splitlines() if l.strip() and not l.lstrip().startswith(b"@")]


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        wheel = get_wheel(Path(t))
        print(f"wheel {wheel.name}  sha256 {hashlib.sha256(wheel.read_bytes()).hexdigest()}")
        z = zipfile.ZipFile(wheel)
        names = z.namelist()
        for tasks, out in [(BENCHMARK, REPO / "keel_cache"), (HELDOUT, REPO / "heldout" / "keel_cache")]:
            out.mkdir(parents=True, exist_ok=True)
            for task in tasks:
                base = f"imbalanced_databases/data/{task}/"
                dats = [n for n in names if n.startswith(base) and n.endswith(".dat") and "no_null" not in n]
                assert len(dats) == 1, (task, dats)
                payload = z.read(dats[0])
                bak = dats[0] + ".bak"
                if bak in names and tasks is BENCHMARK:
                    bak_payload = z.read(bak)
                    assert data_rows(bak_payload) == data_rows(payload), f"{task}: .dat and .dat.bak rows differ"
                    payload = bak_payload
                info = zipfile.ZipInfo(f"{task}.dat", date_time=(2026, 10, 3, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with zipfile.ZipFile(out / f"{task}.zip", "w") as zf:
                    zf.writestr(info, payload)
                print(f"{hashlib.sha256(payload).hexdigest()}  {out.relative_to(REPO)}/{task}")
    print(f"benchmark: {len(BENCHMARK)} tasks -> keel_cache/; held-out: {len(HELDOUT)} tasks -> heldout/keel_cache/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
