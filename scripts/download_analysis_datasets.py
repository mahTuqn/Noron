"""Download and extract additional CC BY 4.0 helmet/PPE analysis datasets.

Datasets are stored under ext/ (short paths avoid Windows MAX_PATH issues) and are never merged into the project's
original train/validation/test split. Re-run this script to reuse complete cached zips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "ext"

DATASETS = {
    "deboniz_helmet": {
        "title": "Safety Helmet Detection Dataset (YOLO Format)",
        "version": "Hugging Face commit a14955d41c4209935d2bee70a2e7224a8fdc0f4a",
        "source_revision": "a14955d41c4209935d2bee70a2e7224a8fdc0f4a",
        "url": "https://huggingface.co/datasets/DeBoniz/helmet-detection-dataset/resolve/a14955d41c4209935d2bee70a2e7224a8fdc0f4a/helmet-detection-dataset.zip",
        "expected_bytes": 1_466_442_744,
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/DeBoniz/helmet-detection-dataset",
        "attribution": "DeBoniz, Safety Helmet Detection Dataset (YOLO Format), Hugging Face Datasets, CC BY 4.0.",
        "scope": "External-domain analysis: urban traffic, surveillance, and industrial scenes; classes Helmet and No-Helmet.",
    },
    "mendeley_ppe": {
        "title": "Dataset of Personal Protective Equipment (PPE)",
        "version": "Mendeley Data v6, DOI 10.17632/zkzghjvpn2.6",
        "url": "https://data.mendeley.com/public-api/zip/zkzghjvpn2/download/6",
        "expected_bytes": 235_392_977,
        "expected_inner_archive_sha256": "7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d",
        "license": "CC BY 4.0",
        "source": "https://data.mendeley.com/datasets/zkzghjvpn2/6",
        "attribution": "Huang, Mei-Ling; Cheng, Ying. Dataset of Personal Protective Equipment (PPE), Mendeley Data v6, DOI 10.17632/zkzghjvpn2.6, CC BY 4.0.",
        "scope": "Construction/industrial PPE image analysis; includes Helmet, NoHelmet, NoVest, and Vest labels and mixed upstream sources.",
    },
    "mendeley_motorcycle_video": {
        "title": "Dataset for the Detection of Helmet Use in Motorcycle Drivers",
        "version": "Mendeley Data v1, DOI 10.17632/bmy35m25pw.1",
        "url": "https://data.mendeley.com/public-api/zip/bmy35m25pw/download/1",
        "license": "CC BY 4.0",
        "source": "https://data.mendeley.com/datasets/bmy35m25pw/1",
        "attribution": "Mercado Reyna et al. Dataset for the Detection of Helmet Use in Motorcycle Drivers, Mendeley Data v1, DOI 10.17632/bmy35m25pw.1, CC BY 4.0.",
        "scope": "32 motorcycle-driver videos: same 16 participants recorded with and without a helmet. Out-of-domain video analysis; no frame-level bounding boxes are supplied.",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(slug: str) -> tuple[str, Path]:
    spec = DATASETS[slug]
    base = EXTERNAL / slug
    downloads = base / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    filename = Path(unquote(urlparse(spec["url"]).path)).name
    if not filename.lower().endswith(".zip"):
        filename = f"{slug}.zip"
    target = downloads / filename
    expected = spec.get("expected_bytes")
    if target.exists() and (expected is None or target.stat().st_size == expected):
        return slug, target

    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    response = requests.get(spec["url"], stream=True, timeout=(30, 180), headers={"User-Agent": "Noron-Helmet-Analysis/1.0"})
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or expected or 0)
    written = 0
    last_report = time.monotonic()
    with partial.open("wb") as f:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            written += len(chunk)
            if time.monotonic() - last_report >= 15:
                pct = f"{written / total * 100:.1f}%" if total else f"{written / 2**20:.0f} MiB"
                print(f"[{slug}] {pct} ({written / 2**20:.0f} MiB)", flush=True)
                last_report = time.monotonic()
    response.close()
    if total and written != total:
        raise RuntimeError(f"{slug}: downloaded {written} bytes, expected {total}")
    partial.replace(target)
    return slug, target


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            candidate = (destination / member.filename).resolve()
            if candidate != base and base not in candidate.parents:
                raise RuntimeError(f"Unsafe path in {archive.name}: {member.filename}")
        for i, member in enumerate(members, 1):
            zf.extract(member, destination)
            if i % 5000 == 0:
                print(f"[extract {archive.name}] {i}/{len(members)} entries", flush=True)


def extract_nested_archives(root: Path, max_depth: int = 3) -> None:
    """Mendeley sometimes publishes a zip containing the actual data zip."""
    for _ in range(max_depth):
        nested = [p for p in root.rglob("*.zip") if not p.with_suffix("").exists()]
        if not nested:
            return
        for archive in nested:
            destination = archive.with_suffix("")
            print(f"[nested extract] {archive} -> {destination}", flush=True)
            safe_extract(archive, destination)


def write_metadata(slug: str, archive: Path) -> None:
    spec = dict(DATASETS[slug])
    spec.update({
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive_relative_path": str(archive.relative_to(ROOT)).replace("\\", "/"),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "status": "downloaded; extracted separately; not merged into the original benchmark split",
    })
    base = archive.parents[1]
    (base / "source.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_embedded_archive(slug: str) -> None:
    expected = DATASETS[slug].get("expected_inner_archive_sha256")
    if not expected:
        return
    extracted = EXTERNAL / slug / "extracted"
    candidates = list(extracted.rglob("20250731-PPE2286y.zip")) if extracted.exists() else []
    if not candidates:
        raise RuntimeError(f"{slug}: expected embedded archive was not found after extraction")
    embedded = candidates[0]
    actual = sha256_file(embedded)
    if actual != expected:
        raise RuntimeError(f"{slug}: embedded archive SHA-256 mismatch ({actual})")
    metadata_path = EXTERNAL / slug / "source.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["verified_embedded_archive"] = str(embedded.relative_to(ROOT)).replace("\\", "/")
    metadata["verified_embedded_archive_sha256"] = actual
    metadata["embedded_sha256_record"] = "Mendeley Data public API content_details.sha256_hash"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", *DATASETS], default="all")
    parser.add_argument("--no-extract", action="store_true", help="download archives and write provenance only")
    args = parser.parse_args()
    slugs = list(DATASETS) if args.source == "all" else [args.source]
    failures: list[str] = []
    downloaded: dict[str, Path] = {}

    with ThreadPoolExecutor(max_workers=min(3, len(slugs))) as pool:
        future_map = {pool.submit(download_one, slug): slug for slug in slugs}
        for future in as_completed(future_map):
            slug = future_map[future]
            try:
                _, archive = future.result()
                downloaded[slug] = archive
                print(f"[{slug}] archive ready: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)", flush=True)
            except Exception as exc:
                failures.append(f"{slug}: {type(exc).__name__}: {exc}")
                print(f"[{slug}] FAILED: {exc}", file=sys.stderr, flush=True)

    for slug, archive in downloaded.items():
        try:
            write_metadata(slug, archive)
            if not args.no_extract:
                target = EXTERNAL / slug / "extracted"
                already_extracted = target.exists() and any(target.iterdir())
                if not already_extracted:
                    print(f"[{slug}] extracting to {target}", flush=True)
                    safe_extract(archive, target)
                else:
                    print(f"[{slug}] extracted directory already populated; keeping it", flush=True)
                extract_nested_archives(target)
                verify_embedded_archive(slug)
            print(f"[{slug}] sha256={sha256_file(archive)}", flush=True)
        except Exception as exc:
            failures.append(f"{slug} extract/metadata: {type(exc).__name__}: {exc}")
            print(f"[{slug}] FAILED: {exc}", file=sys.stderr, flush=True)

    if failures:
        print("\nFailures:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
