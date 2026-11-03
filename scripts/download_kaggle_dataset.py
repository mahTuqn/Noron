"""Download a public Kaggle dataset and save source metadata beside it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="andrewmvd/hard-hat-detection")
    parser.add_argument("--output", default="data/raw/kaggle_hard_hat_detection.zip")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_url = f"https://www.kaggle.com/api/v1/datasets/view/{args.ref}"
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{args.ref}"
    session = requests.Session()

    meta_response = session.get(meta_url, timeout=45)
    meta_response.raise_for_status()
    meta = meta_response.json()
    print(f"Dataset: {meta.get('title', args.ref)}")
    print(f"Owner: {meta.get('ownerName', 'unknown')}")
    print(f"License: {meta.get('licenseName', 'not stated')}")
    print(f"Kaggle page: https://www.kaggle.com/datasets/{args.ref}")
    print(f"Description: {(meta.get('description') or '').replace(chr(10), ' ')[:500]}")

    digest_hex = None
    if output.exists() and output.stat().st_size > 0:
        print(f"Archive already exists ({output.stat().st_size:,} bytes), skipping download.")
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest_hex = digest.hexdigest()
    else:
        partial = output.with_suffix(output.suffix + ".part")
        response = session.get(download_url, stream=True, timeout=(45, 180))
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        digest = hashlib.sha256()
        received = 0
        try:
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if total and (received == total or received // (128 * 1024 * 1024) != (received - len(chunk)) // (128 * 1024 * 1024)):
                        print(f"Downloaded {received / total:.0%} ({received:,}/{total:,} bytes)", flush=True)
                stream.flush()
                os.fsync(stream.fileno())
            if total and received != total:
                raise RuntimeError(f"Incomplete download: got {received}, expected {total} bytes")
            partial.replace(output)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        digest_hex = digest.hexdigest()
        print(f"SHA-256: {digest_hex}")

    metadata_path = output.with_suffix(output.suffix + ".source.json")
    source = {
        "ref": args.ref,
        "page": f"https://www.kaggle.com/datasets/{args.ref}",
        "title": meta.get("title"),
        "owner": meta.get("ownerName"),
        "license": meta.get("licenseName"),
        "version": meta.get("currentVersionNumber"),
        "archive": str(output.relative_to(root)),
        "archive_bytes": output.stat().st_size,
        "sha256": digest_hex,
    }
    metadata_path.write_text(json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
