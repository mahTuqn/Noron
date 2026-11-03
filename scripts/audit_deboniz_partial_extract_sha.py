#!/usr/bin/env python3
"""Compare DeBoniz partial-extract stills to the complete canonical extract by SHA-256."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTIAL_ROOT = ROOT / "ext" / "deboniz_helmet" / "partial_extract_1" / "datasetmodelloscelto" / "dataset10kadjusted"
CANONICAL_ROOT = ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted"
OUTPUT_DIR = ROOT / "results" / "external_data_analysis"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CSV_PATH = OUTPUT_DIR / "deboniz_partial_extract_sha_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "deboniz_partial_extract_sha_audit_summary.json"
REPORT_PATH = OUTPUT_DIR / "deboniz_partial_extract_sha_audit_report.md"


def images(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.as_posix().lower())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    partial_files = images(PARTIAL_ROOT)
    canonical_files = images(CANONICAL_ROOT)
    canonical_by_split_name: dict[tuple[str, str], list[Path]] = defaultdict(list)
    canonical_by_name: dict[str, list[Path]] = defaultdict(list)
    for p in canonical_files:
        split = p.parent.parent.name
        canonical_by_split_name[(split, p.name)].append(p)
        canonical_by_name[p.name].append(p)

    canonical_hash_cache: dict[Path, str] = {}
    counts: Counter[str] = Counter()
    matched_canonical_paths: set[Path] = set()
    rows = []
    for p in partial_files:
        partial_split = p.parent.parent.name
        partial_hash = sha256(p)
        exact_split_candidates = canonical_by_split_name.get((partial_split, p.name), [])
        fallback_candidates = canonical_by_name.get(p.name, []) if not exact_split_candidates else []
        candidates = exact_split_candidates or fallback_candidates
        candidate_hashes = []
        exact_matches = []
        for candidate in candidates:
            if candidate not in canonical_hash_cache:
                canonical_hash_cache[candidate] = sha256(candidate)
            digest = canonical_hash_cache[candidate]
            candidate_hashes.append((candidate, digest))
            if digest == partial_hash:
                exact_matches.append(candidate)

        if exact_matches and exact_split_candidates:
            status = "exact_sha256_match_same_split_and_basename"
            canonical_match = exact_matches[0]
            matched_canonical_paths.add(canonical_match)
        elif exact_matches:
            status = "exact_sha256_match_same_basename_other_split"
            canonical_match = exact_matches[0]
            matched_canonical_paths.add(canonical_match)
        elif candidates:
            status = "sha256_mismatch_same_basename"
            canonical_match = candidates[0]
        else:
            status = "no_canonical_basename_match"
            canonical_match = None

        counts[status] += 1
        rows.append({
            "partial_path": rel(p),
            "partial_split": partial_split,
            "partial_sha256": partial_hash,
            "partial_size_bytes": p.stat().st_size,
            "canonical_match_path": rel(canonical_match) if canonical_match else "",
            "canonical_candidate_paths_json": json.dumps([rel(c) for c in candidates], ensure_ascii=False),
            "canonical_candidate_sha256_json": json.dumps([{"path": rel(c), "sha256": h} for c, h in candidate_hashes], ensure_ascii=False),
            "canonical_candidate_count": len(candidates),
            "same_split_and_basename_candidate": bool(exact_split_candidates),
            "exact_sha256_match": bool(exact_matches),
            "match_status": status,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "partial_path", "partial_split", "partial_sha256", "partial_size_bytes", "canonical_match_path",
        "canonical_candidate_paths_json", "canonical_candidate_sha256_json", "canonical_candidate_count",
        "same_split_and_basename_candidate", "exact_sha256_match", "match_status",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    exact_count = sum(v for k, v in counts.items() if k.startswith("exact_sha256_match"))
    mismatched_count = counts["sha256_mismatch_same_basename"]
    missing_count = counts["no_canonical_basename_match"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "partial_image_count": len(partial_files),
        "canonical_image_count": len(canonical_files),
        "exact_sha256_match_count": exact_count,
        "same_split_and_basename_exact_match_count": counts["exact_sha256_match_same_split_and_basename"],
        "same_basename_other_split_exact_match_count": counts["exact_sha256_match_same_basename_other_split"],
        "sha256_mismatch_count": mismatched_count,
        "canonical_basename_missing_count": missing_count,
        "canonical_files_matched_by_partial": len(matched_canonical_paths),
        "canonical_files_not_present_in_partial": len(canonical_files) - len(matched_canonical_paths),
        "partial_rows_by_match_status": dict(sorted(counts.items())),
        "partial_root": rel(PARTIAL_ROOT),
        "canonical_root": rel(CANONICAL_ROOT),
        "csv_path": rel(CSV_PATH),
        "all_partial_files_exactly_match_canonical": len(partial_files) > 0 and exact_count == len(partial_files) and mismatched_count == 0 and missing_count == 0,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# DeBoniz partial extraction SHA-256 audit",
        "",
        f"Compared **{len(partial_files):,}** partial-extract still images against **{len(canonical_files):,}** images in the complete canonical extract.",
        "",
        f"- Exact SHA-256 matches: **{exact_count:,}** ({counts['exact_sha256_match_same_split_and_basename']:,} same split and basename; {counts['exact_sha256_match_same_basename_other_split']:,} same basename in another split).",
        f"- SHA-256 mismatches with a same-basename candidate: **{mismatched_count:,}**.",
        f"- No canonical same-basename candidate: **{missing_count:,}**.",
        f"- Canonical files represented by the partial extraction: **{len(matched_canonical_paths):,}**; canonical files not represented there: **{len(canonical_files) - len(matched_canonical_paths):,}**.",
        "",
        f"Per-file hashes and match paths are recorded in `{CSV_PATH.relative_to(ROOT).as_posix()}`.",
        "Source images and labels were read only; no source data was modified or copied.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
