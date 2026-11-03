#!/usr/bin/env python3
"""Build a read-only, full-coverage inventory of helmet project media.

The manifest includes the canonical combined still-image dataset, both
downloaded image datasets, and one row for every frame index in every
motorcycle video. Source annotations and media are never modified.

Run from any directory with:
    python E:/Noron/scripts/build_master_data_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guidance
    raise SystemExit("OpenCV is required to read video frame metadata (pip install opencv-python)") from exc


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CORE_ROOT = ROOT / "data" / "processed_combined"
DEBONIZ_ROOT = ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted"
PPE_DATA_ROOT = ROOT / "ext" / "mendeley_ppe" / "extracted"
VIDEO_DATA_ROOT = ROOT / "ext" / "mendeley_motorcycle_video" / "extracted"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "external_data_analysis"
EXPECTED_IMAGE_ROWS = 33_023
EXPECTED_VIDEO_CLIPS = 32
EXPECTED_VIDEO_FRAME_ROWS = 114_640

# IDs are source-local; source_class_name retains the original meaning.
CLASS_MAPS: dict[str, dict[int, str]] = {
    "core_processed_combined": {0: "wearing_helmet", 1: "not_wearing_helmet"},
    "deboniz_helmet": {0: "Helmet", 1: "No-Helmet"},
    "mendeley_ppe": {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"},
}

FIELDS = [
    "row_id", "record_type", "dataset", "source", "relative_path", "original_split",
    "original_split_supplied", "source_class_name", "source_class_id",
    "label_path", "label_file_status", "label_scope", "label_class_ids_json",
    "label_class_names_json", "label_class_counts_json", "parsed_annotation_rows",
    "parsed_bbox_rows", "polygon_like_rows", "malformed_label_rows",
    "helmet_or_nohelmet_box_count", "helmet_annotation_absent",
    "semantic_mapping_status", "coverage_status", "training_inclusion_status",
    "inclusion_rationale", "sha256", "exact_duplicate_group", "exact_duplicate_count",
    "exact_duplicate_splits_json", "exact_duplicate_cross_split", "paired_split_group",
    "video_class", "frame_index_0based", "timestamp_seconds", "fps",
    "frame_count", "clip_duration_seconds", "frame_annotation_status",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def image_files(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def annotation_inventory(label_path: Path, dataset: str) -> dict[str, Any]:
    class_map = CLASS_MAPS[dataset]
    out: dict[str, Any] = {
        "label_path": rel(label_path) if label_path.exists() else "",
        "label_file_status": "missing" if not label_path.exists() else "empty",
        "label_class_ids_json": "[]",
        "label_class_names_json": "[]",
        "label_class_counts_json": "{}",
        "parsed_annotation_rows": 0,
        "parsed_bbox_rows": 0,
        "polygon_like_rows": 0,
        "malformed_label_rows": 0,
        "helmet_or_nohelmet_box_count": 0,
        "helmet_annotation_absent": "unknown",
    }
    if not label_path.exists():
        return out

    counts: Counter[int] = Counter()
    raw_rows = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if raw_rows:
        out["label_file_status"] = "present_nonempty"
    for line in raw_rows:
        parts = line.split()
        try:
            class_id = int(parts[0])
            nums = [float(x) for x in parts[1:]]
            if not nums or not all(math.isfinite(value) for value in nums):
                raise ValueError("missing or non-finite coordinates")
        except (ValueError, IndexError, OverflowError):
            out["malformed_label_rows"] += 1
            continue
        counts[class_id] += 1
        out["parsed_annotation_rows"] += 1
        if len(parts) == 5:
            out["parsed_bbox_rows"] += 1
        elif len(parts) > 5:
            out["polygon_like_rows"] += 1
        else:
            out["malformed_label_rows"] += 1
        if dataset == "mendeley_ppe" and class_id in (0, 1):
            out["helmet_or_nohelmet_box_count"] += 1
        elif dataset in ("core_processed_combined", "deboniz_helmet") and class_id in (0, 1) and len(parts) == 5:
            out["helmet_or_nohelmet_box_count"] += 1

    class_ids = sorted(counts)
    out["label_class_ids_json"] = json_cell(class_ids)
    out["label_class_names_json"] = json_cell({str(cid): class_map.get(cid, f"unknown_class_{cid}") for cid in class_ids})
    out["label_class_counts_json"] = json_cell({str(cid): counts[cid] for cid in class_ids})
    if dataset == "mendeley_ppe":
        out["helmet_annotation_absent"] = "true" if out["helmet_or_nohelmet_box_count"] == 0 else "false"
    elif raw_rows:
        out["helmet_annotation_absent"] = "false" if out["helmet_or_nohelmet_box_count"] else "unknown"
    else:
        out["helmet_annotation_absent"] = "unknown"
    return out


def core_rows() -> list[dict[str, Any]]:
    manifest_path = CORE_ROOT / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Core manifest not found: {manifest_path}")
    out = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        for item in csv.DictReader(stream):
            split = item["split"].strip()
            path = CORE_ROOT / "images" / split / item["image"]
            if not path.is_file():
                raise FileNotFoundError(f"Core manifest image is missing: {path}")
            label_path = CORE_ROOT / "labels" / split / f"{path.stem}.txt"
            ann = annotation_inventory(label_path, "core_processed_combined")
            rec = {field: "" for field in FIELDS}
            rec.update({
                "row_id": f"core:{split}:{path.name}", "record_type": "still_image",
                "dataset": "core_processed_combined", "source": item.get("source", ""),
                "relative_path": rel(path), "original_split": split,
                "original_split_supplied": "true", "label_scope": "image_detection_boxes",
                "semantic_mapping_status": "mapped_to_project_classes",
                "coverage_status": "image_and_source_label_inventoried",
                "training_inclusion_status": "include_as_labeled" if ann["label_file_status"] == "present_nonempty" and not ann["malformed_label_rows"] else "review_label_status",
                "inclusion_rationale": "Core project's existing image and source label; source class IDs are preserved and mapped to project classes.",
                "sha256": sha256_file(path),
                "paired_split_group": "",
            })
            rec.update(ann)
            out.append(rec)
    # Detect files omitted by the core manifest rather than silently dropping them.
    discovered = {p.resolve() for split in ("train", "val", "test") for p in image_files(CORE_ROOT / "images" / split)}
    listed = {(CORE_ROOT / "images" / row["original_split"] / Path(row["relative_path"]).name).resolve() for row in out}
    if discovered != listed:
        raise RuntimeError(f"Core manifest/image mismatch: {len(discovered - listed)} unlisted, {len(listed - discovered)} missing")
    return out


def source_image_rows(dataset: str, root: Path, split_names: Iterable[str], class_semantics_status: str, rationale: str) -> list[dict[str, Any]]:
    out = []
    for split in split_names:
        image_dir = root / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Expected image directory not found: {image_dir}")
        label_dir = root / split / "labels"
        for path in image_files(image_dir):
            label_path = label_dir / f"{path.stem}.txt"
            ann = annotation_inventory(label_path, dataset)
            rec = {field: "" for field in FIELDS}
            rec.update({
                "row_id": f"{dataset}:{split}:{path.name}", "record_type": "still_image",
                "dataset": dataset, "source": dataset, "relative_path": rel(path),
                "original_split": split, "original_split_supplied": "true",
                "label_scope": "image_detection_boxes", "semantic_mapping_status": class_semantics_status,
                "coverage_status": "image_and_source_label_inventoried",
                "training_inclusion_status": "include_as_labeled",
                "inclusion_rationale": rationale,
                "sha256": sha256_file(path), "paired_split_group": "",
            })
            rec.update(ann)
            if ann["label_file_status"] == "missing" or ann["malformed_label_rows"] or ann["polygon_like_rows"]:
                rec["training_inclusion_status"] = "manual_label_review_required"
                rec["inclusion_rationale"] += " Source annotation is missing, malformed, or polygon-like; keep this row in the manifest and review/convert labels before detection training."
            elif dataset == "mendeley_ppe" and ann["helmet_annotation_absent"] == "true":
                rec["training_inclusion_status"] = "helmet_annotation_review_required"
                rec["inclusion_rationale"] += " PPE image has no Helmet/NoHelmet box; do not treat as verified helmet-free background until reviewed."
            out.append(rec)
    return out


def video_paths() -> list[Path]:
    return sorted((p for p in VIDEO_DATA_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}), key=lambda p: (p.parent.name.lower(), int(re.search(r"p(\d+)", p.stem, re.I).group(1)) if re.search(r"p(\d+)", p.stem, re.I) else 999, p.name.lower()))


def video_metadata(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video metadata: {path}")
    count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if count < 1 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid frame metadata for {path}: frame_count={count}, fps={fps}")
    return count, fps


def make_video_frame_records() -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    existing_inventory = ROOT / "results" / "external_data_analysis" / "video_inventory.csv"
    clip_specs: list[tuple[Path, str, str, int, float]] = []
    metadata_source = "opencv_container_metadata"
    if existing_inventory.is_file():
        with existing_inventory.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                path = ROOT / Path(row["relative_path"])
                if not path.is_file():
                    raise FileNotFoundError(f"Video inventory path is missing: {path}")
                participant_match = re.search(r"p(\d+)", path.stem, re.I)
                participant = participant_match.group(1) if participant_match else path.stem
                clip_specs.append((path, participant, row["clip_label"], int(row["frames"]), float(row["fps"])))
        metadata_source = rel(existing_inventory)
    else:
        paths = video_paths()
        for path in paths:
            participant_match = re.search(r"p(\d+)", path.stem, re.I)
            participant = participant_match.group(1) if participant_match else path.stem
            class_name = path.parent.name
            nframes, fps = video_metadata(path)
            clip_specs.append((path, participant, class_name, nframes, fps))

    video_paths_seen = [path.resolve() for path, *_ in clip_specs]
    if len(video_paths_seen) != len(set(video_paths_seen)):
        raise RuntimeError("Duplicate video paths in frame inventory")
    counts: Counter[str] = Counter()
    total_frames = 0
    for path, participant, class_name, nframes, fps in clip_specs:
        total_frames += nframes
        counts[class_name] += 1

    def generate() -> Iterator[dict[str, Any]]:
        for path, participant, class_name, nframes, fps in clip_specs:
            for frame_index in range(nframes):
                rec = {field: "" for field in FIELDS}
                rec.update({
                    "row_id": f"video:{rel(path)}:frame:{frame_index:06d}",
                    "record_type": "video_frame_index", "dataset": "mendeley_motorcycle_video",
                    "source": "mendeley_motorcycle_video", "relative_path": rel(path),
                    "original_split": "not_provided", "original_split_supplied": "false",
                    "source_class_name": class_name, "label_scope": "clip_level_only",
                    "coverage_status": "frame_index_inventoried_no_image_extracted",
                    "training_inclusion_status": "frame_annotation_required",
                    "inclusion_rationale": "Clip supplies only With/Without Helmet label; no frame-level person/head boxes exist. Annotate frame boxes before detection training.",
                    "paired_split_group": f"mendeley_participant_pair:p{participant}",
                    "video_class": class_name, "frame_index_0based": frame_index,
                    "timestamp_seconds": f"{frame_index / fps:.6f}", "fps": f"{fps:.6f}",
                    "frame_count": nframes, "clip_duration_seconds": f"{nframes / fps:.6f}",
                    "frame_annotation_status": "not_annotated_clip_label_only",
                })
                yield rec

    summary = {
        "video_count": len(clip_specs), "unique_video_path_count": len(set(video_paths_seen)),
        "frame_index_rows": total_frames,
        "clips_by_class": dict(sorted(counts.items())),
        "participant_pairs": len({f"p{participant}" for _, participant, _, _, _ in clip_specs}),
        "frame_index_convention": "zero_based; timestamp_seconds = frame_index / container FPS",
        "frame_images_extracted": False,
        "metadata_source": metadata_source,
    }
    return generate(), summary


def assign_exact_duplicate_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["sha256"]].append(row)
    cross_split_group_count = 0
    cross_split_row_count = 0
    duplicate_group_count = 0
    duplicate_row_count = 0
    for digest, rows in groups.items():
        splits = sorted({r["original_split"] for r in rows})
        duplicated = len(rows) > 1
        cross_split = len(splits) > 1
        duplicate_group_count += int(duplicated)
        duplicate_row_count += len(rows) if duplicated else 0
        cross_split_group_count += int(cross_split)
        cross_split_row_count += len(rows) if cross_split else 0
        group_id = f"sha256:{digest}"
        for row in rows:
            row["exact_duplicate_group"] = group_id
            row["exact_duplicate_count"] = len(rows)
            row["exact_duplicate_splits_json"] = json_cell(splits)
            row["exact_duplicate_cross_split"] = "true" if cross_split else "false"
    return {
        "unique_sha256": len(groups), "duplicate_groups": duplicate_group_count,
        "rows_in_duplicate_groups": duplicate_row_count,
        "cross_split_duplicate_groups": cross_split_group_count,
        "rows_in_cross_split_duplicate_groups": cross_split_row_count,
    }


def count_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(row["dataset"] for row in rows)
    by_dataset_split = Counter(f"{row['dataset']}|{row['original_split']}" for row in rows)
    by_inclusion = Counter(row["training_inclusion_status"] for row in rows)
    by_label_status = Counter(f"{row['dataset']}|{row['label_file_status']}" for row in rows)
    return {
        "image_rows": len(rows), "images_by_dataset": dict(sorted(by_dataset.items())),
        "images_by_dataset_and_original_split": dict(sorted(by_dataset_split.items())),
        "training_inclusion_status_counts": dict(sorted(by_inclusion.items())),
        "label_file_status_counts": dict(sorted(by_label_status.items())),
        "polygon_like_rows_total": sum(int(row["polygon_like_rows"] or 0) for row in rows),
        "helmet_annotation_absent_images": sum(row["helmet_annotation_absent"] == "true" for row in rows),
    }


def write_csv(path: Path, image_rows: list[dict[str, Any]], video_rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_rows)
        count = len(image_rows)
        for row in video_rows:
            writer.writerow(row)
            count += 1
    return count


def verify_manifest_csv(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    video_paths_seen: set[str] = set()
    video_row_ids: set[str] = set()
    frame_ranges: dict[str, list[int]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            counts[row["record_type"]] += 1
            if row["record_type"] == "video_frame_index":
                video_paths_seen.add(row["relative_path"])
                video_row_ids.add(row["row_id"])
                frame_ranges[row["relative_path"]].append(int(row["frame_index_0based"]))
    duplicate_row_ids = counts["video_frame_index"] - len(video_row_ids)
    result = {
        "still_image_rows": counts["still_image"],
        "video_frame_index_rows": counts["video_frame_index"],
        "unique_video_paths": len(video_paths_seen),
        "unique_video_row_ids": len(video_row_ids),
        "duplicate_video_row_ids": duplicate_row_ids,
        "video_frame_ranges_contiguous": all(
            indexes == list(range(len(indexes))) for indexes in frame_ranges.values()
        ),
    }
    expected = (EXPECTED_IMAGE_ROWS, EXPECTED_VIDEO_FRAME_ROWS, EXPECTED_VIDEO_CLIPS, 0, True)
    actual = (result["still_image_rows"], result["video_frame_index_rows"], result["unique_video_paths"], result["duplicate_video_row_ids"], result["video_frame_ranges_contiguous"])
    if actual != expected:
        raise RuntimeError(f"Master manifest coverage assertion failed: expected {expected}, got {actual}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory (default: results/external_data_analysis)")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    image_rows: list[dict[str, Any]] = []
    image_rows.extend(core_rows())
    image_rows.extend(source_image_rows(
        "deboniz_helmet", DEBONIZ_ROOT, ("train", "val", "test"),
        "source_classes_preserved_visual_semantic_review_recommended",
        "Complete DeBoniz source split and annotation; source classes Helmet/No-Helmet are kept distinct from the project's current classes pending visual semantic review.",
    ))
    ppe_roots = sorted(PPE_DATA_ROOT.rglob("train/images"))
    if len(ppe_roots) != 1:
        raise RuntimeError(f"Expected exactly one Mendeley PPE train/images directory, found {len(ppe_roots)}")
    ppe_root = ppe_roots[0].parent.parent
    image_rows.extend(source_image_rows(
        "mendeley_ppe", ppe_root, ("train", "valid"),
        "source_class_ids_preserved; Helmet/NoHelmet subset mapped; PPE coverage review required",
        "Mendeley PPE source image and all four original PPE classes are inventoried; classes 2/3 describe vest rather than helmet status.",
    ))

    if len(image_rows) != EXPECTED_IMAGE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_IMAGE_ROWS:,} still images, found {len(image_rows):,}")

    # Make ambiguity visible: copied partial extraction is a redundant staging copy,
    # not a fourth source dataset. Record its coverage in the summary below.
    partial_deboniz = ROOT / "ext" / "deboniz_helmet" / "partial_extract_1" / "datasetmodelloscelto" / "dataset10kadjusted"
    partial_images = sum(len(image_files(partial_deboniz / split / "images")) for split in ("train", "val", "test") if (partial_deboniz / split / "images").is_dir())
    partial_sha_audit_path = ROOT / "results" / "external_data_analysis" / "deboniz_partial_extract_sha_audit_summary.json"
    if partial_sha_audit_path.is_file():
        partial_sha_audit = json.loads(partial_sha_audit_path.read_text(encoding="utf-8"))
    else:
        partial_sha_audit = {}
    if partial_sha_audit.get("all_partial_files_exactly_match_canonical"):
        partial_copy_rationale = (
            f"SHA-256 audit verified all {partial_sha_audit['partial_image_count']:,} partial-extract images as exact matches to canonical files with the same split and basename; "
            f"{partial_sha_audit['sha256_mismatch_count']} mismatches and {partial_sha_audit['canonical_basename_missing_count']} missing matches. "
            "The canonical source dataset is inventoried once to avoid weighting staging copies as additional samples."
        )
    else:
        partial_copy_rationale = "partial_extract_1 is a staging extraction and is inventoried separately from the complete canonical DeBoniz dataset."

    duplicate_summary = assign_exact_duplicate_groups(image_rows)
    video_rows, video_summary = make_video_frame_records()
    if video_summary["video_count"] != EXPECTED_VIDEO_CLIPS or video_summary["unique_video_path_count"] != EXPECTED_VIDEO_CLIPS:
        raise RuntimeError(f"Expected {EXPECTED_VIDEO_CLIPS} unique video clips, found {video_summary['video_count']} clips / {video_summary['unique_video_path_count']} unique paths")
    if video_summary["frame_index_rows"] != EXPECTED_VIDEO_FRAME_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_VIDEO_FRAME_ROWS:,} video frame rows, found {video_summary['frame_index_rows']:,}")
    csv_path = output_dir / "master_data_manifest.csv"
    total_rows = write_csv(csv_path, image_rows, video_rows)
    validation = verify_manifest_csv(csv_path)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_path": rel(csv_path) if csv_path.resolve().is_relative_to(ROOT.resolve()) else str(csv_path.resolve()),
        "manifest_rows_total": total_rows,
        "images": count_images(image_rows),
        "exact_duplicate_analysis": duplicate_summary,
        "videos": video_summary,
        "coverage_assertions": validation,
        "class_maps": {dataset: {str(cid): name for cid, name in classes.items()} for dataset, classes in CLASS_MAPS.items()},
        "excluded_redundant_staging_copy": {
            "path": rel(partial_deboniz), "image_file_count": partial_images,
            "rationale": partial_copy_rationale,
            "sha256_audit_summary_path": rel(partial_sha_audit_path) if partial_sha_audit_path.is_file() else "",
        },
        "method_notes": [
            "No source images, videos, or labels are changed; CSV contains source-relative paths.",
            "Every canonical still-image dataset record has a row; each video contributes one row per zero-based frame index.",
            "Video frame rows represent indexes/timestamps; no frame image is extracted and clip-level labels are not treated as bounding-box annotations.",
            "Exact duplicates are grouped by SHA-256. Group-aware repartitioning should happen before creating new train/validation/test splits.",
            "Mendeley PPE images without Helmet/NoHelmet boxes are flagged for label-completeness review, not assumed to be helmet-free negatives.",
            "DeBoniz polygon-like and malformed rows remain represented and require conversion/review before YOLO detection training.",
        ],
    }
    summary_path = output_dir / "master_data_manifest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path.resolve()), "summary": str(summary_path.resolve()), "rows": total_rows, "images": len(image_rows), "videos": video_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
