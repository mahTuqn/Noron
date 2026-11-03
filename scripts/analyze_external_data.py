"""Inventory external helmet/PPE datasets without changing the original split."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "ext"
OUT = ROOT / "results" / "external_data_analysis"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CORE_ROOT = ROOT / "data" / "processed_combined"


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def inspect_split(source: str, split: str, image_dir: Path, label_dir: Path, names: dict[int, str]) -> dict:
    images = image_files(image_dir)
    class_boxes: Counter[str] = Counter()
    total_boxes = missing = empty = malformed = invalid_geometry = 0
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            missing += 1
            continue
        content = label.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            empty += 1
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) != 5:
                malformed += 1
                continue
            try:
                class_id = int(float(parts[0]))
                x, y, width, height = map(float, parts[1:])
            except (ValueError, OverflowError):
                malformed += 1
                continue
            total_boxes += 1
            class_boxes[names.get(class_id, f"class_{class_id}")] += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                invalid_geometry += 1
    return {
        "source": source,
        "split": split,
        "image_count": len(images),
        "box_count": total_boxes,
        "boxes_by_class": dict(class_boxes),
        "missing_label_files": missing,
        "empty_label_files": empty,
        "malformed_rows": malformed,
        "invalid_normalized_boxes": invalid_geometry,
        "image_directory": str(image_dir.relative_to(ROOT)).replace("\\", "/"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    deb = EXT / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted"
    for split in ("train", "val", "test"):
        rows.append(inspect_split("DeBoniz Helmet Detection", split, deb / split / "images", deb / split / "labels", {0: "Helmet", 1: "No-Helmet"}))

    ppe = EXT / "mendeley_ppe" / "extracted" / "Dataset of Personal Protective Equipment (PPE)" / "20250731-PPE2286y" / "20250731-ppe2286y"
    for split in ("train", "valid"):
        rows.append(inspect_split("Mendeley PPE", split, ppe / split / "images", ppe / split / "labels", {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}))

    with (OUT / "image_split_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["source", "split", "image_count", "box_count", "boxes_by_class", "missing_label_files", "empty_label_files", "malformed_rows", "invalid_normalized_boxes", "image_directory"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "boxes_by_class": json.dumps(row["boxes_by_class"], ensure_ascii=False)})

    # Exact file-level SHA-256 duplicate check against the project's fixed benchmark.
    hash_groups: dict[str, list[str]] = defaultdict(list)
    image_roots = [("core", CORE_ROOT / "images")]
    for slug in ("deboniz_helmet", "mendeley_ppe"):
        image_roots.append((slug, EXT / slug / "extracted"))
    image_total = 0
    for source, root in image_roots:
        if source == "core":
            files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        else:
            files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "__MACOSX" not in str(p)]
        for idx, image in enumerate(files, 1):
            image_total += 1
            hash_groups[sha256(image)].append(f"{source}:{image.relative_to(ROOT).as_posix()}")
            if idx % 5000 == 0:
                print(f"Hashed {source}: {idx}/{len(files)} images", flush=True)

    exact_duplicate_groups = [paths for paths in hash_groups.values() if len(paths) > 1]
    duplicate_source_pairs: Counter[str] = Counter()
    core_external_groups = 0
    within_source_groups: Counter[str] = Counter()
    cross_split_groups: Counter[str] = Counter()
    core_test_cross_split_rows: list[dict[str, str]] = []
    for group in exact_duplicate_groups:
        owners: dict[str, set[str]] = defaultdict(set)
        for item in group:
            source, relative = item.split(":", 1)
            split = next((part for part in Path(relative).parts if part in {"train", "val", "valid", "test"}), "unspecified")
            owners[source].add(split)
        source_names = sorted(owners)
        if "core" in owners and any(source != "core" for source in owners):
            core_external_groups += 1
            for source in source_names:
                if source != "core":
                    duplicate_source_pairs[f"core <-> {source}"] += 1
        if len(source_names) == 1:
            within_source_groups[source_names[0]] += 1
            if len(owners[source_names[0]]) > 1:
                cross_split_groups[source_names[0]] += 1
        if "core" in owners and "test" in owners["core"] and owners["core"] & {"train", "val"}:
            core_test_cross_split_rows.append({
                "duplicate_group": str(len(core_test_cross_split_rows) + 1),
                "core_splits": ",".join(sorted(owners["core"])),
                "files": " | ".join(item.split(":", 1)[1] for item in group if item.startswith("core:")),
            })
        elif "core" not in owners:
            for i, first in enumerate(source_names):
                for second in source_names[i + 1:]:
                    duplicate_source_pairs[f"{first} <-> {second}"] += 1
    duplicate_report = {
        "method": "SHA-256 of original image file bytes; detects exact file duplicates, not resized/recompressed near-duplicates",
        "images_hashed": image_total,
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "exact_duplicate_images_in_groups": sum(len(paths) for paths in exact_duplicate_groups),
        "groups_with_core_and_external_images": core_external_groups,
        "groups_by_source_pair": dict(duplicate_source_pairs),
        "duplicate_groups_within_single_source": dict(within_source_groups),
        "within_source_groups_spanning_splits": dict(cross_split_groups),
        "core_test_cross_split_groups": len(core_test_cross_split_rows),
        "groups": exact_duplicate_groups[:200],
    }
    with (OUT / "core_test_exact_duplicates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["duplicate_group", "core_splits", "files"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(core_test_cross_split_rows)

    video_root = EXT / "mendeley_motorcycle_video" / "extracted" / "Dataset for the Detection of Helmet Use in Motorcycle Drivers"
    video_rows = []
    for video in sorted(video_root.rglob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        video_rows.append({
            "filename": video.name,
            "clip_label": video.parent.name,
            "relative_path": video.relative_to(ROOT).as_posix(),
            "frames": frames,
            "fps": fps,
            "duration_seconds": round(frames / fps, 3) if fps > 0 else None,
            "width": width,
            "height": height,
            "size_bytes": video.stat().st_size,
        })
    with (OUT / "video_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["filename", "clip_label", "relative_path", "frames", "fps", "duration_seconds", "width", "height", "size_bytes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(video_rows)

    source_meta = {}
    for slug in ("deboniz_helmet", "mendeley_ppe", "mendeley_motorcycle_video"):
        source_meta[slug] = json.loads((EXT / slug / "source.json").read_text(encoding="utf-8"))
    manual_review_path = OUT / "manual_review_cases.csv"
    manual_reviews = list(csv.DictReader(manual_review_path.open(encoding="utf-8-sig"))) if manual_review_path.exists() else []
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_benchmark_modified": False,
        "image_splits": rows,
        "image_totals": {
            source: sum(r["image_count"] for r in rows if r["source"] == source)
            for source in sorted({r["source"] for r in rows})
        },
        "duplicate_check": duplicate_report,
        "video_summary": {
            "clip_count": len(video_rows),
            "clip_labels": dict(Counter(row["clip_label"] for row in video_rows)),
            "total_duration_seconds": round(sum(row["duration_seconds"] or 0 for row in video_rows), 3),
            "total_size_bytes": sum(row["size_bytes"] for row in video_rows),
            "mean_fps": round(sum(row["fps"] for row in video_rows) / len(video_rows), 3) if video_rows else None,
            "resolutions": dict(Counter(f"{row['width']}x{row['height']}" for row in video_rows)),
            "label_scope": "Clip-level folder label only; no frame-level detection boxes are supplied.",
        },
        "sources": source_meta,
        "manual_review_cases": {
            "file": str(manual_review_path.relative_to(ROOT)).replace("\\", "/"),
            "count": len(manual_reviews),
            "cases": [{"case_id": row["case_id"], "assessment": row["assessment"]} for row in manual_reviews],
        },
        "notes": [
            "External datasets are kept separate from data/processed_combined; metrics must not be mixed with the original held-out test.",
            "Exact SHA-256 duplicate checking does not identify resized/recompressed near-duplicates.",
            "DeBoniz card lists 18,159 images; extracted image files are counted directly in the inventory, so any difference is documented rather than silently corrected.",
            "DeBoniz train has 44 annotation rows that do not match the five-field YOLO detection format; they look polygon-like and are excluded from valid-box counts.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"image_totals": summary["image_totals"], "duplicate_check": {k: duplicate_report[k] for k in ("images_hashed", "exact_duplicate_groups", "exact_duplicate_images_in_groups")}, "video_summary": summary["video_summary"]}, ensure_ascii=False, indent=2), flush=True)
    print(f"Inventory saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
