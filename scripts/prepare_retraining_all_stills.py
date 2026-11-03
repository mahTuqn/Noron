"""Build an auditable, leakage-aware binary helmet dataset from all still images.

This preparation step never starts training. Every source image is recorded in
manifest.csv. Mendeley PPE images with no Helmet/NoHelmet boxes are held in
pending_review and are deliberately excluded from YOLO splits until reviewed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "wearing_helmet", 1: "not_wearing_helmet"}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXPECTED_COUNTS = {"core": 12_580, "deboniz": 18_157, "mendeley_ppe": 2_286}
MENDEDLEY_CLASS_NAMES = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}


def inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"Path must be a child of the project root: {resolved}")
    return resolved


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "__macosx" not in str(p).lower()
    )


def parse_box_line(line: str, source: str, path: Path, line_number: int):
    """Return source class, binary-target class, normalized xywh, polygon flag."""
    parts = line.split()
    if not parts:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Bad annotation at {path}:{line_number}: {line}") from exc

    polygon = source == "deboniz" and len(coords) > 4
    if len(coords) == 4:
        x, y, width, height = coords
    elif polygon and len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[0::2], coords[1::2]
        if any(value < 0 or value > 1 for value in coords):
            raise ValueError(f"Polygon coordinate outside [0,1] at {path}:{line_number}")
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
        width, height = xmax - xmin, ymax - ymin
    else:
        raise ValueError(f"Unsupported annotation row at {path}:{line_number}: {line}")

    if source == "mendeley_ppe":
        if class_id not in MENDEDLEY_CLASS_NAMES:
            raise ValueError(f"Unexpected Mendeley class {class_id} at {path}:{line_number}")
        target_class = class_id if class_id in (0, 1) else None
    else:
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unexpected binary class {class_id} at {path}:{line_number}")
        target_class = class_id

    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError(f"Invalid normalized box at {path}:{line_number}: {line}")
    original_box = (x, y, width, height)
    xmin, xmax = x - width / 2, x + width / 2
    ymin, ymax = y - height / 2, y + height / 2
    overflow = max(0.0, -xmin, xmax - 1.0, -ymin, ymax - 1.0)
    if overflow > 0.0001:
        raise ValueError(f"Box extends {overflow:.8f} outside normalized image at {path}:{line_number}: {line}")
    clamped = overflow > 0
    if clamped:
        xmin, xmax = max(0.0, xmin), min(1.0, xmax)
        ymin, ymax = max(0.0, ymin), min(1.0, ymax)
        x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
        width, height = xmax - xmin, ymax - ymin
        if width <= 0 or height <= 0:
            raise ValueError(f"Clamping collapsed box at {path}:{line_number}: {line}")
    return class_id, target_class, (x, y, width, height), polygon, clamped, overflow, original_box


def load_labels(label_path: Path | None, source: str):
    if label_path is None or not label_path.is_file():
        return [], Counter(), Counter(), 0, 0, [], "missing_source_label_file"
    lines = label_path.read_text(encoding="utf-8", errors="strict").splitlines()
    parsed = []
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    polygon_rows = 0
    clamped_rows = 0
    clamp_records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        item = parse_box_line(line, source, label_path, line_number)
        if item is None:
            continue
        class_id, target_class, box, polygon, clamped, overflow, original_box = item
        source_name = MENDEDLEY_CLASS_NAMES[class_id] if source == "mendeley_ppe" else CLASS_NAMES[class_id]
        source_counts[source_name] += 1
        if target_class is not None:
            target_counts[CLASS_NAMES[target_class]] += 1
            x, y, width, height = box
            parsed.append(f"{target_class} {x:.8f} {y:.8f} {width:.8f} {height:.8f}")
        if polygon:
            polygon_rows += 1
        if clamped:
            clamped_rows += 1
            clamp_records.append({
                "source_label_path": rel(label_path),
                "line_number": line_number,
                "source_class_id": class_id,
                "mapped_target_class": "" if target_class is None else CLASS_NAMES[target_class],
                "overflow_fraction": f"{overflow:.10f}",
                "original_xywh": json.dumps([round(value, 10) for value in original_box]),
                "clamped_xywh": json.dumps([round(value, 10) for value in box]),
            })
    reason = "empty_source_label_file" if not parsed and not source_counts else ""
    return parsed, source_counts, target_counts, polygon_rows, clamped_rows, clamp_records, reason


def seeded_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def normalize_repo_path(value: str) -> str:
    text = value.replace("\\", "/")
    root = ROOT.as_posix().rstrip("/")
    if text.lower().startswith(root.lower() + "/"):
        text = text[len(root) + 1:]
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def load_hash_manifest(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Master image hash manifest not found: {path}")
    result: dict[str, dict[str, str]] = {}
    video_clip_paths: set[str] = set()
    video_frame_index_rows = 0
    still_image_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"relative_path", "sha256"}.issubset(reader.fieldnames):
            raise ValueError(f"Hash manifest needs relative_path and sha256 columns: {path}")
        for row in reader:
            record_type = (row.get("record_type") or "").strip().lower()
            relpath = normalize_repo_path(row["relative_path"])
            if record_type == "video_frame_index":
                video_frame_index_rows += 1
                if relpath:
                    video_clip_paths.add(relpath)
                continue
            if "video" in record_type:
                continue
            digest = (row.get("sha256") or "").strip().lower()
            if relpath and digest:
                result[relpath] = row
                still_image_rows += 1
    if not result:
        raise ValueError(f"No still-image hashes found in {path}")
    stats = {
        "still_image_rows": still_image_rows,
        "video_frame_index_rows": video_frame_index_rows,
        "video_clip_count": len(video_clip_paths),
        "video_clip_paths": sorted(video_clip_paths),
    }
    return result, stats


def deboniz_clip_id(image_name: str) -> str:
    match = re.search(r"(?:^|_)(mp4[-_]\d+)(?=_|\.|$)", image_name, flags=re.IGNORECASE)
    if not match:
        return ""
    return "deboniz_clip:" + match.group(1).lower().replace("_", "-")


def assign_component_ids(records: list[dict]) -> None:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(first: str, second: str) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            low, high = sorted((root_a, root_b))
            parent[high] = low

    record_tags: dict[int, list[str]] = {}
    for record in records:
        tags = ["sha256:" + record["sha256"]]
        if record["clip_group_id"]:
            tags.append(record["clip_group_id"])
        record_tags[id(record)] = tags
        for tag in tags:
            find(tag)
        for tag in tags[1:]:
            union(tags[0], tag)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for tag in list(parent):
        members_by_root[find(tag)].append(tag)
    stable_ids = {
        root: "group:" + hashlib.sha256("|".join(sorted(tags)).encode("utf-8")).hexdigest()[:16]
        for root, tags in members_by_root.items()
    }
    for record in records:
        root = find(record_tags[id(record)][0])
        record["split_group_id"] = stable_ids[root]
        record["group_id"] = stable_ids[root]


def assign_group_splits(records: list[dict], seed: int) -> dict[str, str]:
    """Assign exact-hash and video-clip connected groups together."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["eligibility_status"] == "eligible":
            groups[record["split_group_id"]].append(record)

    by_signature: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for digest, members in groups.items():
        labels = set()
        for member in members:
            labels.update(member["target_classes"])
        signature = "+".join(sorted(labels)) or "none"
        by_signature[signature].append((digest, members))

    assignments: dict[str, str] = {}
    split_order = ["train", "val", "test"]
    for signature, signature_groups in sorted(by_signature.items()):
        total_images = sum(len(members) for _, members in signature_groups)
        targets = {split: total_images * RATIOS[split] for split in split_order}
        current = Counter()
        pinned = [(group_id, members) for group_id, members in signature_groups
                  if any(member["clip_group_id"] == "deboniz_clip:mp4-266" for member in members)]
        for group_id, members in pinned:
            assignments[group_id] = "test"
            current["test"] += len(members)
        signature_groups = [(group_id, members) for group_id, members in signature_groups if group_id not in assignments]
        # Place larger duplicate groups first; the seeded hash breaks size ties reproducibly.
        signature_groups.sort(key=lambda item: (-len(item[1]), seeded_key(seed, f"{signature}:{item[0]}")))
        for digest, members in signature_groups:
            weight = len(members)
            def cost(split: str) -> tuple[float, int]:
                after = current[split] + weight
                delta = (after - targets[split]) ** 2 - (current[split] - targets[split]) ** 2
                return delta / max(targets[split], 1.0), split_order.index(split)
            chosen = min(split_order, key=cost)
            assignments[digest] = chosen
            current[chosen] += weight
    return assignments


def target_name(record: dict) -> str:
    short_path = hashlib.sha1(record["source_image_path"].encode("utf-8")).hexdigest()[:12]
    image_path = Path(record["source_image_path"])
    return f"{record['source_key']}__{short_path}__{image_path.name}"


def build_source_rows(master_by_path: dict[str, dict[str, str]]) -> tuple[list[dict], list[dict]]:
    core = ROOT / "data" / "processed_combined"
    deboniz = ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted"
    mendeley = (ROOT / "ext" / "mendeley_ppe" / "extracted" / "Dataset of Personal Protective Equipment (PPE)"
                / "20250731-PPE2286y" / "20250731-ppe2286y")
    sources = []
    for split in ("train", "val", "test"):
        sources.append(("core", "core", split, core / "images" / split, core / "labels" / split))
        sources.append(("deboniz", "deboniz", split,
                        deboniz / split / "images", deboniz / split / "labels"))
    for split in ("train", "valid"):
        sources.append(("mendeley_ppe", "mendeley_ppe", split,
                        mendeley / split / "images", mendeley / split / "labels"))

    rows = []
    coordinate_clamps = []
    actual_counts = Counter()
    audited_images = 0
    target_id_maps = {
        "core": {0: 0, 1: 1},
        "deboniz": {0: 0, 1: 1},
        "mendeley_ppe": {0: 0, 1: 1, 2: None, 3: None},
    }
    source_names = {
        "core": {0: "wearing_helmet", 1: "not_wearing_helmet"},
        "deboniz": {0: "Helmet", 1: "No-Helmet"},
        "mendeley_ppe": MENDEDLEY_CLASS_NAMES,
    }
    for source_key, source, source_split, image_dir, label_dir in sources:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Source image directory not found: {image_dir}")
        paths = image_paths(image_dir)
        for image in paths:
            relative_image_path = rel(image)
            master = master_by_path.get(relative_image_path)
            if master is None:
                raise KeyError(f"Image missing from master hash manifest: {relative_image_path}")
            if master.get("record_type") != "still_image":
                raise ValueError(f"Expected still_image master row for {relative_image_path}")
            if master.get("label_path"):
                label = ROOT / Path(normalize_repo_path(master["label_path"]))
            else:
                label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing source label for {relative_image_path}: {label}")

            source_count_ids = json.loads(master.get("label_class_counts_json") or "{}")
            source_counts = Counter({source_names[source][int(class_id)]: int(count)
                                     for class_id, count in source_count_ids.items()})
            target_counts = Counter()
            for class_id, count in source_count_ids.items():
                target_id = target_id_maps[source].get(int(class_id))
                if target_id is not None:
                    target_counts[CLASS_NAMES[target_id]] += int(count)
            polygon_rows = int(master.get("polygon_like_rows") or 0)
            source_label_count = int(master.get("parsed_annotation_rows") or sum(source_counts.values()))
            label_status = master.get("label_file_status", "")
            is_mendeley_unlabeled = source == "mendeley_ppe" and int(master.get("helmet_or_nohelmet_box_count") or 0) == 0
            user_00751 = source == "mendeley_ppe" and image.stem.startswith("00751")
            if label_status != "present_nonempty":
                status = "pending_unlabeled_source"
                reason = f"Master inventory label status: {label_status or 'unknown'}"
            elif is_mendeley_unlabeled:
                status = "pending_manual_label" if user_00751 else "pending_missing_helmet_labels"
                reason = "No Helmet or NoHelmet boxes; vest-only annotations are not safe binary negatives. Review and complete labels before use."
            else:
                status = "eligible"
                reason = ""

            parsed = None
            clamped_rows = 0
            if source == "mendeley_ppe" or (source == "deboniz" and polygon_rows > 0):
                parsed, parsed_source_counts, parsed_target_counts, parsed_polygons, clamped_rows, image_clamps, label_reason = load_labels(label, source)
                if label_reason:
                    raise ValueError(f"Master inventory marks {relative_image_path} as labeled but parsing failed: {label_reason}")
                if source == "mendeley_ppe" and status == "eligible" and len(parsed) != sum(target_counts.values()):
                    raise ValueError(f"Mendeley target box count disagrees with master inventory for {relative_image_path}")
                if source == "deboniz" and status == "eligible" and (parsed_polygons != polygon_rows or len(parsed) != sum(target_counts.values())):
                    raise ValueError(f"DeBoniz polygon/target counts disagree with master inventory for {relative_image_path}")
                coordinate_clamps.extend({
                    **clamp,
                    "source_image_path": relative_image_path,
                    "retained_in_binary_target": str(status == "eligible" and bool(clamp["mapped_target_class"])).lower(),
                } for clamp in image_clamps)
                if status != "eligible":
                    parsed = None
            clip_id = deboniz_clip_id(image.name) if source == "deboniz" else ""
            user_note = ""
            if user_00751:
                user_note = "User correction: visible red safety helmet; source label contains Vest boxes only. Preserve source label unchanged and manually add the full helmet box before supervised use."
            elif source == "deboniz" and clip_id == "deboniz_clip:mp4-266":
                user_note = "User correction: ordinary baseball cap is not a helmet; preserve No-Helmet label and keep this clip group held out in test for hard-negative review."

            row = {
                "source_key": source_key,
                "source_dataset": {"core": "Core SHWD + Kaggle combined", "deboniz": "DeBoniz Helmet Detection", "mendeley_ppe": "Mendeley PPE"}[source_key],
                "source_split": source_split,
                "source_image_path": relative_image_path,
                "source_label_path": rel(label),
                "sha256": master["sha256"].lower(),
                "source_width": "",
                "source_height": "",
                "source_size_bytes": image.stat().st_size,
                "source_label_box_count": source_label_count,
                "source_boxes_by_class": json.dumps(dict(source_counts), ensure_ascii=False, sort_keys=True),
                "target_box_count": sum(target_counts.values()),
                "target_boxes_by_class": json.dumps(dict(target_counts), ensure_ascii=False, sort_keys=True),
                "polygon_rows_converted": polygon_rows,
                "coordinate_boxes_clamped": clamped_rows,
                "eligibility_status": status,
                "pending_reason": reason,
                "user_review_note": user_note,
                "clip_group_id": clip_id,
                "source_label_class_ids": "0:wearing_helmet;1:not_wearing_helmet" if source != "mendeley_ppe" else "0:Helmet;1:NoHelmet;2:NoVest;3:Vest",
                "target_classes": sorted(target_counts),
                "target_label_lines": parsed,
                "generated_split": "",
                "generated_image_path": "",
                "generated_label_path": "",
                "group_id": "",
                "split_group_id": "",
            }
            rows.append(row)
            actual_counts[source_key] += 1
            audited_images += 1
            if audited_images % 2500 == 0:
                print(f"Audited labels and paths for {audited_images:,}/33,023 images", flush=True)
    for source_key, expected in EXPECTED_COUNTS.items():
        if actual_counts[source_key] != expected:
            raise ValueError(f"Expected {expected:,} images for {source_key}, found {actual_counts[source_key]:,}")
    assign_component_ids(rows)
    return rows, coordinate_clamps


def write_report(output: Path, rows: list[dict], split_counts: Counter, assignments: dict[str, str], seed: int,
                 master_manifest: Path, inventory_stats: dict[str, object], coordinate_clamps: list[dict]):
    source_counts = Counter(row["source_key"] for row in rows)
    eligible_by_source = Counter(row["source_key"] for row in rows if row["eligibility_status"] == "eligible")
    pending_by_source = Counter(row["source_key"] for row in rows if row["eligibility_status"] != "eligible")
    pending_reasons = Counter(row["eligibility_status"] for row in rows if row["eligibility_status"] != "eligible")
    source_boxes: dict[str, Counter] = defaultdict(Counter)
    target_boxes: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        source_boxes[row["source_key"]].update(json.loads(row["source_boxes_by_class"]))
        target_boxes[row["source_key"]].update(json.loads(row["target_boxes_by_class"]))
    all_hashes: dict[str, list[dict]] = defaultdict(list)
    eligible_hashes: dict[str, set[str]] = defaultdict(set)
    clip_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        all_hashes[row["sha256"]].append(row)
        if row["eligibility_status"] == "eligible":
            eligible_hashes[row["sha256"]].add(row["generated_split"])
            if row["clip_group_id"]:
                clip_splits[row["clip_group_id"]].add(row["generated_split"])
    duplicate_groups = [members for members in all_hashes.values() if len(members) > 1]
    eligible_duplicate_groups = [members for members in all_hashes.values() if sum(m["eligibility_status"] == "eligible" for m in members) > 1]
    leaking_hashes = [digest for digest, splits in eligible_hashes.items() if len(splits) > 1]
    if leaking_hashes:
        raise AssertionError(f"Exact-hash groups crossed splits: {len(leaking_hashes)}")
    leaking_clips = [clip for clip, splits in clip_splits.items() if len(splits) > 1]
    if leaking_clips:
        raise AssertionError(f"DeBoniz clip groups crossed splits: {len(leaking_clips)}")
    mp4_266_rows = [row for row in rows if row["clip_group_id"] == "deboniz_clip:mp4-266"]
    mp4_266_splits = sorted({row["generated_split"] for row in mp4_266_rows if row["eligibility_status"] == "eligible"})
    if mp4_266_rows and mp4_266_splits != ["test"]:
        raise AssertionError(f"mp4-266 must remain held out in test; got {mp4_266_splits}")

    report = {
        "generated_by": "scripts/prepare_retraining_all_stills.py",
        "seed": seed,
        "split_ratios": RATIOS,
        "source_image_counts": dict(source_counts),
        "total_source_images_manifested": len(rows),
        "master_manifest": rel(master_manifest),
        "master_manifest_still_image_rows": inventory_stats["still_image_rows"],
        "video_diagnostic_inventory": {
            "clip_count": inventory_stats["video_clip_count"],
            "frame_index_rows": inventory_stats["video_frame_index_rows"],
            "master_manifest": rel(master_manifest),
            "supervised_training_use": False,
            "reason": "Video source annotations are clip-level; there are no frame-level detection boxes.",
        },
        "eligible_images": len(rows) - sum(pending_by_source.values()),
        "eligible_by_source": dict(eligible_by_source),
        "pending_by_source": dict(pending_by_source),
        "pending_status_counts": dict(pending_reasons),
        "images_per_training_split": dict(split_counts),
        "target_boxes_by_source": {key: dict(value) for key, value in target_boxes.items()},
        "source_boxes_by_source": {key: dict(value) for key, value in source_boxes.items()},
        "deboniz_polygon_rows_converted_to_enclosing_xywh": sum(row["polygon_rows_converted"] for row in rows),
        "source_annotation_rows_with_tiny_rounding_overflow": len(coordinate_clamps),
        "clamped_rows_retained_in_binary_target": sum(clamp["retained_in_binary_target"] == "true" for clamp in coordinate_clamps),
        "mendeley_vest_boxes_ignored_for_binary_target": source_boxes["mendeley_ppe"]["Vest"] + source_boxes["mendeley_ppe"]["NoVest"],
        "exact_hash_duplicate_groups_all_images": len(duplicate_groups),
        "exact_hash_duplicate_source_rows_in_groups": sum(len(members) for members in duplicate_groups),
        "exact_hash_groups_with_multiple_eligible_images": len(eligible_duplicate_groups),
        "exact_hash_leakage_groups_across_train_val_test": len(leaking_hashes),
        "deboniz_video_clip_groups": len(clip_splits),
        "deboniz_clip_leakage_groups_across_train_val_test": len(leaking_clips),
        "user_reviewed_mp4_266_image_count": len(mp4_266_rows),
        "user_reviewed_mp4_266_split": mp4_266_splits,
        "user_reviewed_mendeley_00751_status": next((row["eligibility_status"] for row in rows if row["source_key"] == "mendeley_ppe" and Path(row["source_image_path"]).stem.startswith("00751")), "not_found"),
        "duplicate_policy": "Keep every eligible source image, but assign exact SHA-256 duplicates and DeBoniz frames sharing an mp4-NNN clip tag to the same split. This prevents group leakage without silently dropping source rows.",
        "pending_policy": "Mendeley images with zero Helmet and NoHelmet boxes are copied to pending_review and excluded from train/val/test. The 00751 user-corrected image is explicitly pending_manual_label. Vest-only source labels are preserved and never treated as binary background.",
        "recommended_training_initialization": "Use stock yolov8m.pt because core images were reallocated; an existing checkpoint may have trained on core images now used for test.",
        "limitations": [
            "Exact SHA-256 grouping catches byte-identical files only; resized or recompressed near-duplicates require a separate perceptual-hash/manual audit.",
            "Mendeley images with at least one Helmet or NoHelmet box are treated as target-annotated; visual completeness still needs human review for omitted objects.",
            "DeBoniz is mixed-domain; source Helmet remains positive, and its metrics should be reported separately from the core worksite domain.",
            "Bounding boxes for DeBoniz polygon annotations are enclosing axis-aligned boxes and may include background compared with the polygon mask.",
            "Core source train/val/test membership is recorded in the manifest but the retraining split is rebuilt by class-presence groups. DeBoniz mp4-NNN frames are kept together; mp4-266 is pinned to test for the user's hard-negative correction.",
        ],
        "training_started": False,
        "output": str(output),
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# All-stills retraining dataset preparation",
        "",
        "This is a preparation artifact only; no model training was run.",
        "",
        f"- Source images inventoried: **{len(rows):,}** (core {source_counts['core']:,}, DeBoniz {source_counts['deboniz']:,}, Mendeley PPE {source_counts['mendeley_ppe']:,}).",
        f"- Master inventory linkage: **{inventory_stats['still_image_rows']:,}** still-image rows, plus **{inventory_stats['video_clip_count']} videos / {inventory_stats['video_frame_index_rows']:,} indexed frames** for full diagnostic inference.",
        f"- Eligible binary-labeled images copied into YOLO splits: **{report['eligible_images']:,}**.",
        f"- Mendeley images held for label review: **{pending_by_source['mendeley_ppe']:,}**; they have no Helmet/NoHelmet target boxes and were not treated as background.",
        f"- User-corrected Mendeley `00751` status: **{report['user_reviewed_mendeley_00751_status']}**; its source label is preserved unchanged.",
        f"- User-corrected DeBoniz `mp4-266` images: **{len(mp4_266_rows)}**, pinned to **test**; No-Helmet labels remain unchanged for hard-negative review.",
        f"- DeBoniz polygon rows converted to enclosing normalized xywh boxes: **{report['deboniz_polygon_rows_converted_to_enclosing_xywh']:,}**.",
        f"- Source annotation rows with tiny normalized edge overrun: **{len(coordinate_clamps):,}**, of which **{report['clamped_rows_retained_in_binary_target']:,}** are retained in binary target labels; see `coordinate_clamps.csv`.",
        f"- Exact-hash duplicate groups: **{len(duplicate_groups):,}**; groups crossing train/val/test: **{len(leaking_hashes)}**. DeBoniz clip groups crossing splits: **{len(leaking_clips)}**.",
        "",
        "## Split counts",
        "",
        "| Split | Eligible images |",
        "|---|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {split_counts[split]:,} |")
    lines += [
        "",
        "## Label handling",
        "",
        "- Binary mapping: source class 0/Helmet → `wearing_helmet`; source class 1/NoHelmet → `not_wearing_helmet`.",
        "- DeBoniz is mixed-domain; preserve source `Helmet` as positive (including motorcycle helmets) and report its held-out metrics separately from the core worksite split.",
        "- Mendeley Vest and NoVest boxes are omitted from binary labels. Images without a helmet-task box go to `pending_review/`, with their original source labels retained there.",
        "- Every source image is listed in `manifest.csv`, including pending rows, and links by source path/SHA-256 to the master inventory. The master inventory retains the 32-video/114,640-frame diagnostic index.",
        "- Source label files are never changed. `coordinate_clamps.csv` lists parsed source annotation rows with tiny rounding overrun, including whether the row is retained as a binary target. Vest/NoVest rows remain excluded and are marked `false`.",
        "- Exact duplicate images remain represented as separate source rows and training copies, but their SHA-256 group is assigned to one split only.",
        "- Use stock `yolov8m.pt` for the first run: core images were reallocated, so the previous checkpoint may have seen images now assigned to test.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += ["", "See `dataset_report.json`, `manifest.csv`, `pending_review.csv`, `duplicate_groups.csv`, and `video_clip_groups.csv` for machine-readable details.", ""]
    (output / "RETRAINING_DATASET_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/retraining_all_stills")
    parser.add_argument("--config", default="configs/data_retrain_all_stills.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-manifest", default="results/external_data_analysis/master_data_manifest.csv",
                        help="Master inventory from image audit; reused to avoid hashing every source image again")
    parser.add_argument("--overwrite", action="store_true", help="Replace only a prior output carrying this script's marker")
    parser.add_argument("--resume", action="store_true", help="Reuse already copied files from a marked prior output and regenerate audit/config files")
    args = parser.parse_args()
    output = inside_project(ROOT / args.output)
    config_path = inside_project(ROOT / args.config)

    if output.exists():
        marker = output / ".generated_by_prepare_retraining_all_stills"
        if not (args.overwrite or args.resume) or not marker.is_file():
            raise SystemExit(f"Output exists and was not safely marked by this builder: {output}. Choose another path or use --overwrite for a marked prior output.")
        if args.overwrite and not args.resume:
            shutil.rmtree(output)
    if config_path.exists() and not (args.overwrite or args.resume):
        raise SystemExit(f"Config already exists: {config_path}. Choose another path or rerun with --overwrite.")

    hash_manifest = inside_project(ROOT / args.hash_manifest)
    master_by_path, inventory_stats = load_hash_manifest(hash_manifest)
    if inventory_stats["still_image_rows"] != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Master inventory has {inventory_stats['still_image_rows']} still-image rows; expected {sum(EXPECTED_COUNTS.values())}")
    if inventory_stats["video_clip_count"] != 32 or inventory_stats["video_frame_index_rows"] != 114_640:
        raise ValueError("Master video inventory must contain 32 clips and 114,640 frame-index rows; got "
                         f"{inventory_stats['video_clip_count']} clips / {inventory_stats['video_frame_index_rows']} frame rows")
    rows, coordinate_clamps = build_source_rows(master_by_path)
    split_assignment = assign_group_splits(rows, args.seed)
    split_counts: Counter[str] = Counter()
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / ".generated_by_prepare_retraining_all_stills"
    if not marker_path.exists():
        marker_path.write_text("generated output; safe for --overwrite\n", encoding="utf-8")
    for split in RATIOS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "pending_review" / "images").mkdir(parents=True, exist_ok=True)
    (output / "pending_review" / "source_labels").mkdir(parents=True, exist_ok=True)

    source_root = ROOT
    for index, row in enumerate(rows, 1):
        source_image = source_root / Path(row["source_image_path"])
        basename = target_name(row)
        if row["eligibility_status"] == "eligible":
            split = split_assignment[row["split_group_id"]]
            row["generated_split"] = split
            image_target = output / "images" / split / basename
            label_target = output / "labels" / split / f"{Path(basename).stem}.txt"
            if not (args.resume and image_target.is_file() and label_target.is_file()):
                shutil.copy2(source_image, image_target)
                if row["target_label_lines"] is None:
                    shutil.copy2(source_root / Path(row["source_label_path"]), label_target)
                else:
                    label_target.write_text("\n".join(row["target_label_lines"]) + "\n", encoding="utf-8")
            row["generated_image_path"] = image_target.relative_to(ROOT).as_posix()
            row["generated_label_path"] = label_target.relative_to(ROOT).as_posix()
            split_counts[split] += 1
        else:
            row["generated_split"] = "pending_review"
            image_target = output / "pending_review" / "images" / basename
            if not (args.resume and image_target.is_file()):
                shutil.copy2(source_image, image_target)
            row["generated_image_path"] = image_target.relative_to(ROOT).as_posix()
            if row["source_label_path"]:
                original_label = source_root / Path(row["source_label_path"])
                label_target = output / "pending_review" / "source_labels" / f"{Path(basename).stem}.txt"
                if not (args.resume and label_target.is_file()):
                    shutil.copy2(original_label, label_target)
                row["generated_label_path"] = label_target.relative_to(ROOT).as_posix()
        del row["target_classes"]
        del row["target_label_lines"]
        if index % 2500 == 0:
            print(f"Prepared {index:,}/{len(rows):,} source images", flush=True)

    manifest_fields = [
        "source_key", "source_dataset", "source_split", "source_image_path", "source_label_path",
        "sha256", "group_id", "split_group_id", "clip_group_id", "source_width", "source_height", "source_size_bytes",
        "source_label_box_count", "source_boxes_by_class", "target_box_count", "target_boxes_by_class",
        "polygon_rows_converted", "coordinate_boxes_clamped", "eligibility_status", "pending_reason", "user_review_note", "source_label_class_ids",
        "generated_split", "generated_image_path", "generated_label_path",
    ]
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(rows)

    pending = [row for row in rows if row["eligibility_status"] != "eligible"]
    pending_fields = [field for field in manifest_fields if field in {"source_key", "source_dataset", "source_split", "source_image_path", "source_label_path", "sha256", "group_id", "clip_group_id", "source_label_box_count", "source_boxes_by_class", "eligibility_status", "pending_reason", "user_review_note", "generated_image_path", "generated_label_path"}]
    with (output / "pending_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pending_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pending)

    with (output / "coordinate_clamps.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["source_image_path", "source_label_path", "line_number", "source_class_id", "mapped_target_class", "retained_in_binary_target", "overflow_fraction", "original_xywh", "clamped_xywh"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coordinate_clamps)

    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        hash_groups[row["sha256"]].append(row)
    with (output / "duplicate_groups.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["group_id", "sha256", "member_count", "member_sources", "member_source_splits", "assigned_eligible_split", "source_paths"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for digest, members in sorted(hash_groups.items()):
            if len(members) < 2:
                continue
            eligible_splits = sorted({member["generated_split"] for member in members if member["eligibility_status"] == "eligible"})
            writer.writerow({
                "group_id": ";".join(sorted({member["group_id"] for member in members})), "sha256": digest, "member_count": len(members),
                "member_sources": ";".join(sorted({member["source_key"] for member in members})),
                "member_source_splits": ";".join(sorted({member["source_split"] for member in members})),
                "assigned_eligible_split": eligible_splits[0] if eligible_splits else "pending_review",
                "source_paths": " | ".join(member["source_image_path"] for member in members),
            })

    video_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["clip_group_id"]:
            video_groups[row["clip_group_id"]].append(row)
    with (output / "video_clip_groups.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["clip_group_id", "member_count", "generated_splits", "eligible_count", "pending_count", "user_review_notes", "source_paths"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for clip_id, members in sorted(video_groups.items()):
            writer.writerow({
                "clip_group_id": clip_id,
                "member_count": len(members),
                "generated_splits": ";".join(sorted({member["generated_split"] for member in members})),
                "eligible_count": sum(member["eligibility_status"] == "eligible" for member in members),
                "pending_count": sum(member["eligibility_status"] != "eligible" for member in members),
                "user_review_notes": " | ".join(sorted({member["user_review_note"] for member in members if member["user_review_note"]})),
                "source_paths": " | ".join(member["source_image_path"] for member in members),
            })

    data_root = output.relative_to(config_path.parent.parent).as_posix()
    # YAML paths are relative to the config file directory; configs/ -> project root.
    config = {
        "path": f"../{data_root}",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_report(output, rows, split_counts, split_assignment, args.seed, hash_manifest, inventory_stats, coordinate_clamps)
    print(json.dumps({
        "source_images": len(rows),
        "eligible_images": sum(split_counts.values()),
        "pending_images": len(pending),
        "images_per_split": dict(split_counts),
        "polygon_rows_converted": sum(row["polygon_rows_converted"] for row in rows),
        "coordinate_boxes_clamped": len(coordinate_clamps),
        "hash_manifest": str(hash_manifest),
        "video_inventory": {key: value for key, value in inventory_stats.items() if key != "video_clip_paths"},
        "output": str(output),
        "config": str(config_path),
        "training_started": False,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
