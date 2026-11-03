"""Create a DeBoniz-only YAML test view from the grouped retraining manifest.

This writes an image-list, standalone evaluation YAML and audit JSON only. It
does not modify source labels, train/validation/test images or run inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "retraining_all_stills"
CLASS_NAMES = {0: "wearing_helmet", 1: "not_wearing_helmet"}
SOURCE_DATASET = "DeBoniz Helmet Detection"
REQUIRED_COLUMNS = {
    "source_dataset", "source_split", "generated_split", "eligibility_status",
    "generated_image_path", "generated_label_path", "sha256", "split_group_id",
    "target_boxes_by_class", "clip_group_id",
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"Artifact path must stay inside the project: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/retraining_all_stills/manifest.csv")
    parser.add_argument("--list-file", default="data/retraining_all_stills/images/deboniz_heldout_test.txt")
    parser.add_argument("--config", default="configs/data_retrain_deboniz_heldout.yaml")
    parser.add_argument("--audit-report", default="results/deboniz_heldout_eval_audit.json")
    args = parser.parse_args()

    manifest_path = project_path(args.manifest)
    list_path = project_path(args.list_file)
    config_path = project_path(args.config)
    audit_path = project_path(args.audit_report)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Retraining manifest not found: {manifest_path}")
    if list_path.parent != (DATASET_ROOT / "images").resolve():
        raise ValueError("The YOLO image-list must be placed directly under data/retraining_all_stills/images")
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {DATASET_ROOT}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not REQUIRED_COLUMNS.issubset(fields):
            raise ValueError(f"Manifest lacks required columns: {sorted(REQUIRED_COLUMNS - fields)}")
        all_rows = list(reader)

    selected = [
        row for row in all_rows
        if row["source_dataset"] == SOURCE_DATASET
        and row["generated_split"] == "test"
        and row["eligibility_status"] == "eligible"
    ]
    if not selected:
        raise ValueError("No eligible DeBoniz rows are assigned to generated_split=test")
    selected.sort(key=lambda row: row["generated_image_path"])

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        if row["eligibility_status"] == "eligible" and row["split_group_id"]:
            group_splits[row["split_group_id"]].add(row["generated_split"])

    image_test_dir = (DATASET_ROOT / "images" / "test").resolve()
    label_test_dir = (DATASET_ROOT / "labels" / "test").resolve()
    source_splits: Counter[str] = Counter()
    target_boxes: Counter[str] = Counter()
    list_entries: list[str] = []
    subset_digest = hashlib.sha256()
    selected_groups: set[str] = set()
    selected_hashes: set[str] = set()
    mp4_266_count = 0

    for row in selected:
        image = project_path(row["generated_image_path"])
        label = project_path(row["generated_label_path"])
        if not image.is_file():
            raise FileNotFoundError(f"Generated DeBoniz test image is missing: {image}")
        if not label.is_file():
            raise FileNotFoundError(f"Generated DeBoniz test label is missing: {label}")
        if image.parent.resolve() != image_test_dir or label.parent.resolve() != label_test_dir:
            raise ValueError(f"Selected row is outside the generated held-out test split: {image}")
        if not row["split_group_id"] or group_splits[row["split_group_id"]] != {"test"}:
            raise ValueError(f"Selected row's grouped split is not exclusively test: {row['split_group_id']}")
        if len(row["sha256"]) != 64:
            raise ValueError(f"Invalid source SHA-256 in manifest for {row['source_image_path']}")

        source_splits[row["source_split"]] += 1
        selected_groups.add(row["split_group_id"])
        selected_hashes.add(row["sha256"])
        if row["clip_group_id"] == "deboniz_clip:mp4-266":
            mp4_266_count += 1
        for class_name, count in json.loads(row["target_boxes_by_class"]).items():
            if class_name not in CLASS_NAMES.values():
                raise ValueError(f"Unexpected binary target class {class_name!r} in manifest")
            target_boxes[class_name] += int(count)

        entry = f"./test/{image.name}"
        list_entries.append(entry)
        subset_digest.update(f"{row['generated_image_path']}\t{row['sha256']}\n".encode("utf-8"))

    if len(list_entries) != len(set(list_entries)):
        raise ValueError("Image-list entries are not unique")
    if mp4_266_count != 2:
        raise ValueError(f"Expected both grouped mp4-266 hard negatives in the held-out list; found {mp4_266_count}")

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_bytes = ("\n".join(list_entries) + "\n").encode("utf-8")
    list_path.write_bytes(list_bytes)

    config = {
        "path": Path(os.path.relpath(DATASET_ROOT, config_path.parent)).as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/deboniz_heldout_test.txt",
        "names": CLASS_NAMES,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Statically resolve the same ./ paths Ultralytics' image-list loader uses.
    resolved_entries = [
        (list_path.parent / entry[2:]).resolve() if entry.startswith("./") else Path(entry).resolve()
        for entry in list_path.read_text(encoding="utf-8").splitlines()
    ]
    if resolved_entries != [project_path(row["generated_image_path"]) for row in selected]:
        raise AssertionError("Image-list resolution does not match the selected manifest rows")
    if any(not image.is_file() for image in resolved_entries):
        raise FileNotFoundError("One or more image-list entries do not resolve to existing images")

    loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data_root = (config_path.parent / loaded_config["path"]).resolve()
    test_list = (config_data_root / loaded_config["test"]).resolve()
    if test_list != list_path or loaded_config["names"] != CLASS_NAMES:
        raise AssertionError("Standalone YAML paths or class names do not match the generated artifacts")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "filter": {
            "source_dataset": SOURCE_DATASET,
            "eligibility_status": "eligible",
            "generated_split": "test",
            "original_source_split_recorded": True,
        },
        "selected_image_count": len(selected),
        "selected_counts_by_original_source_split": dict(sorted(source_splits.items())),
        "target_boxes_by_class": dict(sorted(target_boxes.items())),
        "selected_group_count": len(selected_groups),
        "selected_unique_sha256_count": len(selected_hashes),
        "user_reviewed_mp4_266_hard_negative_count": mp4_266_count,
        "group_leakage_out_of_test": 0,
        "source_image_list": list_path.relative_to(ROOT).as_posix(),
        "source_image_list_sha256": sha256_bytes(list_bytes),
        "selected_subset_sha256": subset_digest.hexdigest(),
        "yaml_config": config_path.relative_to(ROOT).as_posix(),
        "class_names": CLASS_NAMES,
        "generated_image_and_label_existence_checked": True,
        "yaml_and_list_paths_statically_resolved": True,
        "inference_run": False,
        "labels_or_splits_modified": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_images": len(selected),
        "source_split_counts": dict(sorted(source_splits.items())),
        "target_boxes_by_class": dict(sorted(target_boxes.items())),
        "image_list": str(list_path),
        "yaml": str(config_path),
        "audit": str(audit_path),
        "inference_run": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
