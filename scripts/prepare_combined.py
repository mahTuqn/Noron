"""Combine prepared SHWD with the public Kaggle hard-hat dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "wearing_helmet", 1: "not_wearing_helmet"}
KAGGLE_CLASS_MAP = {"helmet": 0, "head": 1}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def ensure_project_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Output must stay inside {ROOT}: {resolved}")
    return resolved


def pixel_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def mapped_objects(xml_root: ET.Element, width: int, height: int) -> tuple[list[str], Counter[str], int]:
    lines: list[str] = []
    counts: Counter[str] = Counter()
    ignored = 0
    for obj in xml_root.findall("object"):
        source_name = obj.findtext("name", "").strip().lower()
        if source_name == "person":
            ignored += 1
            continue
        if source_name not in KAGGLE_CLASS_MAP:
            raise ValueError(f"Unexpected Kaggle class {source_name!r}")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        xmin, xmax = max(0.0, min(xmin, width)), max(0.0, min(xmax, width))
        ymin, ymax = max(0.0, min(ymin, height)), max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        class_id = KAGGLE_CLASS_MAP[source_name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        counts[source_name] += 1
    return lines, counts, ignored


def deterministic_split(groups: dict[str, list[tuple]], seed: int):
    assigned = {}
    rng = random.Random(seed)
    for category, rows in sorted(groups.items()):
        rows = list(rows)
        rng.shuffle(rows)
        train_end = int(len(rows) * SPLIT_RATIOS["train"])
        val_end = train_end + int(len(rows) * SPLIT_RATIOS["val"])
        for row in rows[:train_end]:
            assigned[row[0]] = "train"
        for row in rows[train_end:val_end]:
            assigned[row[0]] = "val"
        for row in rows[val_end:]:
            assigned[row[0]] = "test"
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shwd", default="data/processed")
    parser.add_argument("--kaggle-archive", default="data/raw/kaggle_hard_hat_detection.zip")
    parser.add_argument("--output", default="data/processed_combined")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shwd_root = (ROOT / args.shwd).resolve()
    archive_path = (ROOT / args.kaggle_archive).resolve()
    output = ensure_project_path(ROOT / args.output)
    if not archive_path.is_file():
        raise SystemExit(f"Kaggle archive not found: {archive_path}; run scripts/download_kaggle_dataset.py")
    for split in ("train", "val", "test"):
        if not (shwd_root / "images" / split).is_dir() or not (shwd_root / "labels" / split).is_dir():
            raise SystemExit(f"Prepared SHWD split is incomplete: {shwd_root / split}")

    for child in (output / "images", output / "labels"):
        resolved_child = ensure_project_path(child)
        if resolved_child.exists():
            for split_dir in resolved_child.iterdir():
                if split_dir.is_dir() and split_dir.name in {"train", "val", "test"}:
                    for generated_file in split_dir.iterdir():
                        if generated_file.is_file():
                            generated_file.unlink()
        resolved_child.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (resolved_child / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    source_totals: dict[str, Counter[str]] = {"shwd": Counter(), "kaggle": Counter()}
    image_totals: Counter[str] = Counter()
    fingerprint_set: set[str] = set()

    # Preserve the source dataset's official train/validation/test allocation.
    for split in ("train", "val", "test"):
        image_dir, label_dir = shwd_root / "images" / split, shwd_root / "labels" / split
        for image_index, source_image in enumerate(sorted(p for p in image_dir.iterdir() if p.is_file()), 1):
            source_label = label_dir / f"{source_image.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"Missing SHWD label for {source_image}")
            with Image.open(source_image) as image:
                fingerprint_set.add(pixel_digest(image))
            target_name = f"shwd_{source_image.name}"
            shutil.copy2(source_image, output / "images" / split / target_name)
            shutil.copy2(source_label, output / "labels" / split / f"{Path(target_name).stem}.txt")
            class_counts = Counter()
            for line in source_label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    if class_id not in CLASS_NAMES:
                        raise ValueError(f"Unexpected SHWD class id {class_id} in {source_label}")
                    class_counts[CLASS_NAMES[class_id]] += 1
            source_totals["shwd"].update(class_counts)
            image_totals[split] += 1
            manifest_rows.append({
                "image": target_name, "split": split, "source": "SHWD",
                "wearing_helmet": class_counts["wearing_helmet"],
                "not_wearing_helmet": class_counts["not_wearing_helmet"],
                "ignored_person": 0,
            })
            if image_index % 1000 == 0:
                print(f"Copied SHWD {split}: {image_index} images", flush=True)

    kaggle_groups = defaultdict(list)
    person_only_images = 0
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        xml_paths = sorted(n for n in names if n.startswith("annotations/") and n.lower().endswith(".xml"))
        if len(xml_paths) != 5000:
            raise ValueError(f"Expected 5000 Kaggle XML annotations, found {len(xml_paths)}")
        for xml_index, xml_path in enumerate(xml_paths, 1):
            xml_root = ET.fromstring(archive.read(xml_path))
            image_name = Path(xml_root.findtext("filename", "")).name
            image_member = f"images/{image_name}"
            if image_member not in names:
                raise FileNotFoundError(f"Kaggle image missing from archive: {image_member}")
            size = xml_root.find("size")
            if size is None:
                raise ValueError(f"Missing image dimensions in {xml_path}")
            width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions in {xml_path}: {width}x{height}")
            label_lines, class_counts, ignored_person = mapped_objects(xml_root, width, height)
            if not label_lines:
                person_only_images += 1
            else:
                present = tuple(sorted(class_counts))
                category = "+".join(present)
                kaggle_groups[category].append((xml_path, image_member, label_lines, class_counts, ignored_person, width, height))
            if xml_index % 1000 == 0:
                print(f"Audited Kaggle annotations: {xml_index}/{len(xml_paths)}", flush=True)

        assigned_splits = deterministic_split(kaggle_groups, args.seed)
        seen_kaggle: set[str] = set()
        duplicate_count = 0
        all_rows = sorted((row for rows in kaggle_groups.values() for row in rows), key=lambda row: row[0])
        for image_index, row in enumerate(all_rows, 1):
            xml_path, image_member, label_lines, class_counts, ignored_person, width, height = row
            image_bytes = archive.read(image_member)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.size != (width, height):
                    raise ValueError(f"Image size disagrees with annotation for {image_member}")
                fingerprint = pixel_digest(image)
            if fingerprint in fingerprint_set or fingerprint in seen_kaggle:
                duplicate_count += 1
                continue
            seen_kaggle.add(fingerprint)
            split = assigned_splits[xml_path]
            source_name = Path(image_member).name
            target_name = f"kaggle_{Path(source_name).stem}{Path(source_name).suffix.lower()}"
            image_target = output / "images" / split / target_name
            image_target.write_bytes(image_bytes)
            (output / "labels" / split / f"{Path(target_name).stem}.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8"
            )
            fingerprint_set.add(fingerprint)
            image_totals[split] += 1
            source_totals["kaggle"].update(class_counts)
            manifest_rows.append({
                "image": target_name, "split": split, "source": "Kaggle Hard Hat Detection",
                "wearing_helmet": class_counts["helmet"],
                "not_wearing_helmet": class_counts["head"],
                "ignored_person": ignored_person,
            })
            if image_index % 1000 == 0:
                print(f"Converted Kaggle images: {image_index}/{len(all_rows)}", flush=True)

    config = {
        "path": "../data/processed_combined",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    (ROOT / "configs" / "data_combined.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "source", "wearing_helmet", "not_wearing_helmet", "ignored_person"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "seed": args.seed,
        "split_ratio_for_kaggle": SPLIT_RATIOS,
        "images_per_split": dict(image_totals),
        "objects_by_source": {source: dict(counts) for source, counts in source_totals.items()},
        "ignored_kaggle_person_boxes": sum(int(row["ignored_person"]) for row in manifest_rows),
        "kaggle_person_only_images_skipped": person_only_images,
        "duplicate_images_skipped": duplicate_count,
        "kaggle_classes": {"helmet": "wearing_helmet", "head": "not_wearing_helmet", "person": "ignored"},
        "box_note": "Kaggle boxes cover helmets or heads; SHWD annotations retain their source box extents, which are not geometrically identical.",
        "kaggle_page": "https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection",
        "output": str(output),
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
