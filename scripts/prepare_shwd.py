"""Convert SHWD Pascal VOC annotations to Ultralytics YOLO format.

The script checks source class names and preview labels before converting.
The `hat` and `person` source labels represent wearing states; their box
geometry differs in a small number of source images.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_MAP = {"hat": 0, "person": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="SHWD directory containing images and VOC XML")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--confirm-mapping", action="store_true", help="Confirm visual review of the source box semantics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    output_root = args.output.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if not output_root.is_relative_to(project_root):
        parser.error("--output must stay inside the project directory")
    if not args.confirm_mapping:
        parser.error("Review source labels/boxes, then rerun with --confirm-mapping.")
    if args.test_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio + args.val_ratio >= 1:
        parser.error("test/val ratios must be positive and sum to less than 1")

    xmls = sorted(args.source.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No Pascal VOC XML annotations found in {args.source}")
    image_index: dict[str, Path] = {}
    image_stems: dict[str, Path] = {}
    for candidate in args.source.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            image_index.setdefault(candidate.name.lower(), candidate)
            image_stems.setdefault(candidate.stem.lower(), candidate)
    records, source_classes, excluded_unknown = [], Counter(), set()
    for xml_path in xmls:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or (xml_path.stem + ".jpg")
        image = (image_index.get(Path(filename).name.lower())
                 or image_stems.get(Path(filename).stem.lower())
                 or image_stems.get(xml_path.stem.lower()))
        if image is None:
            continue
        size = root.find("size")
        if size is None:
            continue
        width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            continue
        object_names = [(obj.findtext("name") or "").strip().lower() for obj in root.findall("object")]
        source_classes.update(object_names)
        unknown_for_image = set(object_names) - set(CLASS_MAP)
        if unknown_for_image:
            excluded_unknown.add(image.stem)
            continue
        labels = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            box = obj.find("bndbox")
            if box is None:
                continue
            x1, y1, x2, y2 = (float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            labels.append(f"{CLASS_MAP[name]} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        records.append((image, labels))

    print(f"Images paired with XML: {len(records)}")
    print(f"Source object labels: {dict(source_classes)}")
    print("Mapping: hat -> wearing_helmet; person -> not_wearing_helmet")
    unknown = sorted(set(source_classes) - set(CLASS_MAP))
    if unknown:
        print(f"Excluded {len(excluded_unknown)} images containing unsupported labels: {unknown}")
    if not records:
        raise SystemExit("No usable image/XML pairs found")
    by_stem = {image.stem: (image, labels) for image, labels in records}
    if len(by_stem) != len(records):
        raise SystemExit("Duplicate image stems found; resolve duplicates before conversion")
    image_sets = args.source / "ImageSets" / "Main"
    split_files = {split: image_sets / f"{split}.txt" for split in ("train", "val", "test")}
    if all(path.is_file() for path in split_files.values()):
        splits = {
            split: [by_stem[Path(line).stem] for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and Path(line).stem in by_stem]
            for split, path in split_files.items()
        }
        assigned = [Path(line).stem for path in split_files.values()
                    for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        overlap = sorted(stem for stem, count in Counter(assigned).items() if count > 1)
        if overlap:
            raise SystemExit(f"Official splits overlap for {len(overlap)} image IDs")
        missing = sorted(set(assigned) - set(by_stem) - excluded_unknown)
        if missing:
            raise SystemExit(f"Official split references {len(missing)} missing annotation/image pairs")
        uncovered = sorted(set(by_stem) - set(assigned))
        if uncovered:
            raise SystemExit(f"Official splits omit {len(uncovered)} usable annotations")
        print("Using official ImageSets/Main train.txt, val.txt, test.txt splits")
    else:
        random.Random(args.seed).shuffle(records)
        n_test, n_val = round(len(records) * args.test_ratio), round(len(records) * args.val_ratio)
        splits = {"test": records[:n_test], "val": records[n_test:n_test+n_val], "train": records[n_test+n_val:]}
        print("No complete official split found; using a seeded random split")
    for split, items in splits.items():
        image_dir, label_dir = args.output / "images" / split, args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        expected_images = {image.name for image, _ in items}
        expected_labels = {f"{image.stem}.txt" for image, _ in items}
        for stale in image_dir.iterdir():
            if stale.is_file() and stale.suffix.lower() in IMAGE_EXTS and stale.name not in expected_images:
                stale.unlink()
        for stale in label_dir.glob("*.txt"):
            if stale.name not in expected_labels:
                stale.unlink()
        for image, labels in items:
            shutil.copy2(image, image_dir / image.name)
            (label_dir / f"{image.stem}.txt").write_text("\n".join(labels), encoding="utf-8")
        print(f"{split}: {len(items)} images")
    print("Review converted labels and example boxes before training.")


if __name__ == "__main__":
    main()
