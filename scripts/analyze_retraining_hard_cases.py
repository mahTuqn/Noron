"""Create diagnostic predictions for the three user-reviewed hard-case images.

This script reads source media and labels only. Its predictions and previews
are written to a new timestamped results directory; no labels or source images
are edited, and no metrics or training targets are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEBONIZ_IMAGES = ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted" / "test" / "images"
DEBONIZ_LABELS = ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted" / "test" / "labels"
PPE_ROOT = ROOT / "ext" / "mendeley_ppe" / "extracted"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "retraining_hard_cases"

DEBONIZ_CLASS_NAMES = {0: "wearing_helmet", 1: "not_wearing_helmet"}
PPE_CLASS_NAMES = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "image"


def parse_source_labels(path: Path, class_names: dict[int, str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected source label file is missing: {path}")
    boxes = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            coordinates = [float(item) for item in fields[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Malformed annotation at {path}:{line_number}: {raw_line}") from exc
        boxes.append({
            "line_number": line_number,
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"unknown_class_{class_id}"),
            "coordinates": coordinates,
            "coordinate_format": "normalized_yolo_xywh" if len(coordinates) == 4 else "source_format_uninterpreted",
            "raw_line": line,
        })
    return boxes


def collect_cases() -> list[dict[str, Any]]:
    deboniz_images = sorted(DEBONIZ_IMAGES.glob("*mp4-266*"))
    if len(deboniz_images) != 2:
        raise ValueError(f"Expected exactly 2 canonical DeBoniz mp4-266 images, found {len(deboniz_images)}")
    ppe_images = sorted(path for path in PPE_ROOT.rglob("00751*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(ppe_images) != 1:
        raise ValueError(f"Expected exactly 1 Mendeley PPE 00751 image, found {len(ppe_images)}")

    cases: list[dict[str, Any]] = []
    for image_path in deboniz_images:
        label_path = DEBONIZ_LABELS / f"{image_path.stem}.txt"
        cases.append({
            "case_id": f"deboniz_mp4-266_{safe_name(image_path.stem)}",
            "dataset": "DeBoniz Helmet Detection",
            "image_path": image_path,
            "label_path": label_path,
            "source_class_names": DEBONIZ_CLASS_NAMES,
            "label_scope": "source_detection_boxes",
            "review_status": "qualitative_hard_negative_review",
            "review_note": (
                "User confirmed the displayed baseball cap is not a safety helmet. "
                "Preserve the original No-Helmet boxes; inspect whether the model still predicts a helmet."
            ),
        })

    ppe_image = ppe_images[0]
    ppe_label = ppe_image.parent.parent / "labels" / f"{ppe_image.stem}.txt"
    source_boxes = parse_source_labels(ppe_label, PPE_CLASS_NAMES)
    helmet_boxes = [box for box in source_boxes if box["class_id"] in (0, 1)]
    cases.append({
        "case_id": "mendeley_ppe_00751",
        "dataset": "Mendeley PPE",
        "image_path": ppe_image,
        "label_path": ppe_label,
        "source_class_names": PPE_CLASS_NAMES,
        "label_scope": "vest_boxes_only_missing_helmet_task_boxes" if not helmet_boxes else "source_detection_boxes",
        "review_status": "qualitative_unscorable_missing_helmet_nohelmet_boxes" if not helmet_boxes else "qualitative_review",
        "review_note": (
            "User identified a visible red safety helmet, but the source annotation has only Vest boxes "
            "and no Helmet/NoHelmet boxes. Qualitative review only; exclude from helmet-specific scoring."
        ),
    })
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="YOLO .pt checkpoint")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Parent for a new UTC timestamped output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    cases = collect_cases()
    for case in cases:
        if not case["image_path"].is_file():
            raise FileNotFoundError(f"Source image is missing: {case['image_path']}")
        case["source_boxes"] = parse_source_labels(case["label_path"], case["source_class_names"])

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project inference dependencies (ultralytics and opencv-python).") from exc

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_root.resolve() / run_stamp
    predictions_dir = run_dir / "predictions"
    previews_dir = run_dir / "previews"
    predictions_dir.mkdir(parents=True, exist_ok=False)
    previews_dir.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(weights))
    model_names = getattr(model, "names", {})
    if isinstance(model_names, list):
        model_names = {index: name for index, name in enumerate(model_names)}

    processed = []
    for case in cases:
        image_path: Path = case["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read source image: {image_path}")
        height, width = image.shape[:2]
        result = model.predict(source=image, imgsz=512, conf=0.25, verbose=False)[0]
        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
                x1, y1, x2, y2 = [float(value) for value in coords]
                detections.append({
                    "class_id": int(class_id),
                    "class_name": str(model_names.get(int(class_id), class_id)),
                    "confidence": float(confidence),
                    "xyxy_pixels": [x1, y1, x2, y2],
                    "xywh_normalized": [
                        ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                        (x2 - x1) / width, (y2 - y1) / height,
                    ],
                })

        preview_path = previews_dir / f"{case['case_id']}.jpg"
        if not cv2.imwrite(str(preview_path), result.plot()):
            raise RuntimeError(f"Could not write annotated preview: {preview_path}")
        prediction = {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "source_image": image_path.relative_to(ROOT).as_posix(),
            "source_image_sha256": sha256_file(image_path),
            "source_label": case["label_path"].relative_to(ROOT).as_posix(),
            "source_label_sha256": sha256_file(case["label_path"]),
            "source_label_scope": case["label_scope"],
            "review_status": case["review_status"],
            "review_note": case["review_note"],
            "source_annotations": case["source_boxes"],
            "source_annotation_interpretation": case["source_class_names"],
            "image_width": width,
            "image_height": height,
            "inference": {"weights_sha256": sha256_file(weights), "imgsz": 512, "confidence_threshold": 0.25},
            "predictions_are_ground_truth": False,
            "scoring_performed": False,
            "predictions": detections,
            "annotated_preview": preview_path.relative_to(run_dir).as_posix(),
        }
        prediction_path = predictions_dir / f"{case['case_id']}.json"
        prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        processed.append({
            "case_id": case["case_id"],
            "review_status": case["review_status"],
            "source_image": image_path.relative_to(ROOT).as_posix(),
            "prediction_json": prediction_path.relative_to(run_dir).as_posix(),
            "annotated_preview": preview_path.relative_to(run_dir).as_posix(),
            "prediction_count": len(detections),
        })

    summary = {
        "run_id": run_stamp,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights),
        "inference": {"imgsz": 512, "confidence_threshold": 0.25},
        "purpose": "Qualitative diagnostic review of user-reported model hard cases; no metric scoring or training labels are produced.",
        "source_media_modified": False,
        "source_labels_modified": False,
        "cases": processed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(run_dir), "cases_processed": len(processed), "inference_run": True}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
