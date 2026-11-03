"""Summarize test errors by box size and image brightness at a fixed threshold."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def size_group(area_fraction: float) -> str:
    if area_fraction < 0.01:
        return "small_<1%"
    if area_fraction < 0.04:
        return "medium_1-4%"
    return "large_>=4%"


def brightness_group(mean_luminance: float) -> str:
    if mean_luminance < 85:
        return "dark_<85"
    if mean_luminance < 170:
        return "mid_85-170"
    return "bright_>=170"


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        box = np.array([(xc - bw / 2) * width, (yc - bh / 2) * height,
                        (xc + bw / 2) * width, (yc + bh / 2) * height], dtype=float)
        rows.append((class_id, box, bw * bh))
    return rows


def match_boxes(ground_truth, predictions, threshold: float):
    candidates = []
    for gt_index, (gt_class, gt_box, _) in enumerate(ground_truth):
        for pred_index, (pred_class, pred_box, _) in enumerate(predictions):
            if gt_class != pred_class:
                continue
            overlap = box_iou(gt_box, pred_box)
            if overlap >= threshold:
                candidates.append((overlap, gt_index, pred_index))
    matched_gt, matched_pred = set(), set()
    for _, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return matched_gt, matched_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/data_combined.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU used to match predictions to labels")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Use FP16 inference to reduce VRAM; pass --no-fp16 to disable")
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()

    cfg_path = Path(args.data).resolve()
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset_root = (cfg_path.parent / config.get("path", "..")).resolve()
    image_dir = (dataset_root / config["test"]).resolve()
    label_dir = (dataset_root / "labels" / "test").resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Test images not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        raise SystemExit("No test images found")

    model = YOLO(args.weights)
    name_map = model.names
    class_name = lambda cid: str(name_map.get(cid, cid)) if isinstance(name_map, dict) else str(name_map[cid])
    output = Path(args.outdir) if args.outdir else ROOT / "results" / "error_analysis" / f"{Path(args.weights).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    output = output.resolve()
    if output != ROOT and ROOT not in output.parents:
        raise ValueError(f"Output must stay inside project: {output}")
    output.mkdir(parents=True, exist_ok=True)

    strata: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    class_totals: dict[int, Counter[str]] = defaultdict(Counter)
    top_examples = []

    def predict_in_chunks(chunk_size: int = 16):
        for start in range(0, len(images), chunk_size):
            chunk_paths = [str(p) for p in images[start : start + chunk_size]]
            results = model.predict(source=chunk_paths, imgsz=args.imgsz, conf=args.conf,
                                    iou=0.7, device=args.device, quantize=16 if args.fp16 else None,
                                    stream=True, batch=1, workers=0, verbose=False)
            for offset, prediction in enumerate(results, 1):
                yield start + offset, prediction
                del prediction
            del results
            if args.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    for index, result in predict_in_chunks():
        result = result.cpu()
        image_path = Path(result.path)
        height, width = result.orig_shape
        label_path = label_dir / f"{image_path.stem}.txt"
        gt = read_labels(label_path, width, height)
        if result.boxes is None or len(result.boxes) == 0:
            pred = []
        else:
            pred = [
                (int(class_id), box.astype(float), float(confidence))
                for class_id, box, confidence in zip(
                    result.boxes.cls.cpu().numpy(), result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
                )
            ]
        matched_gt, matched_pred = match_boxes(gt, pred, args.iou)
        bgr = result.orig_img
        luminance = float(np.mean(0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]))
        light = brightness_group(luminance)
        for gt_index, (class_id, _, area) in enumerate(gt):
            status = "tp" if gt_index in matched_gt else "fn"
            class_totals[class_id][status] += 1
            for kind, group in (("size", size_group(area)), ("brightness", light)):
                strata[(kind, group, class_id)][status] += 1
        for pred_index, (class_id, box, _) in enumerate(pred):
            if pred_index not in matched_pred:
                class_totals[class_id]["fp"] += 1
                predicted_area_fraction = (
                    max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (width * height)
                )
                for kind, group in (
                    ("brightness", light),
                    ("size", size_group(predicted_area_fraction)),
                ):
                    strata[(kind, group, class_id)]["fp"] += 1

        fn_count = len(gt) - len(matched_gt)
        fp_count = len(pred) - len(matched_pred)
        if fn_count or fp_count:
            rgb = Image.fromarray(bgr[:, :, ::-1].copy()).convert("RGB")
            draw = ImageDraw.Draw(rgb)
            for gt_index, (class_id, box, _) in enumerate(gt):
                color = "#00a676" if gt_index in matched_gt else "#e63946"
                draw.rectangle(tuple(box.tolist()), outline=color, width=max(2, width // 500))
                if gt_index not in matched_gt:
                    draw.text((box[0], max(0, box[1] - 12)), f"FN {class_name(class_id)}", fill=color, font=ImageFont.load_default())
            for pred_index, (class_id, box, confidence) in enumerate(pred):
                if pred_index not in matched_pred:
                    draw.rectangle(tuple(box.tolist()), outline="#4361ee", width=max(2, width // 500))
                    draw.text((box[0], max(0, box[1] - 12)), f"FP {class_name(class_id)} {confidence:.2f}", fill="#4361ee", font=ImageFont.load_default())
            scale = min(640 / width, 400 / height, 1.0)
            rgb.thumbnail((int(width * scale), int(height * scale)))
            top_examples.append((fn_count + fp_count, image_path.name, fn_count, fp_count, luminance, rgb))
            top_examples.sort(key=lambda row: (row[0], row[2]), reverse=True)
            del top_examples[12:]
        if index % 250 == 0 or index == len(images):
            print(f"Analyzed {index}/{len(images)} test images", flush=True)
        del result

    rows = []
    for (kind, group, class_id), counts in sorted(strata.items()):
        tp, fn, fp = counts["tp"], counts["fn"], counts["fp"]
        rows.append({"stratum": kind, "group": group, "class_id": class_id, "class_name": class_name(class_id),
                     "ground_truth": tp + fn, "TP": tp, "FN": fn, "FP": fp,
                     "recall": tp / (tp + fn) if tp + fn else None})
    with (output / "strata_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stratum", "group", "class_id", "class_name", "ground_truth", "TP", "FN", "FP", "recall"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "test_images": len(images),
        "confidence_threshold": args.conf,
        "matching_iou": args.iou,
        "imgsz": args.imgsz,
        "box_size_bands": {"small_<1%": "area fraction < 0.01", "medium_1-4%": "0.01-0.04", "large_>=4%": ">= 0.04"},
        "brightness_bands": {"dark_<85": "mean luminance < 85/255", "mid_85-170": "85-169", "bright_>=170": ">=170"},
        "per_class_at_threshold": {
            class_name(cid): {**dict(counts), "recall": counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None}
            for cid, counts in class_totals.items()
        },
        "occlusion_note": "The dataset has no explicit occlusion labels; inspect error_examples.jpg for qualitative review of missed or confused objects.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if top_examples:
        canvas = Image.new("RGB", (1320, 4 * 430), "#edf2f4")
        draw = ImageDraw.Draw(canvas)
        for position, (error_count, filename, fn_count, fp_count, luminance, image) in enumerate(top_examples):
            x, y = 20 + (position % 2) * 650, 15 + (position // 2) * 430
            draw.text((x, y), f"{filename} · FN={fn_count} FP={fp_count} · mean Y={luminance:.0f}", fill="#102b3f")
            canvas.paste(image, (x, y + 22))
        canvas.save(output / "error_examples.jpg", quality=90)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved error analysis to {output}")


if __name__ == "__main__":
    main()
