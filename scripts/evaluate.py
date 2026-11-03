"""Evaluate a trained checkpoint on the held-out test split."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    cfg_path = Path(args.data)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config["path"] = str((cfg_path.parent / config.get("path", "..")).resolve())
    if not (Path(config["path"]) / config["test"]).exists():
        raise SystemExit("Test split not found. Prepare data and keep test split independent.")
    model = YOLO(args.weights)
    run_name = f"{Path(args.weights).parent.parent.name}_test_{datetime.now():%Y%m%d_%H%M%S}"
    metrics = model.val(data=str(cfg_path.resolve()), split="test", imgsz=args.imgsz, batch=args.batch,
                        workers=args.workers, device=args.device, plots=True,
                        project=str((ROOT / "results").resolve()), name=run_name)
    result_dir = ROOT / "results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    names = model.names
    class_ids = [int(class_id) for class_id in metrics.box.ap_class_index]
    per_class = []
    for index, class_id in enumerate(class_ids):
        precision, recall, map50, map5095 = metrics.box.class_result(index)
        per_class.append({
            "class_id": class_id,
            "class_name": names[class_id],
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(map50),
            "mAP50_95": float(map5095),
        })
    with (result_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"])
        writer.writeheader()
        writer.writerows(per_class)
    inference_ms = float(metrics.speed.get("inference", float("nan")))
    summary = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(cfg_path.resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "inference_ms_per_image": inference_ms,
        "inference_fps": 1000.0 / inference_ms if inference_ms > 0 else None,
        "per_class": per_class,
    }
    (result_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Inference ms/image: {inference_ms:.2f}")
    print(f"Inference FPS: {1000.0 / inference_ms:.2f}" if inference_ms > 0 else "Inference FPS: unavailable")
    print(f"Per-class metrics: {result_dir / 'per_class_metrics.csv'}")


if __name__ == "__main__":
    main()
