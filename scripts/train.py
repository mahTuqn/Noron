"""Train YOLOv8n/s/m with a shared, recorded baseline configuration."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--models", nargs="+", default=["n", "s", "m"], choices=["n", "s", "m"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1, help="Conservative default for 4 GB GPUs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2, help="Conservative default for Windows")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop after this many validation epochs without improvement")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of train split; use 0.01 for a pipeline smoke run")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Disable by default to limit dense-label batch expansion")
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"Dataset config not found: {args.data}")
    data_path = Path(args.data).resolve()
    run_name = datetime.now().strftime("helmet_%Y%m%d_%H%M%S")
    for size in args.models:
        model = YOLO(f"yolov8{size}.pt")
        model.train(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, fraction=args.fraction,
                    device=args.device, seed=args.seed, workers=args.workers, patience=args.patience, mosaic=args.mosaic,
                    project=str((Path.cwd() / "runs" / "detect").resolve()), name=f"{run_name}_yolov8{size}",
                    exist_ok=False, pretrained=True, deterministic=True)
        run_dir = Path("runs/detect") / f"{run_name}_yolov8{size}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "project_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"Completed yolov8{size}: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
