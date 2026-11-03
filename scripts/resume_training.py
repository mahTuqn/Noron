"""Resume an interrupted Ultralytics run from its last.pt checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to the interrupted run's last.pt")
    args = parser.parse_args()
    checkpoint = Path(args.weights).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Resume checkpoint not found: {checkpoint}")

    # Ultralytics restores epoch, optimizer, scheduler, dataset, and run settings
    # from the checkpoint and continues in the original run directory.
    model = YOLO(str(checkpoint))
    model.train(resume=True)


if __name__ == "__main__":
    main()
