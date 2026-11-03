"""Run frame-by-frame diagnostic inference on the Mendeley motorcycle videos.

Every decoded frame is passed through YOLO. The source clips only have
clip-level With Helmet / Without Helmet labels and provide no detection boxes;
outputs from this script are predictions for review, not supervised labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INVENTORY = Path("results/external_data_analysis/video_inventory.csv")
DEFAULT_OUTPUT = Path("results/motorcycle_video_diagnostic")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer on every frame of every video listed in the inventory CSV."
    )
    parser.add_argument("--weights", required=True, type=Path, help="YOLO .pt weights")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--video-root", type=Path, default=Path("."),
                        help="Base directory used to resolve inventory relative_path entries")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Maximum decoded frames per inference batch (lower to reduce memory use)")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0; default auto")
    parser.add_argument("--expected-videos", type=int, default=32,
                        help="Fail if inventory has a different number of videos; set 0 to disable")
    parser.add_argument("--save-annotated-videos", action="store_true",
                        help="Also encode one annotated MP4 per source clip (uses substantial disk)")
    args = parser.parse_args()
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.expected_videos < 0:
        parser.error("--expected-videos must be zero or positive")
    return args


def inventory_rows(inventory: Path, video_root: Path, expected_count: int) -> list[dict[str, Any]]:
    with inventory.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No video rows found in inventory: {inventory}")
    if expected_count and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} videos, inventory contains {len(rows)}")
    seen: set[Path] = set()
    for row in rows:
        video_path = (video_root / row["relative_path"]).resolve()
        if video_path in seen:
            raise ValueError(f"Duplicate video path in inventory: {video_path}")
        seen.add(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Inventory video is missing: {video_path}")
        row["resolved_path"] = video_path
        row["expected_frames"] = int(row["frames"])
        row["fps"] = float(row["fps"])
        row["width"] = int(row["width"])
        row["height"] = int(row["height"])
    return sorted(rows, key=lambda row: (row["clip_label"], row["filename"]))


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    inventory_path = args.inventory.resolve()
    video_root = args.video_root.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    rows = inventory_rows(inventory_path, video_root, args.expected_videos)

    # Import inference dependencies only after validating inputs.
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the project inference dependencies (ultralytics and opencv-python).") from exc

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_root.resolve() / run_stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    detections_path = run_dir / "frame_detections.jsonl"
    summary_path = run_dir / "summary.json"
    model = YOLO(str(weights))
    model_names = getattr(model, "names", {})
    if isinstance(model_names, list):
        model_names = {i: name for i, name in enumerate(model_names)}

    all_summaries: list[dict[str, Any]] = []
    frame_count_mismatches: list[dict[str, Any]] = []
    overall = {"videos": len(rows), "frames_decoded": 0, "frames_with_detections": 0,
               "detection_count": 0, "frames_expected_from_inventory": sum(r["expected_frames"] for r in rows)}
    with detections_path.open("w", encoding="utf-8", newline="\n") as output:
        for video_num, row in enumerate(rows, start=1):
            video_path: Path = row["resolved_path"]
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or row["fps"]
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            participant_match = re.search(r"p(\d+)", Path(row["filename"]).stem, re.I)
            participant_id = participant_match.group(1) if participant_match else None
            video_key = f"{safe_name(row['clip_label'])}_{safe_name(Path(row['filename']).stem)}"
            annotated_path = run_dir / "annotated" / f"{video_key}.mp4" if args.save_annotated_videos else None
            writer = None
            if annotated_path is not None:
                annotated_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    cap.release()
                    raise RuntimeError(f"Could not create annotated video: {annotated_path}")

            video_counts = {"frames_decoded": 0, "frames_with_detections": 0, "detection_count": 0}
            while True:
                frames = []
                frame_indices = []
                for _ in range(args.batch_size):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                    frame_indices.append(video_counts["frames_decoded"] + len(frames) - 1)
                if not frames:
                    break
                predict_args: dict[str, Any] = {
                    "source": frames, "batch": len(frames), "conf": args.conf,
                    "imgsz": args.imgsz, "verbose": False,
                }
                if args.device is not None:
                    predict_args["device"] = args.device
                results = model.predict(**predict_args)
                if len(results) != len(frames):
                    raise RuntimeError(
                        f"Inference returned {len(results)} results for {len(frames)} frames in {video_path}"
                    )
                # Ultralytics returns results in source order; serialize them in that same order.
                for frame_index, frame, result in zip(frame_indices, frames, results):
                    frame_height, frame_width = frame.shape[:2]
                    detections = []
                    boxes = result.boxes
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.cpu().tolist()
                        confs = boxes.conf.cpu().tolist()
                        classes = boxes.cls.int().cpu().tolist()
                        for coords, confidence, class_id in zip(xyxy, confs, classes):
                            x1, y1, x2, y2 = [float(v) for v in coords]
                            detections.append({
                                "class_id": int(class_id),
                                "class_name": str(model_names.get(int(class_id), class_id)),
                                "confidence": float(confidence),
                                "xyxy_pixels": [x1, y1, x2, y2],
                                "xywh_normalized": [
                                    ((x1 + x2) / 2) / frame_width, ((y1 + y2) / 2) / frame_height,
                                    (x2 - x1) / frame_width, (y2 - y1) / frame_height,
                                ],
                            })
                    record = {
                        "video_relative_path": row["relative_path"],
                        "clip_label": row["clip_label"],
                        "participant_id": participant_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps if fps > 0 else None,
                        "fps": fps, "width": frame_width, "height": frame_height,
                        "detections": detections,
                    }
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    if detections:
                        video_counts["frames_with_detections"] += 1
                        video_counts["detection_count"] += len(detections)
                    video_counts["frames_decoded"] += 1
                    if writer is not None:
                        writer.write(result.plot())
            cap.release()
            if writer is not None:
                writer.release()
            summary = {
                "video_relative_path": row["relative_path"], "clip_label": row["clip_label"],
                "participant_id": participant_id, "expected_frames": row["expected_frames"],
                "container_reported_frames": reported_frames, **video_counts,
                "frame_count_matches_inventory": video_counts["frames_decoded"] == row["expected_frames"],
                "annotated_video": str(annotated_path.relative_to(run_dir)) if annotated_path else None,
            }
            all_summaries.append(summary)
            if not summary["frame_count_matches_inventory"]:
                mismatch = {
                    "video_relative_path": row["relative_path"],
                    "expected_frames": row["expected_frames"],
                    "decoded_frames": video_counts["frames_decoded"],
                }
                frame_count_mismatches.append(mismatch)
            for key in ("frames_decoded", "frames_with_detections", "detection_count"):
                overall[key] += video_counts[key]
            print(f"[{video_num}/{len(rows)}] {video_key}: {video_counts['frames_decoded']} frames", flush=True)

    metadata = {
        "run_id": run_stamp,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights_path": str(weights), "weights_sha256": sha256_file(weights),
        "inventory_path": str(inventory_path), "inventory_sha256": sha256_file(inventory_path),
        "confidence_threshold": args.conf, "image_size": args.imgsz,
        "batch_size": args.batch_size, "device": args.device or "auto",
        "annotated_videos_saved": bool(args.save_annotated_videos),
        "overall": overall, "videos": all_summaries,
        "frame_count_mismatches": frame_count_mismatches,
        "interpretation": (
            "Diagnostic predictions / pseudo-label candidates only. Source labels are clip-level; "
            "no frame-level bounding-box ground truth is supplied. Do not use these predictions "
            "as supervised labels without human review and correction."
        ),
    }
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed: {overall['frames_decoded']} frames from {len(rows)} videos")
    print(f"Detections: {detections_path}")
    print(f"Summary: {summary_path}")
    if frame_count_mismatches:
        print("ERROR: decoded frame counts did not match inventory; partial outputs and summary were preserved:", file=sys.stderr)
        for mismatch in frame_count_mismatches:
            print(
                f"  {mismatch['video_relative_path']}: expected {mismatch['expected_frames']}, "
                f"decoded {mismatch['decoded_frames']}",
                file=sys.stderr,
            )
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
