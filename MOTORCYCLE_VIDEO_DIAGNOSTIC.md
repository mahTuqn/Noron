# Motorcycle video diagnostic inference

`ext/mendeley_motorcycle_video` contains 32 source clips (16 paired participants,
with and without a helmet), about 114,640 frames and 63m45s of footage. The clips
are 1920×1080 at about 29.97 FPS. The supplied labels describe each clip only;
there are no frame-level object boxes. They are useful for qualitative,
out-of-domain analysis, but they are not detection ground truth.

## Run on every frame

From the project root, run the script with the model checkpoint to inspect:

```powershell
python scripts/infer_motorcycle_video_dataset.py `
  --weights runs/detect/helmet/weights/best.pt `
  --output-root results/motorcycle_video_diagnostic `
  --conf 0.25
```

The default inventory is `results/external_data_analysis/video_inventory.csv`.
Its relative paths resolve from the current directory; run the command from the
project root. The default `--expected-videos 32` makes the script stop if the
inventory is incomplete. Set `--video-root` if those inventory paths should be
resolved against another directory. `--device cpu` selects CPU inference;
`--device 0` requests CUDA device 0. Ultralytics chooses automatically if
`--device` is omitted. `--imgsz` defaults to 640. Frames are inferred in
batches of 16 by default; use `--batch-size 4` (or `1`) if memory is limited,
or increase it if the inference device has more headroom. Batches flush at
each clip boundary, and output rows remain ordered by clip and frame.

Each invocation creates a new UTC timestamped output directory. It writes:

- `frame_detections.jsonl`: one record for every decoded frame, including frames
  with no detections. Each record includes source-relative path, clip label,
  participant ID, frame index/time, dimensions, and boxes in pixel `xyxy` plus
  normalized `xywh` coordinates.
- `summary.json`: per-video and overall decoded-frame counts, inventory frame
  count checks, settings, model and inventory SHA-256 hashes, and an explicit
  warning about label provenance.
- Optional `annotated/` MP4s if `--save-annotated-videos` is supplied.

Annotated video encoding is off by default. It is independent of batch
inference and does not change the per-frame JSONL output.

Example with visual output:

```powershell
python scripts/infer_motorcycle_video_dataset.py `
  --weights runs/detect/helmet/weights/best.pt `
  --output-root results/motorcycle_video_diagnostic `
  --conf 0.25 --save-annotated-videos
```

The script reads source videos only; it does not rewrite them. It decodes and
infers on every frame rather than sampling. It checks each clip's decoded frame
count against the inventory and records mismatches in the summary. If any clip
has fewer or more decoded frames than inventoried, it completes the remaining
clips, preserves the JSONL and summary, names each mismatched clip, and exits
with status 2 so the run cannot be mistaken for a complete pass.

## How to use the results

These outputs are diagnostic predictions or pseudo-label candidates. A clip
label such as “With Helmet” cannot determine which person or head in any frame
wears a helmet, where a helmet box belongs, or whether a hardhat-like object is
actually a motorcycle helmet. In particular, the existing model's known
confusion between ordinary hats and safety helmets makes unreviewed
pseudo-labels unsafe as training targets.

There is no safe automatic conversion of these clip labels into YOLO detection
labels without inventing boxes. To use video frames for supervised training,
annotate helmet and non-helmet head/person boxes on selected frames (and review
all proposed model boxes), then split train/validation/test by participant ID
so paired footage of one person cannot leak across splits. Model-generated
boxes can speed up annotation only after a human confirms or corrects them.
Until that review exists, retain the 32 clips as held-out qualitative analysis
and do not merge their predicted boxes into the supervised training set.

## Resource planning

The clips total about 2.66 GiB and 114,640 frames. At a previously observed
inference rate of roughly 53 FPS, model execution alone would take about
36 minutes; decoding, saving JSONL and optional video encoding add time, and
the actual rate depends on hardware and selected weights. Annotated output can
require several additional GiB. The script is not run automatically during
training and does not start training.
