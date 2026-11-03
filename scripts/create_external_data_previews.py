"""Create small contact sheets for manual inspection of external datasets."""
from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_data_analysis"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLORS = {0: (0, 170, 70), 1: (220, 65, 60), 2: (45, 135, 220), 3: (220, 160, 30)}
CLASS_NAMES = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}


def pick_spread(files: list[Path], count: int) -> list[Path]:
    if not files:
        return []
    if len(files) <= count:
        return files
    indices = [round(i * (len(files) - 1) / (count - 1)) for i in range(count)] if count > 1 else [len(files) // 2]
    return [files[i] for i in indices]


def draw_yolo(image: Image.Image, label_path: Path, class_names: dict[int, str]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if not label_path.exists():
        return image
    for row in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = row.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except (ValueError, OverflowError):
            continue
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)
        color = COLORS.get(class_id, (255, 255, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0, y1 - 13)), class_names.get(class_id, str(class_id)), fill=color)
    return image


def image_sheet() -> None:
    deb = ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted" / "test"
    ppe = ROOT / "ext" / "mendeley_ppe" / "extracted" / "Dataset of Personal Protective Equipment (PPE)" / "20250731-PPE2286y" / "20250731-ppe2286y" / "valid"
    picks: list[tuple[Path, Path, str, dict[int, str]]] = []
    for split, title, names, count in (
        (deb, "DeBoniz external test", {0: "Helmet", 1: "No-Helmet"}, 8),
        (ppe, "Mendeley PPE validation", CLASS_NAMES, 8),
    ):
        files = sorted(p for p in (split / "images").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        picks.extend((image, split / "labels" / f"{image.stem}.txt", title, names) for image in pick_spread(files, count))

    tile_w, tile_h, label_h, cols = 320, 220, 34, 4
    rows = (len(picks) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image_path, labels, source, names) in enumerate(picks):
        image = Image.open(image_path).convert("RGB")
        image = draw_yolo(image, labels, names)
        image.thumbnail((tile_w, tile_h))
        x = (i % cols) * tile_w + (tile_w - image.width) // 2
        y = (i // cols) * (tile_h + label_h)
        canvas.paste(image, (x, y))
        draw.text(((i % cols) * tile_w + 8, y + tile_h + 3), f"{source} · {image_path.name[:28]}", fill=(20, 30, 45))
    canvas.save(OUT / "external_image_samples.jpg", quality=88)


def video_sheet() -> None:
    root = ROOT / "ext" / "mendeley_motorcycle_video" / "extracted" / "Dataset for the Detection of Helmet Use in Motorcycle Drivers"
    videos = sorted(root.rglob("*.mp4"), key=lambda p: (p.parent.name.lower(), p.name.lower()))
    tile_w, tile_h, label_h, cols = 320, 180, 32, 4
    rows = (len(videos) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, video in enumerate(videos):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((tile_w, tile_h))
        else:
            image = Image.new("RGB", (tile_w, tile_h), (225, 225, 225))
        x = (i % cols) * tile_w + (tile_w - image.width) // 2
        y = (i // cols) * (tile_h + label_h)
        canvas.paste(image, (x, y))
        label = f"{video.parent.name} · {video.name}"
        draw.text(((i % cols) * tile_w + 8, y + tile_h + 3), label, fill=(20, 30, 45), font=font)
    canvas.save(OUT / "motorcycle_video_samples.jpg", quality=88)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image_sheet()
    video_sheet()
    print(f"Created previews in {OUT}")


if __name__ == "__main__":
    main()
