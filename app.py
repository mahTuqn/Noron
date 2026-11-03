"""Browser demo for Helmet Detection with YOLOv8 (Streamlit)."""

from __future__ import annotations

import importlib.util
import hashlib
import math
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)

import tempfile
import streamlit as st
from PIL import Image

WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Helmet Detection · YOLOv8", page_icon="🦺", layout="wide")

st.markdown("""
<style>
  .block-container {max-width: 1180px; padding-top: 2.1rem;}
  .hero h1 {color:#ffffff !important; margin:0 0 .45rem 0;}
  .hero p {font-size:1.05rem;}
  [data-testid="stMetric"] {background:#f5f7fa; border:1px solid #e3e8ef; padding:14px 16px; border-radius:12px;}
  .hero {padding:1.25rem 1.5rem; border-radius:16px; background:linear-gradient(115deg,#102b3f,#176b66); color:white; margin-bottom:1.3rem;}
  .hero p {color:#d5e8e5; margin-bottom:0;}
</style>
<div class="hero"><h1 style="color:#fff !important">Phát hiện mũ bảo hộ</h1><p>YOLOv8 · Phân tích ảnh và video công trường</p></div>
""", unsafe_allow_html=True)

has_cv2 = importlib.util.find_spec("cv2") is not None
has_ultralytics = importlib.util.find_spec("ultralytics") is not None
if not has_cv2 or not has_ultralytics:
    missing = [name for name, ready in (("opencv-python-headless", has_cv2), ("ultralytics", has_ultralytics)) if not ready]
    st.warning("Môi trường hiện thiếu: " + ", ".join(missing) + ". Cài các gói trong `requirements.txt` để bật suy luận YOLO.")

available = sorted(WEIGHTS_DIR.glob("*.pt"))
default_checkpoint = next((index + 1 for index, path in enumerate(available) if path.name == "helmet_best.pt"), 0)
left, right = st.columns([1, 2], gap="large")
sample_dir = ROOT / "data" / "processed_combined" / "images" / "val"
if not sample_dir.is_dir():
    sample_dir = ROOT / "data" / "processed" / "images" / "val"

sample_image_sources = {
    "Validation của dự án": (sample_dir, "Ảnh validation của dự án"),
    "DeBoniz · external test": (
        ROOT / "ext" / "deboniz_helmet" / "extracted" / "datasetmodelloscelto" / "dataset10kadjusted" / "test" / "images",
        "DeBoniz external test",
    ),
    "Mendeley PPE · valid": (
        ROOT / "ext" / "mendeley_ppe" / "extracted" / "Dataset of Personal Protective Equipment (PPE)" / "20250731-PPE2286y" / "20250731-ppe2286y" / "valid" / "images",
        "Mendeley PPE valid",
    ),
}
sample_image_sources = {
    name: details for name, details in sample_image_sources.items() if details[0].is_dir()
}
moto_video_root = ROOT / "ext" / "mendeley_motorcycle_video" / "extracted" / "Dataset for the Detection of Helmet Use in Motorcycle Drivers"
moto_video_classes = {
    "With Helmet": moto_video_root / "With Helmet",
    "Without Helmet": moto_video_root / "Without Helmet",
}
moto_video_classes = {
    name: folder for name, folder in moto_video_classes.items() if folder.is_dir()
}
with left:
    st.subheader("Thiết lập")
    st.caption("Chọn checkpoint YOLOv8 đã huấn luyện hoặc nạp tệp .pt.")
    model_choice = st.selectbox("Checkpoint trong dự án", ["Chưa có checkpoint"] + [p.name for p in available], index=default_checkpoint)
    weights_upload = st.file_uploader("Hoặc tải checkpoint .pt", type=["pt"], help="Tệp được lưu tạm trong phiên demo.")
    confidence = st.slider("Ngưỡng confidence", 0.05, 0.95, 0.25, 0.05)
    input_kind = st.radio("Loại đầu vào", ["Ảnh", "Video"], horizontal=True)
    st.caption("FPS demo bao gồm đọc khung hình, suy luận và vẽ kết quả.")

with right:
    st.subheader("Kết quả")
    uploaded = None
    sample_path = None
    sample_source_label = None
    sample_images = []
    sample_video_path = None
    sample_video_label = None
    if input_kind == "Ảnh":
        uploaded = st.file_uploader("Tải ảnh lên để phân tích", type=["jpg", "jpeg", "png"], key="input-image")
        if sample_image_sources:
            selected_source = st.selectbox("Nguồn ảnh mẫu", list(sample_image_sources), key="sample-image-source")
            selected_dir, sample_source_label = sample_image_sources[selected_source]
            # Spread up to 50 deterministic examples across the full source split.
            all_sample_images = sorted(p for p in selected_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if len(all_sample_images) > 50:
                sample_images = [all_sample_images[round(i * (len(all_sample_images) - 1) / 49)] for i in range(50)]
            else:
                sample_images = all_sample_images
            if sample_images:
                selected_sample = st.selectbox(
                    f"Chọn ảnh ({len(sample_images)} mẫu tối đa từ nguồn này)",
                    ["— Chọn ảnh mẫu —"] + [p.name for p in sample_images],
                    key=f"sample-image-{selected_source}",
                )
                if selected_sample != "— Chọn ảnh mẫu —":
                    sample_path = selected_dir / selected_sample
        else:
            st.caption("Chưa có thư mục ảnh mẫu trong dự án.")
    else:
        video_source = st.radio(
            "Nguồn video",
            ["Tải video lên", "Video mẫu Mendeley motorcycle"],
            horizontal=True,
            key="video-source",
        )
        if video_source == "Tải video lên":
            uploaded = st.file_uploader("Tải video lên để phân tích", type=["mp4", "mov", "avi", "mkv"], key="input-video")
        elif moto_video_classes:
            selected_video_class = st.selectbox("Nhãn clip-level", list(moto_video_classes), key="moto-video-class")
            video_files = sorted(moto_video_classes[selected_video_class].glob("*.mp4"))
            if video_files:
                selected_video_name = st.selectbox(
                    "Video mẫu Mendeley",
                    [p.name for p in video_files],
                    key=f"moto-video-{selected_video_class}",
                )
                sample_video_path = moto_video_classes[selected_video_class] / selected_video_name
                sample_video_label = selected_video_class
                st.caption(
                    "Video Mendeley motorcycle · dữ liệu ngoài miền công trường. "
                    "With/Without Helmet là nhãn ở cấp clip; không có bounding box theo khung hình. "
                    "Kết quả chỉ phục vụ xem xét định tính, không phải đánh giá định lượng trên tập test."
                )
            else:
                st.warning("Không tìm thấy video MP4 trong thư mục lớp đã chọn.")
        else:
            st.caption("Chưa có thư mục video mẫu Mendeley motorcycle trong dự án.")

    has_input = uploaded is not None or sample_path is not None or sample_video_path is not None
    if has_input:
        if input_kind == "Ảnh":
            image = Image.open(uploaded if uploaded is not None else sample_path).convert("RGB")
            caption = "Ảnh tải lên" if uploaded is not None else f"{sample_source_label}: {sample_path.name}"
            st.image(image, caption=caption, width="stretch")
        elif uploaded is not None:
            st.video(uploaded)
        else:
            # Do not pass a large local dataset file to st.video: Streamlit's
            # media manager loads local paths into memory. Preview one decoded
            # frame instead, then let OpenCV read the source path directly.
            try:
                import cv2

                preview = cv2.VideoCapture(str(sample_video_path))
                ok, frame = preview.read()
                preview.release()
                if ok:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Video mẫu · {sample_video_label} · {sample_video_path.name} (khung hình đầu)", width="stretch")
                else:
                    st.warning("Không đọc được khung hình đầu của video mẫu.")
            except Exception as exc:
                st.warning(f"Không thể tạo ảnh xem trước video: {exc}")

        checkpoint = None
        if weights_upload:
            digest = hashlib.sha256(weights_upload.getvalue()).hexdigest()[:20]
            checkpoint = Path(tempfile.gettempdir()) / f"helmet_demo_{digest}.pt"
            checkpoint.write_bytes(weights_upload.getvalue())
        elif model_choice != "Chưa có checkpoint":
            checkpoint = WEIGHTS_DIR / model_choice

        run = st.button("Chạy phát hiện", type="primary", width="stretch")
        if run:
            if not checkpoint:
                st.info("Để chạy phát hiện, hãy đặt checkpoint YOLOv8 vào thư mục `weights/` hoặc tải tệp .pt ở cột Thiết lập.")
            elif not has_cv2 or not has_ultralytics:
                st.error("Chưa cài đủ thư viện suy luận. Chạy `pip install -r requirements.txt` rồi khởi động lại ứng dụng.")
            else:
                try:
                    import cv2
                    import imageio_ffmpeg
                    from ultralytics import YOLO

                    @st.cache_resource
                    def load_model(path: str):
                        return YOLO(path)

                    model = load_model(str(checkpoint))
                    if input_kind == "Ảnh":
                        import numpy as np
                        import pandas as pd

                        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        result = model.predict(frame, conf=confidence, verbose=False)[0]
                        rendered = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                        st.image(rendered, caption="Kết quả YOLOv8", width="stretch")
                        boxes = result.boxes
                        rows = []
                        if boxes is not None:
                            for box in boxes:
                                class_id = int(box.cls[0].item())
                                rows.append({"Lớp": result.names[class_id], "Confidence": round(float(box.conf[0].item()), 3)})
                        a, b = st.columns(2)
                        a.metric("Phát hiện", len(rows))
                        b.metric("Thời gian suy luận", f"{result.speed.get('inference', 0):.1f} ms/ảnh")
                        if rows:
                            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                        else:
                            st.caption("Không phát hiện đối tượng trên ngưỡng đã chọn.")
                    else:
                        if uploaded is not None:
                            suffix = Path(uploaded.name).suffix or ".mp4"
                        else:
                            suffix = sample_video_path.suffix or ".mp4"
                        with tempfile.TemporaryDirectory(prefix="helmet_video_") as work:
                            if uploaded is not None:
                                src = Path(work) / ("input" + suffix)
                            else:
                                src = sample_video_path
                            dst = Path(work) / "result.mp4"
                            if uploaded is not None:
                                src.write_bytes(uploaded.getvalue())
                            cap = cv2.VideoCapture(str(src))
                            if not cap.isOpened():
                                st.error("Không đọc được video. Thử dùng MP4 H.264.")
                            else:
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps_in = cap.get(cv2.CAP_PROP_FPS)
                                if not math.isfinite(fps_in) or fps_in <= 0:
                                    fps_in = 25.0
                                if width <= 0 or height <= 0:
                                    cap.release()
                                    st.error("Video không có kích thước khung hình hợp lệ.")
                                    st.stop()
                                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))
                                if not writer.isOpened():
                                    cap.release()
                                    st.error("Không khởi tạo được bộ ghi video đầu ra.")
                                    st.stop()
                                count = detections = 0
                                start = time.perf_counter()
                                progress = st.progress(0, text="Đang xử lý video…")
                                total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                                try:
                                    while True:
                                        ok, frame = cap.read()
                                        if not ok:
                                            break
                                        result = model.predict(frame, conf=confidence, verbose=False)[0]
                                        detections += len(result.boxes) if result.boxes is not None else 0
                                        writer.write(result.plot())
                                        count += 1
                                        if count % 10 == 0:
                                            progress.progress(min(count / total, 1.0), text=f"Đang xử lý khung hình {count}/{total}…")
                                finally:
                                    cap.release()
                                    writer.release()
                                if count == 0:
                                    st.error("Video không có khung hình đọc được.")
                                    st.stop()
                                elapsed = max(time.perf_counter() - start, 1e-6)
                                progress.progress(1.0, text="Đã xử lý xong")
                                browser_video = Path(work) / "result_h264.mp4"
                                conversion = subprocess.run(
                                    [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(dst), "-an", "-c:v", "libx264",
                                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(browser_video)],
                                    capture_output=True, text=True, check=False,
                                )
                                if conversion.returncode != 0 or not browser_video.is_file():
                                    st.error("Không chuyển được video sang định dạng H.264.")
                                    st.code(conversion.stderr[-1200:])
                                    st.stop()
                                output_bytes = browser_video.read_bytes()
                                st.video(output_bytes)
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Khung hình", count)
                                m2.metric("Phát hiện cộng dồn", detections)
                                m3.metric("FPS toàn pipeline", f"{count / elapsed:.2f}")
                                st.download_button("Tải video kết quả", output_bytes, file_name="helmet_detection_result.mp4", mime="video/mp4")
                except Exception as exc:
                    st.exception(exc)
    else:
        if input_kind == "Ảnh" and sample_images:
            st.info("Chọn ảnh validation mẫu hoặc tải ảnh riêng lên để bắt đầu.")
        else:
            st.info("Tải ảnh hoặc video lên để bắt đầu.")

st.divider()
st.caption("Chế độ suy luận yêu cầu checkpoint đã huấn luyện và các thư viện trong requirements.txt. Kết quả phụ thuộc chất lượng dữ liệu, nhãn và điều kiện ghi hình.")
