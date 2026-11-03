# Helmet Detection with YOLOv8

Ứng dụng Streamlit phân tích ảnh/video công trường bằng YOLOv8. Dự án có pipeline lấy dữ liệu, chuẩn hóa nhãn, fine-tune YOLOv8n/s/m, đánh giá trên test và demo browser. Kế hoạch chi tiết: [KE_HOACH_DU_AN_HELMET_DETECTION.md](KE_HOACH_DU_AN_HELMET_DETECTION.md).

## Trạng thái

- Đã chuẩn hóa 12.580 ảnh từ SHWD và Kaggle: 8.955 train, 1.356 validation, 2.269 test. Ảnh kiểm tra nhãn: [combined_val_samples.jpg](results/data_audit/combined_val_samples.jpg).
- Đã huấn luyện YOLOv8n/s/m tối đa 10 epoch, cùng cấu hình 512 px, batch 1, seed 42. Bảng so sánh trên split test ban đầu nằm trong [BAO_CAO_KET_QUA.md](BAO_CAO_KET_QUA.md); kiểm tra checksum sau đó phát hiện 9 nhóm ảnh test trùng byte với train/validation, xem [báo cáo dữ liệu mở rộng](BAO_CAO_DATA_MO_RONG.md).
- Checkpoint demo là YOLOv8m best theo validation (epoch 8, mAP50–95 0,4389): [helmet_best.pt](weights/helmet_best.pt). Trên test đạt Precision 0,8691, Recall 0,6772, mAP50 0,7779, mAP50–95 0,4525; tốc độ suy luận 18,86 ms/ảnh trên RTX 3050 Laptop.
- Phân tích lỗi theo kích thước box/độ sáng: [summary.json](results/error_analysis/helmet_best_final_chunked/summary.json), [strata_metrics.csv](results/error_analysis/helmet_best_final_chunked/strata_metrics.csv) và [error_examples.jpg](results/error_analysis/helmet_best_final_chunked/error_examples.jpg).
- Đã kiểm thử trực tiếp app trong browser: ảnh validation cho 10 phát hiện; clip MP4 H.264 20 khung hình được xử lý thành công, video đầu ra phát được và có nút tải xuống. Clip QA là ảnh validation lặp lại để kiểm tra pipeline, không đại diện cho video tự nhiên.
- Đã bổ sung 20.443 ảnh gán nhãn và 32 video CC BY 4.0 vào `ext/`, giữ tách biệt khỏi benchmark gốc. App cho chọn ảnh mẫu từ DeBoniz/Mendeley PPE và video mẫu Mendeley motorcycle.
- Mở app tại `http://127.0.0.1:8502` khi server đang chạy. Xem [kế hoạch dự án](KE_HOACH_DU_AN_HELMET_DETECTION.md) và [báo cáo kết quả](BAO_CAO_KET_QUA.md).

## Môi trường

Khuyến nghị Python 3.12, PyTorch CUDA và batch nhỏ cho GPU RTX 3050 Laptop 4 GB. Thư viện và cache tạm đặt trên ổ E để tránh thiếu chỗ ổ hệ thống.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
New-Item -ItemType Directory .tmp,.pip-cache -Force | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = (Resolve-Path .pip-cache).Path
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Dữ liệu

### SHWD

SHWD lấy từ [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), định dạng Pascal VOC. Giữ split chính thức: 5.456 train, 607 validation, 1.517 test sau khi loại ảnh có ba box `dog` không thuộc bài toán. Ánh xạ `hat` → `wearing_helmet`, `person` → `not_wearing_helmet`.

```powershell
python scripts/prepare_shwd.py --source data/raw/VOC2028 --output data/processed --confirm-mapping
```

### Kaggle Hard Hat Detection

Nguồn [Safety Helmet Detection trên Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) có 5.000 ảnh, box Pascal VOC và ba lớp Helmet/Head/Person. Kaggle metadata ghi giấy phép CC0 Public Domain và ghi nhận Larxel/MakeML. Script tải dataset công khai, lưu metadata và SHA-256 vào `data/raw`:

```powershell
python scripts/download_kaggle_dataset.py
```

Mapping dùng trong bài: `helmet` → `wearing_helmet`, `head` → `not_wearing_helmet`; box `person` bị bỏ. Vì Kaggle không cung cấp split dùng cho bài này, script chia riêng các ảnh Kaggle theo sự hiện diện lớp với seed 42, xấp xỉ 70/15/15. Ảnh trùng pixel với SHWD bị loại; lần xử lý hiện tại không phát hiện ảnh trùng.

```powershell
python scripts/prepare_combined.py
```

Kết quả ở `data/processed_combined`; cấu hình YOLO là `configs/data_combined.yaml`. Manifest và thống kê được lưu trong `data/processed_combined/manifest.csv` và `dataset_report.json`. Hai nguồn có cách đặt box khác nhau: Kaggle box quanh helmet/head, còn SHWD giữ box nguồn với kích thước không đồng nhất. Nhãn biểu thị trạng thái theo annotation, không cam kết box là đường viền chính xác của riêng chiếc mũ.

### Dữ liệu mở rộng phục vụ phân tích

Kho `ext/` có 18.157 ảnh DeBoniz, 2.286 ảnh Mendeley PPE và 32 video Mendeley motorcycle. Nguồn video thuộc miền xe máy, có nhãn ở cấp clip chứ không có box từng frame; dùng cho phân tích định tính, không tính mAP detection. Các mẫu người dùng rà soát, gồm một false positive với mũ lưỡi trai thường và một ảnh PPE thiếu nhãn mũ, nằm trong `results/external_data_analysis/manual_review_cases.csv`. Xem [báo cáo dữ liệu mở rộng](BAO_CAO_DATA_MO_RONG.md), inventory tại `results/external_data_analysis/`, và cấu hình test ngoài miền [data_deboniz_external.yaml](configs/data_deboniz_external.yaml).

## Huấn luyện và đánh giá

Lệnh so sánh đủ ba cỡ mô hình trên cùng tập dữ liệu, seed, batch, kích thước ảnh và augmentation. Batch 1, ảnh 512 px, `workers=0`, `mosaic=0` và tối đa 10 epoch được chọn để phù hợp thời gian/VRAM của máy; patience 7 bật early stopping. Đây là cấu hình phần cứng, không phải yêu cầu cố định trong đề.

```powershell
python scripts/train.py --data configs/data_combined.yaml --models n s m --epochs 10 --imgsz 512 --batch 1 --device 0 --workers 0 --patience 7
```

Nếu phiên huấn luyện bị ngắt, khôi phục từ checkpoint `last.pt` của run đó:

```powershell
python scripts/resume_training.py --weights runs/detect/<run>/weights/last.pt
```

Đánh giá checkpoint đã chọn trên split test ban đầu (xem ghi chú về trùng lặp trong báo cáo):

```powershell
python scripts/evaluate.py --data configs/data_combined.yaml --weights weights/helmet_best.pt --device 0 --imgsz 512 --batch 1 --workers 0
```

Evaluator lưu Precision, Recall, mAP@0.5, mAP@0.5:0.95, thời gian suy luận, FPS suy luận và metric theo lớp vào thư mục `results/`. FPS suy luận không bao gồm thời gian đọc/ghi video; app báo thêm FPS toàn pipeline cho video.

Phân tích lỗi theo kích thước box và độ sáng trung bình của ảnh tại confidence 0,25, IoU matching 0,5:

```powershell
python scripts/analyze_errors.py --weights weights/helmet_best.pt --data configs/data_combined.yaml --device 0 --imgsz 512 --conf 0.25 --iou 0.5 --fp16
```

Lệnh lưu bảng phân tầng và ảnh lỗi tiêu biểu trong `results/error_analysis/`. Dataset không có nhãn che khuất chuyên biệt; trường hợp che khuất phải được đánh giá định tính qua ảnh lỗi và ghi rõ giới hạn này.

## Chạy app

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Mở URL local do Streamlit in ra. Đặt checkpoint `.pt` trong `weights/` hoặc nạp từ giao diện. Với ảnh, chọn mẫu validation của dự án, DeBoniz test hoặc Mendeley PPE valid, hoặc tải ảnh riêng. Với video, chọn clip Mendeley motorcycle từ ổ đĩa hoặc tải MP4/MOV/AVI/MKV. Video motorcycle chỉ dùng xem xét định tính ngoài miền công trường và không có nhãn box từng frame; video kết quả được mã hóa H.264 để phát trên browser.

## Giấy phép và giới hạn

Kaggle ghi CC0 cho Hard Hat Detection. SHWD repository có MIT, nhưng repository lưu ý ảnh có nguồn thu thập khác nhau; MIT của code không tự động xác nhận quyền phân phối mọi ảnh. Dữ liệu đã tải và checkpoint được loại khỏi Git. Ultralytics có các điều khoản cấp phép riêng; xem [Ultralytics License](https://www.ultralytics.com/license) trước khi sử dụng ngoài phạm vi học tập.

## Cấu hình cục bộ

Sau khi clone, chạy lệnh PowerShell sau ở thư mục dự án để tạo cấu hình từ các mẫu:

```powershell
Get-ChildItem configs -Filter '*.example.yaml' | ForEach-Object {
    $target = $_.FullName -replace '\.example\.yaml$', '.yaml'
    if (-not (Test-Path -LiteralPath $target)) {
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }
}
```

Dữ liệu, trọng số mô hình và kết quả chạy được lưu cục bộ. Tải và chuẩn bị dữ liệu theo hướng dẫn ở trên.
