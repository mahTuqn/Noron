# Báo cáo kết quả dự án Helmet Detection with YOLOv8

**Cập nhật:** 03/11/2026  
**Mục tiêu:** huấn luyện và so sánh YOLOv8n, YOLOv8s, YOLOv8m cho hai lớp `wearing_helmet` và `not_wearing_helmet`; đo kết quả trên split test được tách trong quy trình ban đầu và xác nhận ứng dụng xử lý được ảnh/video trong browser.

## 1. Tóm tắt

YOLOv8m đạt kết quả tốt nhất trong ba mô hình theo mAP50–95 trên validation (**0,4389**) và test (**0,4525**). Trên test, YOLOv8m đạt Precision **0,8691**, Recall **0,6772**, mAP50 **0,7779**. YOLOv8s đạt Precision cao nhất (**0,8726**) và tốc độ suy luận đo trong validator cao hơn nhẹ; YOLOv8m cho Recall và mAP cao nhất.

Lớp `not_wearing_helmet` khó phát hiện hơn: YOLOv8m đạt Recall **0,5896** trên test theo metric validator; ở ngưỡng confidence cố định 0,25, phân tích lỗi ghi nhận Recall **0,4163**. Lỗi tập trung mạnh vào box nhỏ. Browser QA xác nhận luồng suy luận ảnh và luồng nạp/xử lý/xuất video H.264 hoạt động. Video QA được ghép lặp từ ảnh validation, nên chỉ xác nhận pipeline video, không đại diện cho hiệu năng trên video công trường tự nhiên.

## 2. Dữ liệu và phạm vi đánh giá

Tập dữ liệu kết hợp có **12.580 ảnh**: 8.955 train, 1.356 validation và 2.269 test. SHWD giữ split nguồn; ảnh Kaggle được chia theo sự hiện diện lớp với seed 42, tỷ lệ xấp xỉ 70/15/15. Không phát hiện ảnh trùng pixel giữa hai nguồn trong lượt chuẩn bị này.

Ánh xạ lớp:

- SHWD: `hat` → `wearing_helmet`; `person` → `not_wearing_helmet`.
- Kaggle Hard Hat Detection: `Helmet` → `wearing_helmet`; `Head` → `not_wearing_helmet`; box `Person` bị bỏ qua.
- Tổng thống kê nguồn sau mapping: 27.997 box đội mũ và 117.299 box không đội mũ. Tập test riêng có 4.544 box đội mũ và 23.396 box không đội mũ.

Hai nguồn có quy ước box không hoàn toàn đồng nhất: Kaggle vẽ box quanh helmet/head; SHWD giữ độ rộng box như annotation nguồn. Vì vậy, kết quả phản ánh chất lượng dự đoán theo annotation kết hợp đã chuẩn hóa, không chứng minh box luôn trùng đường viền riêng của chiếc mũ.

Kiểm tra checksum SHA-256 mở rộng sau này phát hiện **9 nhóm ảnh test có bản sao byte-identical trong train hoặc validation** của bộ dữ liệu kết hợp gốc. Các số liệu dưới đây vẫn là kết quả đo trên split test đã dùng, nhưng test không hoàn toàn độc lập theo kiểm tra trùng file tuyệt đối. Chi tiết nằm trong [báo cáo mở rộng dữ liệu](BAO_CAO_DATA_MO_RONG.md) và `results/external_data_analysis/core_test_exact_duplicates.csv`; metric chưa được tính lại trên split đã khử trùng.

Cấu hình dữ liệu nằm tại [`configs/data_combined.yaml`](configs/data_combined.yaml); báo cáo chuẩn bị dữ liệu tại [`data/processed_combined/dataset_report.json`](data/processed_combined/dataset_report.json). Ảnh audit nhãn: [`results/data_audit/combined_val_samples.jpg`](results/data_audit/combined_val_samples.jpg).

## 3. Cấu hình huấn luyện

Ba cỡ mô hình dùng cùng cấu hình và toàn bộ train split:

| Tham số | Giá trị |
|---|---:|
| Kiến trúc | YOLOv8n, YOLOv8s, YOLOv8m pretrained |
| Epoch tối đa | 10 |
| Image size | 512 px |
| Batch | 1 |
| Device | CUDA 0, NVIDIA RTX 3050 Laptop 4 GB |
| Seed | 42 |
| Workers | 0 |
| Patience | 7 |
| Fraction | 1,0 (toàn bộ train split) |
| Mosaic | 0 (tắt) |
| Dataset YAML | `configs/data_combined.yaml` |

Thiết lập batch 1, ảnh 512 px và workers 0 được chọn để phù hợp giới hạn bộ nhớ máy. Một số lượt chạy cỡ m được tiếp tục từ checkpoint sau khi tiến trình bị ngắt; optimizer và epoch được khôi phục từ `last.pt`. Các run huấn luyện chính: `runs/detect/helmet_20261103_021400_yolov8n`, `...yolov8s`, `...yolov8m`.

### Kết quả validation tốt nhất

Metric theo epoch có mAP50–95 cao nhất trong `results.csv`; đây là căn cứ chọn checkpoint trước khi mở tập test.

| Mô hình | Epoch tốt nhất | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 8 | 0,82985 | 0,57795 | 0,69154 | 0,38286 |
| YOLOv8s | 10 | 0,85660 | 0,59366 | 0,71616 | 0,40860 |
| YOLOv8m | 8 | 0,86806 | 0,68134 | 0,78182 | 0,43887 |

YOLOv8m epoch 10 đạt Precision 0,88529, Recall 0,62262, mAP50 0,75118 và mAP50–95 0,42796. Checkpoint validation tốt nhất vẫn là epoch 8 vì cả hai chỉ số mAP ở epoch 10 thấp hơn. Trọng số đã chọn được lưu tại `weights/helmet_yolov8n_best.pt`, `weights/helmet_yolov8s_best.pt`, `weights/helmet_yolov8m_best.pt`.

## 4. Đánh giá trên split test ban đầu

Mỗi mô hình được đánh giá trên cùng 2.269 ảnh test ở 512 px. Precision/Recall và mAP dưới đây lấy từ validator Ultralytics; thời gian/FPS là tốc độ suy luận model đo trong validator, không gồm thời gian đọc/ghi video hay dựng giao diện.

| Mô hình | Precision | Recall | mAP50 | mAP50–95 | ms/ảnh | FPS suy luận |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0,84669 | 0,57598 | 0,69792 | 0,39504 | 14,23 | 70,29 |
| YOLOv8s | 0,87264 | 0,59215 | 0,72519 | 0,42498 | 14,12 | 70,83 |
| YOLOv8m | **0,86907** | **0,67717** | **0,77787** | **0,45248** | 18,86 | 53,03 |

YOLOv8m cải thiện mAP50–95 khoảng 0,0275 tuyệt đối so với YOLOv8s và 0,0574 so với YOLOv8n, đồng thời có Recall tổng cao nhất. YOLOv8s có Precision cao hơn YOLOv8m khoảng 0,0036 và throughput đo được cao hơn; lựa chọn phụ thuộc ưu tiên Recall/chất lượng tổng thể hay tốc độ.

### Kết quả theo lớp của YOLOv8m

| Lớp | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| `wearing_helmet` | 0,91090 | 0,76474 | 0,84655 | 0,56632 |
| `not_wearing_helmet` | 0,82724 | 0,58959 | 0,70919 | 0,33864 |

Lớp không đội mũ vẫn yếu hơn rõ rệt, đặc biệt ở mAP50–95. Trong demo giám sát an toàn, không nên xem một dự đoán “có mũ” là bảo đảm; cần kiểm tra Recall lớp không đội mũ và đánh giá thêm trên camera/vị trí thực tế trước khi dùng cho quyết định an toàn.

Artifacts test:

- YOLOv8n: [`metrics_summary.json`](results/_test_20261103_104052/metrics_summary.json), [`per_class_metrics.csv`](results/_test_20261103_104052/per_class_metrics.csv), confusion matrix và PR/P/R/F1 curves trong cùng thư mục.
- YOLOv8s: [`metrics_summary.json`](results/_test_20261103_104224/metrics_summary.json), [`per_class_metrics.csv`](results/_test_20261103_104224/per_class_metrics.csv), plots trong cùng thư mục.
- YOLOv8m: [`metrics_summary.json`](results/_test_20261103_104348/metrics_summary.json), [`per_class_metrics.csv`](results/_test_20261103_104348/per_class_metrics.csv), plots trong cùng thư mục.

## 5. Phân tích lỗi của checkpoint YOLOv8m

Phân tích dùng `weights/helmet_best.pt` (bản sao checkpoint tốt nhất của YOLOv8m), 2.269 ảnh test, confidence **0,25**, matching IoU **0,5**, image size **512**. Đây là số đếm tại một ngưỡng cố định nên Recall ở phần này không nhất thiết bằng Recall của validator ở bảng test, vốn được tổng hợp theo operating point của validator.

| Lớp | TP | FN | FP | Recall tại conf=0,25 |
|---|---:|---:|---:|---:|
| `wearing_helmet` | 3.088 | 1.456 | 93 | 0,67958 |
| `not_wearing_helmet` | 9.739 | 13.657 | 456 | 0,41627 |

### Theo kích thước box

Ngưỡng phân nhóm: small `<1%` diện tích ảnh, medium `1–4%`, large `≥4%`. TP/FN được nhóm theo diện tích box ground truth; FP được nhóm theo diện tích box dự đoán sai. Do đó `ground_truth/TP/FN/Recall` là thống kê GT, còn FP trong cùng hàng là số dự đoán sai có kích thước thuộc nhóm đó.

| Lớp | Kích thước | Ground truth | TP | FN | FP | Recall |
|---|---|---:|---:|---:|---:|---:|
| `wearing_helmet` | small | 2.651 | 1.455 | 1.196 | 38 | 0,54885 |
| `wearing_helmet` | medium | 1.171 | 980 | 191 | 28 | 0,83689 |
| `wearing_helmet` | large | 722 | 653 | 69 | 27 | 0,90443 |
| `not_wearing_helmet` | small | 21.450 | 8.266 | 13.184 | 421 | 0,38536 |
| `not_wearing_helmet` | medium | 1.676 | 1.316 | 360 | 31 | 0,78520 |
| `not_wearing_helmet` | large | 270 | 157 | 113 | 4 | 0,58148 |

Điểm yếu nổi bật là box nhỏ: Recall chỉ **0,5489** cho mũ và **0,3854** cho đầu/người không đội mũ ở nhóm small. Phần lớn GT `not_wearing_helmet` trong tập test thuộc nhóm small (21.450/23.396); số FN small là 13.184. Điều này phù hợp với bối cảnh ảnh đông người, người ở xa hoặc vùng đầu nhỏ trong khung hình.

### Theo độ sáng trung bình

Độ sáng được tính theo luminance trung bình của ảnh: dark `<85`, mid `85–169`, bright `≥170` trên thang 0–255. Số lượng ảnh ở các nhóm không cân bằng; không diễn giải chênh lệch thành quan hệ nhân quả.

| Lớp | Độ sáng | Ground truth | TP | FN | FP | Recall |
|---|---|---:|---:|---:|---:|---:|
| `wearing_helmet` | dark | 361 | 201 | 160 | 14 | 0,55679 |
| `wearing_helmet` | mid | 3.630 | 2.466 | 1.164 | 70 | 0,67934 |
| `wearing_helmet` | bright | 553 | 421 | 132 | 9 | 0,76130 |
| `not_wearing_helmet` | dark | 1.031 | 583 | 448 | 37 | 0,56547 |
| `not_wearing_helmet` | mid | 22.132 | 9.018 | 13.114 | 410 | 0,40746 |
| `not_wearing_helmet` | bright | 233 | 138 | 95 | 9 | 0,59227 |

Artifacts phân tích lỗi: [`summary.json`](results/error_analysis/helmet_best_final_chunked/summary.json), [`strata_metrics.csv`](results/error_analysis/helmet_best_final_chunked/strata_metrics.csv), [`error_examples.jpg`](results/error_analysis/helmet_best_final_chunked/error_examples.jpg). Dataset không có nhãn che khuất riêng, vì vậy chưa thể đo Recall/FP theo mức che khuất; ảnh ví dụ chỉ hỗ trợ nhận xét định tính.

## 6. Kiểm tra ứng dụng trong browser

Ứng dụng Streamlit được mở và thao tác tại `http://127.0.0.1:8502`.

- **Ảnh:** chạy checkpoint trên một ảnh mẫu Kaggle; giao diện hiển thị **10 detections**, thời gian suy luận hiển thị **20,4 ms** cho ảnh đó.
- **Video:** chạy clip H.264 **20 frame**, độ phân giải 640×480, thời lượng 0,8 giây. Browser xác nhận video đầu vào và kết quả đầu ra đã nạp hết (`readyState=4`); ứng dụng xử lý **100 detections cộng dồn**, báo **21,65 FPS** theo pipeline và hiển thị nút tải video kết quả.
- Clip dùng trong QA được tạo bằng cách lặp bốn ảnh validation. Vì vậy, kiểm tra chứng minh luồng upload → inference → encode H.264 → phát/tải hoạt động; nó không đánh giá tracking, ổn định theo thời gian hoặc chất lượng trên video thực tế.

Artifacts clip: [`browser_qa_validation_clip.mp4`](results/browser_qa_validation_clip.mp4), [`browser_qa_validation_clip_h264.mp4`](results/browser_qa_validation_clip_h264.mp4).

## 7. Kết luận và giới hạn

Với tiêu chí mAP50–95 và Recall tổng, YOLOv8m là checkpoint phù hợp nhất trong ba kích thước được thử; YOLOv8s là lựa chọn nhanh hơn với Precision nhỉnh hơn đôi chút. Kết quả không đồng nghĩa hệ thống đã sẵn sàng ra quyết định an toàn ngoài thực địa. Recall lớp `not_wearing_helmet`, nhất là trên box nhỏ, còn thấp; ảnh đông người, người ở xa, che khuất, góc camera và khác biệt nguồn annotation là các trường hợp cần đánh giá thêm. Tập QA video chỉ là clip tổng hợp ngắn.

Các bước tiếp theo trước khi triển khai thực tế:

1. Thu thập và gán nhãn video/ảnh từ camera mục tiêu, bao gồm người ở xa và tình huống che khuất.
2. Tách test theo địa điểm/camera hoặc thời gian để đo khả năng khái quát; duy trì test độc lập cho mọi điều chỉnh ngưỡng.
3. Bổ sung dữ liệu lớp không đội mũ và box nhỏ; cân nhắc tăng độ phân giải hoặc dùng crop/tiling nếu tốc độ cho phép.
4. Chạy đánh giá video dài hơn trên thiết bị triển khai, đo độ trễ pipeline, tỷ lệ bỏ khung hình và độ ổn định theo thời gian.
5. Chọn ngưỡng confidence theo chi phí bỏ sót vi phạm an toàn; lưu lại phiên bản model, dữ liệu và cấu hình cho mỗi lần đánh giá.

## 8. Tái lập kết quả

Đánh giá test từng mô hình (chạy tuần tự):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --data configs\data_combined.yaml --weights weights\helmet_yolov8n_best.pt --device 0 --imgsz 512 --batch 1 --workers 0
.\.venv\Scripts\python.exe scripts\evaluate.py --data configs\data_combined.yaml --weights weights\helmet_yolov8s_best.pt --device 0 --imgsz 512 --batch 1 --workers 0
.\.venv\Scripts\python.exe scripts\evaluate.py --data configs\data_combined.yaml --weights weights\helmet_yolov8m_best.pt --device 0 --imgsz 512 --batch 1 --workers 0
```

Phân tích lỗi checkpoint m:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_errors.py --weights weights\helmet_best.pt --data configs\data_combined.yaml --device 0 --imgsz 512 --conf 0.25 --iou 0.5 --fp16 --outdir results\error_analysis\helmet_best_final_chunked
```

Log epoch nguồn: [`YOLOv8n results.csv`](runs/detect/helmet_20261103_021400_yolov8n/results.csv), [`YOLOv8s results.csv`](runs/detect/helmet_20261103_021400_yolov8s/results.csv), [`YOLOv8m results.csv`](runs/detect/helmet_20261103_021400_yolov8m/results.csv).
