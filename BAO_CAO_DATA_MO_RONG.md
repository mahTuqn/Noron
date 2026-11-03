# Báo cáo mở rộng dữ liệu Helmet Detection

**Ngày cập nhật:** 03/11/2026  
**Mục đích:** bổ sung ảnh và video để xem xét độ đa dạng, kiểm tra domain shift và dùng làm dữ liệu mẫu trong app. Các nguồn mới được để riêng; không đưa vào train/validation/test gốc và không huấn luyện lại mô hình trong bước mở rộng này.

## Tóm tắt

- Thêm **20.443 ảnh có nhãn YOLO** từ hai bộ dữ liệu có giấy phép CC BY 4.0.
- Thêm **32 video MP4**, tổng **63 phút 45 giây**, chia đều 16 clip `With Helmet` và 16 clip `Without Helmet`.
- Tạo kiểm kê nhãn, checksum, danh sách thông số video và ảnh xem trước trong `results/external_data_analysis/`.
- Bộ dữ liệu gốc `data/processed_combined/` không bị sửa. Ứng dụng có thể chọn ảnh mẫu từ hai nguồn mới và video mẫu trực tiếp từ ổ đĩa.

## Nguồn và nội dung

| Nguồn | Nội dung đếm từ tệp đã tải | Giấy phép và cách dùng |
|---|---|---|
| [DeBoniz Helmet Detection Dataset](https://huggingface.co/datasets/DeBoniz/helmet-detection-dataset) | 18.157 ảnh, 27.789 dòng bbox hợp lệ, lớp `Helmet` / `No-Helmet`; train 16.384, val 1.130, test 643 | CC BY 4.0. Nhiều ảnh giao thông, người đi xe máy, giám sát và một số ảnh công nghiệp/sản phẩm; dùng như benchmark ngoài miền hỗn hợp, không coi là tập công trường thuần. Dataset card ghi 18.159 ảnh và train 16.386, chênh 2 ảnh so với ZIP thực tế. |
| [Mendeley Dataset of Personal Protective Equipment](https://data.mendeley.com/datasets/zkzghjvpn2/6) | 2.286 ảnh 640×640, 6.038 box; train 1.829, valid 457. Box theo lớp: `Helmet` 2.421, `NoHelmet` 1.174, `NoVest` 1.058, `Vest` 1.385 | CC BY 4.0. Phù hợp xem xét bối cảnh PPE công trường/nhà xưởng; nguồn gốc ảnh gồm nhiều bộ public và ảnh thực tế theo mô tả tác giả. Hai lớp vest cần được bỏ hoặc ánh xạ riêng nếu chỉ đánh giá detector hai lớp mũ. |
| [Mendeley Dataset for the Detection of Helmet Use in Motorcycle Drivers](https://data.mendeley.com/datasets/bmy35m25pw/1) | 32 video, 114.640 khung hình nguồn, tổng 63 phút 45 giây, 1920×1080, 29,97 fps; 16 clip mỗi lớp | CC BY 4.0. Đây là video người lái xe máy ở góc nhìn trên xe, không phải công trường. Nhãn `With/Without Helmet` chỉ ở cấp clip; không có bounding box từng khung hình. Chỉ dùng để xem xét định tính/domain shift, không dùng để tính mAP detection. |

Tổng ảnh mới là **20.443**. DeBoniz có 16.233 dòng bbox hợp lệ `Helmet` và 11.556 `No-Helmet`. Trong 16.384 ảnh train, phát hiện thêm **44 dòng nhãn polygon-like không theo định dạng bbox YOLO 5 trường**; chúng không được tính vào số bbox hợp lệ và cần chuyển sang bbox hoặc loại khỏi train nếu dùng bộ này huấn luyện detect. Các clip là 16 cặp người lái p1–p16, mỗi người được quay một lần đội mũ và một lần không đội mũ. Độ dài min/median/max là 58,926 / 120,454 / 135,168 giây; kích thước tệp min/median/max khoảng 20,0 / 76,9 / 218,0 MiB. Tổng dung lượng video giải nén 2,662 GiB.

## Vị trí dữ liệu

- DeBoniz ZIP: `ext/deboniz_helmet/downloads/helmet-detection-dataset.zip`; split gốc đã giải nén tại `ext/deboniz_helmet/extracted/datasetmodelloscelto/dataset10kadjusted/`.
- Mendeley PPE ZIP: `ext/mendeley_ppe/downloads/mendeley_ppe.zip`; dữ liệu YOLO đã giải nén tại `ext/mendeley_ppe/extracted/`.
- Video ZIP: `ext/mendeley_motorcycle_video/downloads/mendeley_motorcycle_video.zip`; clip đã giải nén theo lớp tại `ext/mendeley_motorcycle_video/extracted/`.
- Attribution, URL tải và SHA-256 của từng ZIP: `ext/<source>/source.json`.
- Tệp dữ liệu nặng nằm trong `ext/`, được loại khỏi Git qua `.gitignore`.

## Kiểm tra trùng và giới hạn benchmark

Đã tính SHA-256 cho **33.023 ảnh** (20.443 ảnh mới và 12.580 ảnh benchmark gốc). Không có nhóm trùng byte giữa dữ liệu mới và benchmark gốc. DeBoniz có 327 nhóm trùng byte nội bộ, tất cả ở trong cùng split; không thấy nhóm trùng byte qua các split của DeBoniz. Tệp PPE mới không có nhóm trùng byte nội bộ. Đây chỉ là kiểm tra trùng file tuyệt đối, không bắt được ảnh đã resize, crop hoặc nén lại.

Kiểm tra này cũng phát hiện **9 nhóm trùng byte giữa ảnh test gốc với train/validation gốc**. Vì vậy, các metric cũ trong [báo cáo kết quả](BAO_CAO_KET_QUA.md) vẫn là kết quả đo trên đúng split đã dùng, nhưng không nên mô tả tập test đó là hoàn toàn độc lập sau kiểm tra trùng chính xác. Split và metric cũ được giữ nguyên trong lần bổ sung dữ liệu này; để có benchmark sạch hơn cần gom ảnh trùng trước khi chia split rồi huấn luyện/đánh giá lại.

Tệp chi tiết: `results/external_data_analysis/core_test_exact_duplicates.csv` và `results/external_data_analysis/summary.json`.

## Ví dụ người dùng rà soát thủ công

- Ảnh DeBoniz `mp4-266`: annotation nguồn ghi `No-Helmet`; người trong ảnh đội mũ lưỡi trai thường. Ảnh chụp app cho thấy mô hình dự đoán `wearing_helmet` với confidence 0,59, nên đây là false positive đã xác nhận. Giữ annotation gốc và dùng ảnh làm hard negative.
- Ảnh Mendeley PPE `00751`: người dùng chỉ ra có người đội mũ bảo hộ nhưng app dự đoán `not_wearing_helmet`. Annotation nguồn chỉ có ba box `Vest`, không có box `Helmet`/`NoHelmet`, nên đây là ứng viên false negative nhưng không thể chấm detection metric bằng nhãn hiện có. Không tự sửa nhãn nguồn; loại ảnh này khỏi metric riêng cho helmet nếu chưa gán nhãn thủ công.

Danh sách và đường dẫn hai mẫu: `results/external_data_analysis/manual_review_cases.csv`.

## Ảnh và video xem trước

Ảnh bên dưới có vẽ lại các bounding box từ nhãn nguồn. Bảng video hiển thị một khung hình giữa clip cho từng tệp để rà soát nhanh.

![Ảnh mẫu từ DeBoniz và Mendeley PPE](results/external_data_analysis/external_image_samples.jpg)

![Khung hình mẫu từ 32 video Mendeley](results/external_data_analysis/motorcycle_video_samples.jpg)

## Dùng trong ứng dụng

Trong app Streamlit tại `http://127.0.0.1:8502/`:

1. Chọn **Ảnh** rồi đặt `Nguồn ảnh mẫu` thành DeBoniz external test hoặc Mendeley PPE valid để xem/chạy suy luận trên tối đa 50 ảnh mẫu.
2. Chọn **Video** rồi chọn `Video mẫu Mendeley motorcycle`; app mở tệp gốc trực tiếp, không cần tải clip dung lượng lớn qua browser.
3. Khi xem video, nhớ nhãn clip-level và miền xe máy chỉ cho phân tích định tính. Kết quả không thay thế đánh giá detection có bounding box. Nếu muốn lấy mẫu tiết kiệm tài nguyên, bắt đầu với 4 cặp người lái (8 clip) và một frame mỗi 5 giây, khoảng 190–200 ảnh; không cần giải mã/lưu toàn bộ 114.640 frame.

## Tái lập kiểm kê

Tải lại nguồn và giải nén vào `ext/`:

```powershell
.venv\Scripts\python.exe scripts\download_analysis_datasets.py
```

Tạo lại thống kê, bảng inventory và preview:

```powershell
.venv\Scripts\python.exe scripts\analyze_external_data.py
.venv\Scripts\python.exe scripts\create_external_data_previews.py
```

DeBoniz có nhãn hai lớp tương ứng trực tiếp với model hiện tại; để tính metric riêng trên split test của nguồn đó, chạy lệnh dưới đây. Kết quả cần được giữ riêng khỏi metric benchmark gốc:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --data configs\data_deboniz_external.yaml --weights weights\helmet_best.pt --device 0 --imgsz 512 --batch 1 --workers 0
```

Mendeley PPE có bốn lớp, còn video chỉ có nhãn clip-level; muốn tính detection metrics trên hai nguồn này cần chuẩn hóa/mapping nhãn hoặc gán box theo từng frame trước.
