# Báo cáo kết quả huấn luyện lại Helmet Detection với YOLOv8

**Ngày hoàn tất:** 03/11/2026  
**Run:** `helmet_20261103_155304_yolov8m`  
**Trạng thái:** Huấn luyện, đánh giá, chẩn đoán video và kiểm tra app hoàn tất. Checkpoint mới khả dụng trong app; model mặc định vẫn được giữ nguyên.

## 1. Điều phối nhóm 4 vai trò

| Vai trò | Phạm vi phụ trách | Kết quả |
|---|---|---|
| Leader (Codex) | Điều phối; kiểm soát split; huấn luyện GPU; đánh giá test; tổng hợp và kiểm tra app | Hoàn thành run 10 epoch, test chung, test DeBoniz, phân tích lỗi, checkpoint và smoke test app |
| Agent dữ liệu | Kiểm kê ảnh/nhãn/video, nhóm trùng lặp và xác minh dữ liệu staging DeBoniz | Manifest đầy đủ; 4.093 ảnh staging khớp SHA-256 với bản canonical; không thiếu hoặc lệch file |
| Agent cấu hình/ứng dụng | Chuẩn bị dataset YAML, grouped split, run log; kiểm tra cách app nạp model; chuẩn bị DeBoniz-only test | 32.540 ảnh đủ nhãn trong split mới; cấu hình đánh giá DeBoniz 2.700 ảnh; app nhận checkpoint mới qua dropdown |
| Agent video/hard case | Công cụ suy luận hard case; batch hóa chẩn đoán video và kiểm tra đếm frame | Hai ảnh mũ thường được phân loại âm; pipeline chạy đủ 114.640 frame theo lô 16 |

## 2. Phạm vi và kiểm soát dữ liệu

- Manifest bao phủ **33.023 ảnh tĩnh canonical**: 12.580 ảnh SHWD/Kaggle hiện có, 18.157 ảnh DeBoniz và 2.286 ảnh Mendeley PPE.
- **32.540 ảnh đủ nhãn mũ** được đưa vào train/validation/test: 22.776 / 4.882 / 4.882.
- **483 ảnh PPE** thiếu box `Helmet`/`NoHelmet` được giữ trong `pending_review`; không biến thành ảnh nền âm giả. `00751` thuộc hàng chờ này vì nhãn nguồn chỉ có box áo.
- Chia theo nhóm SHA-256 và clip; không có nhóm trùng lặp rò rỉ giữa train/validation/test. Có 364 nhóm ảnh trùng; staging copy 4.093 ảnh DeBoniz được xác minh byte-đồng-nhất nên không được tính thêm lần thứ hai.
- 44 dòng polygon DeBoniz được chuyển thành box bao quanh, giữ nhãn nguồn và ghi lại chuyển đổi. Không sửa ảnh hoặc nhãn nguồn.
- Video: **32 clip, 114.640 frame index**. Clip chỉ có nhãn `With Helmet`/`Without Helmet` ở cấp clip, không có ground-truth box theo frame.

## 3. Huấn luyện

| Thuộc tính | Giá trị |
|---|---|
| Mô hình khởi tạo | Stock `yolov8m.pt` |
| Epoch | 10/10, exit code 0 |
| Image size / batch | 512 / 8 |
| GPU | RTX 3050 Laptop, 4 GB |
| Seed / workers / mosaic | 42 / 0 / 0 |
| Training fraction | 1.0 (toàn bộ train split đủ nhãn) |
| Thời gian tổng ghi trong run | 11.664 giây, khoảng 3 giờ 14 phút |

Validation tăng từ mAP50 **0,6790** ở epoch 1 lên **0,9210** ở epoch 10. Đây là metric validation, không thay thế điểm test.

## 4. Kết quả test khóa

### Test chung: 4.882 ảnh

| Lớp | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| `wearing_helmet` | 0,8994 | 0,8878 | 0,9395 | 0,6985 |
| `not_wearing_helmet` | 0,9070 | 0,8494 | 0,9007 | 0,4805 |
| **Tổng** | **0,9032** | **0,8686** | **0,9201** | **0,5895** |

Inference benchmark trong evaluator: 19,1 ms/ảnh, khoảng 52,4 ảnh/giây. Con số khác đôi chút với phép đo giao diện app.

### Test riêng DeBoniz: 2.700 ảnh

Danh sách này dùng `generated_split=test`, giữ nguyên group split và gồm cả hai hard negative `mp4-266`; không rò rỉ nhóm sang split khác.

| Lớp | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| `wearing_helmet` | 0,9529 | 0,9653 | 0,9889 | 0,8307 |
| `not_wearing_helmet` | 0,9637 | 0,9568 | 0,9843 | 0,7085 |
| **Tổng** | **0,9583** | **0,9610** | **0,9866** | **0,7696** |

### Kiểm tra lỗi theo kích thước và độ sáng

Đo ở confidence 0,25 và IoU match 0,5 trên toàn bộ 4.882 ảnh test:

- Recall tổng theo ngưỡng: `wearing_helmet` 0,9177 (6.422 TP, 576 FN); `not_wearing_helmet` 0,8909 (17.749 TP, 2.173 FN).
- Với box nhỏ hơn 1% diện tích ảnh: recall `wearing_helmet` 0,8276; `not_wearing_helmet` 0,8778.
- Box 1–4%: recall lần lượt 0,9664 và 0,9618. Box từ 4% trở lên: 0,9898 và 0,9783.
- Trong nhóm ảnh tối (`mean luminance < 85`), recall `wearing_helmet` 0,9355 và `not_wearing_helmet` 0,8650.
- Dữ liệu không có nhãn occlusion riêng; vì vậy không thể tính chỉ số recall riêng cho đối tượng bị che khuất.

## 5. Các lỗi người dùng đã chỉ ra

- **Hai ảnh mũ lưỡi trai `mp4-266`:** người dùng xác nhận đây là mũ thường, không phải mũ bảo hộ. Model mới dự đoán `not_wearing_helmet` với confidence **0,8268** và **0,8742**. Hai ảnh giữ nhãn âm và nằm trong test DeBoniz.
- **Ảnh PPE `00751`:** model đưa ra 11 dự đoán `wearing_helmet` ở kiểm tra định tính. Ảnh không có box Helmet/NoHelmet nguồn nên không chấm được đúng/sai; không dùng làm nhãn supervised cho đến khi được gán box đầu/mũ.

## 6. Chẩn đoán video

- Hoàn tất **32/32 clip và 114.640/114.640 frame**, **0 frame-count mismatch**.
- Model phát ra 126.208 box; 113.963 frame có ít nhất một detection. Đây là thống kê dự đoán, không phải supervised metric.
- Kết quả được ghi theo thứ tự video/frame trong JSONL; không tạo annotated MP4 để tránh tốn đĩa.

## 7. Checkpoint và ứng dụng

- Checkpoint: [`weights/helmet_retrained.pt`](weights/helmet_retrained.pt), SHA-256 `de964a9c0188979430830163614ace8ea553e0cb565c06c5231dd60c87e0d64b`; khớp byte với run `best.pt`.
- Ứng dụng đã mở thành công tại `http://127.0.0.1:8502/`; dropdown hiển thị `helmet_retrained.pt`.
- Smoke test trên ảnh `kaggle_hard_hat_workers0.png`: app trả **14 detection**, độ trễ YOLO **22,2 ms/ảnh**, có cả hai lớp.
- **`weights/helmet_best.pt` vẫn là model mặc định**, không bị ghi đè. Checkpoint mới được thêm như một lựa chọn riêng để so sánh/đổi thủ công.

## 8. Tệp và kết quả chi tiết

- Kế hoạch và phạm vi: [`KE_HOACH_HUAN_LUYEN_LAI_HELMET.md`](KE_HOACH_HUAN_LUYEN_LAI_HELMET.md)
- Run log: [`results/retraining_run_log.md`](results/retraining_run_log.md)
- Dataset manifest: [`data/retraining_all_stills/manifest.csv`](data/retraining_all_stills/manifest.csv)
- Data config: [`configs/data_retrain_all_stills.yaml`](configs/data_retrain_all_stills.yaml)
- Test chung, metrics và plots: [`results/helmet_20261103_155304_yolov8m_test_20261103_191040/`](results/helmet_20261103_155304_yolov8m_test_20261103_191040/)
- Test DeBoniz riêng: [`results/helmet_20261103_155304_yolov8m_test_20261103_194347/`](results/helmet_20261103_155304_yolov8m_test_20261103_194347/)
- Phân tích lỗi và ảnh lỗi tiêu biểu: [`results/error_analysis/best_20261103_194529/`](results/error_analysis/best_20261103_194529/)
- Hard-case JSON/previews: [`results/retraining_hard_cases/20261103T121336_247960Z/`](results/retraining_hard_cases/20261103T121336_247960Z/)
- Video detections và summary: [`results/motorcycle_video_diagnostic/20261103T121343_648265Z/`](results/motorcycle_video_diagnostic/20261103T121343_648265Z/)
- Orchestration summary/logs: [`results/retraining_post_eval/helmet_20261103_155304_yolov8m_20261103_191027_485/`](results/retraining_post_eval/helmet_20261103_155304_yolov8m_20261103_191027_485/)

## 9. Giới hạn và hướng tiếp theo

1. Gán box Helmet/NoHelmet cho 483 ảnh PPE đang chờ, ưu tiên `00751`, rồi mới thêm chúng vào một vòng fine-tune tiếp theo.
2. Tập trung cải thiện box nhỏ: recall hiện thấp hơn rõ rệt dưới ngưỡng 1% diện tích ảnh.
3. Nếu cần metric video, hãy gán bounding box cho một tập frame được lấy mẫu thủ công; không suy diễn precision/recall từ clip label.
4. Không so sánh trực tiếp với checkpoint cũ như A/B chính thức vì dữ liệu core đã được chia lại và checkpoint cũ có thể đã thấy một phần ảnh test mới.

