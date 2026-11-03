# Kế hoạch huấn luyện lại Helmet Detection

## Mục tiêu

Phát hiện người đang đội mũ bảo hộ, gồm mũ bảo hộ công trường và mũ bảo hộ xe máy; giảm nhầm mũ lưỡi trai/mũ vải thành mũ bảo hộ. Giữ hai lớp: `wearing_helmet` và `not_wearing_helmet`. Mũ thường thuộc lớp `not_wearing_helmet`; mũ bảo hộ đặt riêng không được tính là người đang đội mũ.

## Phạm vi dữ liệu

| Nguồn | Tổng ảnh | Cách xử lý |
|---|---:|---|
| SHWD + Kaggle hiện có | 12.580 | Dùng nhãn hai lớp hiện tại. Ghi lại split gốc trong manifest, sau đó chia lại theo nhóm để gỡ rò rỉ ảnh trùng giữa train/validation/test. |
| DeBoniz | 18.157 | Nguồn nhiều miền gồm mũ bảo hộ công trường và mũ bảo hộ xe máy. Đưa mọi ảnh và nhãn hợp lệ vào bộ dữ liệu; giữ nguyên `Helmet` là lớp dương và `No-Helmet` là lớp âm. Chuyển đủ 44 dòng đa giác sang hộp giới hạn chữ nhật chuẩn hóa. Gom các khung hình có cùng mã `mp4-NNN` vào một nhóm split; báo riêng kết quả DeBoniz để thấy khác biệt miền dữ liệu. |
| Mendeley PPE | 2.286 | Ánh xạ `Helmet` → lớp 0 và `NoHelmet` → lớp 1; bỏ hộp `Vest`/`NoVest` khỏi nhãn nhị phân. |
| **Tổng ảnh tĩnh** | **33.023** | Ghi đủ mọi ảnh nguồn vào `manifest.csv`, kể cả ảnh đang chờ nhãn. |

Trong Mendeley có **483 ảnh không có hộp `Helmet` hoặc `NoHelmet`**. Đưa chúng vào `pending_review`, không đưa vào YAML train/validation/test và tuyệt đối không tạo nhãn rỗng để coi là nền. Ảnh `00751` được người dùng xác nhận có mũ bảo hộ màu đỏ nhưng nhãn nguồn chỉ có hộp áo; giữ nguyên nhãn nguồn, đánh `pending_manual_label` và cần bổ sung hộp đầu/mũ đầy đủ trước khi dùng.

Ảnh DeBoniz thuộc clip **`mp4-266`** là ví dụ mũ lưỡi trai thường, không phải mũ bảo hộ. Giữ nguyên nhãn `No-Helmet`, giữ toàn bộ nhóm clip ở test để kiểm tra lỗi dương tính giả. Không sửa nhãn gốc. Mũ bảo hộ xe máy trong các ảnh/clip khác vẫn là mũ bảo hộ; không đổi nhãn `Helmet` thành âm chỉ vì đó không phải mũ công trường.

Có **32 video Mendeley với 114.640 khung hình**. Chạy suy luận chẩn đoán trên toàn bộ video sau huấn luyện và xem kết quả theo clip; nhãn hiện chỉ ở cấp clip nên không dùng video để huấn luyện phát hiện hộp hoặc tính mAP theo khung hình. Muốn đưa khung hình vào supervised training cần người rà và gán hộp mũ bảo hộ/đầu người.

## Chia tập và kiểm soát nhãn

Tạo bộ dữ liệu riêng, không sửa `data/processed_combined` hay nhãn nguồn. Chia ảnh đủ nhãn theo tỷ lệ xấp xỉ **70/15/15**, seed 42, cân bằng theo nhóm lớp. Ảnh có cùng SHA-256 phải nằm cùng split; các khung DeBoniz cùng mã `mp4-NNN` cũng phải nằm cùng split. Manifest ghi nguồn, split ban đầu, hash, group, nhãn chuyển đổi, trạng thái đủ/chờ nhãn và split mới. Giữ mỗi ảnh đủ nhãn trong tập được gán; không bỏ ảnh trùng khỏi manifest.

SHA-256 chỉ tìm file trùng byte. Ảnh resize, crop hoặc nén lại cần perceptual-hash/soát thủ công trước khi chốt kết quả cuối. Dùng test mới cho so sánh A/B vì split core được chia lại; không diễn giải lại điểm test cũ như thể cùng một tập.

## Huấn luyện và đánh giá

1. Bộ dữ liệu đã chuẩn bị trong `data/retraining_all_stills`: **32.540 ảnh đủ nhãn** được chia train/validation/test; **483 ảnh thiếu box Helmet/NoHelmet** nằm trong `pending_review`, không bị gán nhãn nền. Hai lượt pilot bị ngắt không tham gia chọn mô hình.
2. Lượt huấn luyện đầy đủ `helmet_20261103_155304_yolov8m` đã hoàn tất **10/10 epoch**, exit code 0: khởi tạo từ stock `yolov8m.pt`, `imgsz=512`, `batch=8`, `workers=0`, seed 42, mosaic 0, patience 7, fraction 1.0, RTX 3050 Laptop 4 GB. Validation tốt nhất ở epoch 10: precision 0,9034; recall 0,8667; mAP50 0,9210; mAP50–95 0,5859.
3. Test khóa chung gồm 4.882 ảnh: precision 0,9032; recall 0,8686; mAP50 0,9201; mAP50–95 0,5895. DeBoniz được báo riêng trên 2.700 ảnh test theo nhóm: precision 0,9583; recall 0,9610; mAP50 0,9866; mAP50–95 0,7696. Phân tích ở confidence 0,25/IoU 0,5 cho recall 0,9177 (`wearing_helmet`) và 0,8909 (`not_wearing_helmet`); với box nhỏ hơn 1% diện tích ảnh lần lượt là 0,8276 và 0,8778.
4. Hai hard negative `mp4-266` vẫn ở test và giờ được dự đoán `not_wearing_helmet` ở confidence 0,83/0,87. Ảnh PPE `00751` vẫn chờ gán nhãn vì nhãn nguồn chỉ có box áo; 11 dự đoán mũ của mô hình chỉ là đánh giá định tính. Đã suy luận đủ 114.640/114.640 frame từ 32 video, 0 mismatch; video không có box nhãn theo frame nên không tính mAP từ chúng.
5. Checkpoint ứng viên `weights/helmet_retrained.pt` đã được đưa vào dropdown ứng dụng và smoke test thành công (14 detection, 22,2 ms/ảnh). **`weights/helmet_best.pt` vẫn là mặc định**, không bị ghi đè.

Chi tiết metric, hard-case, video, checkpoint và app smoke test nằm trong `results/retraining_run_log.md` và báo cáo kết quả `BAO_CAO_KET_QUA_HUAN_LUYEN_LAI_HELMET.md`.

**Giới hạn:** 483 ảnh Mendeley vẫn cần gán box mũ, gồm `00751`; nhãn polygon đổi thành hộp chữ nhật có thể chứa thêm nền; tập không có nhãn che khuất riêng nên recall đối tượng bị khuất không thể tính trực tiếp; video chỉ có nhãn cấp clip và kết quả frame-level là chẩn đoán. Checkpoint mới là lựa chọn riêng trong app, chưa thay model mặc định.
