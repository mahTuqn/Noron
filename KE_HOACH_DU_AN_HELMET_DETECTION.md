# Kế hoạch thực hiện dự án Helmet Detection with YOLOv8

- **Môn học:** Mạng nơ-ron và Học sâu.
- **Nhóm đăng ký:** Nhóm 17.
- **Tên tiếng Việt:** Phát hiện mũ bảo hộ lao động bằng YOLOv8.
- **Ngày lập kế hoạch:** 03/11/2026.
- **Tài liệu giao đề:** `Helmet_Detection_with_YOLOv8_nguyen_van_tu_sheet.docx`.
- **Thời lượng đề xuất:** 6 tuần; điều chỉnh sau khi biết hạn nộp thực tế.
- **Trạng thái:** Kế hoạch kèm kết quả triển khai; kết quả thực nghiệm được cập nhật tại mục 19 và [báo cáo kết quả](BAO_CAO_KET_QUA.md).

## 1. Tổng quan

Dự án xây dựng mô hình phát hiện vùng mũ bảo hộ và vùng đầu không đội mũ trong ảnh, video công trường bằng YOLOv8. Nhóm fine-tune ba phiên bản YOLOv8n, YOLOv8s và YOLOv8m, so sánh độ chính xác với tốc độ, phân tích lỗi và xây dựng chương trình demo trên video.

Đầu ra chính là một quy trình có thể chạy lại từ chuẩn bị dữ liệu đến huấn luyện, đánh giá và demo, kèm báo cáo giải thích kết quả. Ưu tiên hoàn thành đầy đủ yêu cầu trong đề trước khi bổ sung cải tiến kiến trúc hoặc giao diện.

Các thông số như thời lượng, tỷ lệ chia dữ liệu, số epoch và quy mô mẫu phân tích trong kế hoạch này là đề xuất triển khai, không phải yêu cầu cứng do giảng viên cung cấp.

## 2. Mục tiêu và câu hỏi nghiên cứu

1. Huấn luyện mô hình nhận biết vùng có mũ bảo hộ và vùng đầu không đội mũ.
2. So sánh YOLOv8n, YOLOv8s và YOLOv8m trong cùng điều kiện đánh giá.
3. Đo đầy đủ mAP@0.5, mAP@0.5:0.95, Precision, Recall và FPS.
4. Phân tích ảnh hưởng của che khuất, vật thể nhỏ và thiếu sáng.
5. Lựa chọn mô hình phù hợp cho demo video dựa trên số liệu đo được.

Các câu hỏi cần trả lời trong báo cáo:

- Mỗi phiên bản đạt kết quả thế nào trên tập test độc lập?
- Mức tăng độ chính xác có tương xứng với chi phí tính toán và tốc độ giảm hay không?
- Lớp nào dễ bị bỏ sót hoặc nhận nhầm hơn?
- Những điều kiện nào làm mô hình suy giảm chất lượng?
- Mô hình nào phù hợp nhất với phần cứng và mục tiêu demo của nhóm?

## 3. Đối chiếu yêu cầu đề bài

| Yêu cầu trong tài liệu | Công việc thực hiện | Minh chứng bàn giao |
|---|---|---|
| Huấn luyện hoặc fine-tune YOLOv8 | Fine-tune từ trọng số pretrained trên dữ liệu mũ bảo hộ | Mã nguồn, cấu hình, log và checkpoint |
| So sánh các cỡ mô hình n/s/m | Huấn luyện cả ba phiên bản trên cùng cách chia dữ liệu | Bảng chỉ số và biểu đồ so sánh |
| Đánh giá mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS | Đánh giá test và đo tốc độ theo quy trình thống nhất | Kết quả tổng thể, theo lớp và mô tả cách đo |
| Phân tích che khuất, vật thể nhỏ, thiếu sáng | Khảo sát các nhóm tình huống khó | Thống kê, ví dụ đúng/sai và nhận xét |
| Demo trên video | Chạy dự đoán, hiển thị và lưu video kết quả | Chương trình demo, hướng dẫn và video |

Tài liệu giao đề không quy định ngưỡng mAP hoặc FPS tối thiểu. Nhóm phải báo cáo trung thực số đo thực tế, không tự đặt kết quả chưa kiểm chứng thành cam kết.

## 4. Phạm vi thực hiện

### 4.1. Phạm vi bắt buộc

- Phát hiện bằng bounding box trên ảnh và từng khung hình video.
- Dùng một bộ dữ liệu đã chuẩn hóa từ SHWD và Kaggle Hard Hat Detection, giữ train, validation và test độc lập theo nguồn.
- Kiểm tra và chuẩn hóa nhãn dữ liệu.
- Fine-tune YOLOv8n, YOLOv8s và YOLOv8m.
- Đánh giá đầy đủ năm chỉ số trong đề.
- Phân tích lỗi trong ba điều kiện được yêu cầu.
- Xây dựng demo xử lý video và xuất video kết quả.
- Hoàn thiện mã nguồn, README, báo cáo và slide thuyết trình.

### 4.2. Đơn vị phát hiện và ý nghĩa nhãn

Phương án cơ sở là phát hiện vùng mũ/đầu, thay vì phát hiện toàn thân công nhân rồi suy luận trạng thái mũ.

| Lớp dự kiến | Ý nghĩa | Điều kiện sử dụng |
|---|---|---|
| `wearing_helmet` | Đối tượng mang trạng thái đội mũ theo nhãn nguồn | SHWD `hat` và Kaggle `helmet` |
| `not_wearing_helmet` | Đối tượng mang trạng thái không đội mũ theo nhãn nguồn | SHWD `person` và Kaggle `head`; Kaggle `person` bị bỏ |

Đã kiểm tra trực quan các ảnh mẫu SHWD. Hầu hết box `hat` dương tính gần vùng đầu/mũ, nhưng có khoảng 0,55% box phủ trên một nửa ảnh; box `person` âm tính thường nhỏ hơn nhiều và nằm quanh đầu. Vì vậy class biểu thị trạng thái đội mũ của annotation nguồn, còn hình học box không hoàn toàn đồng nhất. Báo cáo phải nêu rõ hạn chế này và tránh mô tả box như ranh giới chính xác của riêng chiếc mũ. Ba box `dog` trong một ảnh bị loại cùng toàn bộ ảnh đó để không đưa vật thể không nhãn vào train/test.

Kaggle Hard Hat Detection có 5.000 ảnh Pascal VOC và ba nhãn `helmet`, `head`, `person`; nhãn được ánh xạ lần lượt sang đội mũ, không đội mũ và bỏ qua `person`. Do Kaggle không có split công bố, các ảnh được chia theo sự hiện diện lớp với seed 42 theo tỷ lệ xấp xỉ 70/15/15. Ảnh Kaggle trùng pixel với SHWD bị loại để tránh rò rỉ giữa split. Box Kaggle khoanh helmet/head, còn box SHWD giữ hình học gốc; hai nguồn không đảm bảo cùng quy ước khung bao.

Một vật thể mũ xuất hiện trong ảnh chưa đủ để kết luận một người cụ thể đang đội mũ đúng cách. Phạm vi cơ sở chỉ hỗ trợ nhận biết theo nhãn vùng đầu/mũ của dataset; cần nêu giới hạn khi demo trên cảnh thực tế.

### 4.3. Ngoài phạm vi bắt buộc

- Nhận diện danh tính công nhân.
- Kiểm tra quai mũ hoặc tiêu chuẩn chất lượng của mũ.
- Phát hiện toàn thân và ghép mũ với từng người.
- Theo dõi ID hoặc đếm số người vi phạm duy nhất theo thời gian.
- Phát hiện áo phản quang, găng tay và thiết bị bảo hộ khác.
- Hệ thống nhiều camera, ứng dụng di động hoặc cảnh báo từ xa.
- Cải tiến kiến trúc mạng, huấn luyện từ đầu hoặc triển khai YOLOv9.

Tên nhóm bài toán trong đề có nội dung theo dõi đối tượng, nhưng yêu cầu cụ thể không bắt buộc tracking. YOLOv9 nằm trong phần tham khảo, không phải mô hình bắt buộc triển khai.

### 4.4. Phần mở rộng khi còn thời gian

Chọn tối đa một hướng ưu tiên sau khi hoàn thành phần bắt buộc:

- Tăng kích thước đầu vào để khảo sát lỗi vật thể nhỏ.
- Điều chỉnh augmentation ánh sáng dựa trên lỗi quan sát được.
- Lặp lại huấn luyện với nhiều seed để đánh giá độ ổn định.
- Làm giao diện chọn video và mô hình.
- Bổ sung tracking nếu có đủ thời gian kiểm tra chất lượng ghép ID.

## 5. Dữ liệu và tiền xử lý

### 5.1. Lựa chọn dataset

Đề xuất dùng Safety Helmet Wearing Dataset (SHWD) làm bộ dữ liệu chính. Theo repository gốc, bộ dữ liệu có 7.581 ảnh, 9.044 đối tượng đội mũ và 111.514 đối tượng đầu không đội mũ, với nhãn theo định dạng Pascal VOC. Đây là số liệu do nguồn công bố; nhóm cần đếm lại bản tải thực tế [1].

Chênh lệch số đối tượng giữa hai lớp là lý do phải thống kê phân bố và báo cáo chỉ số riêng từng lớp. Không chỉ dựa vào một chỉ số tổng hợp để kết luận chất lượng.

Hard Hat Workers trên Kaggle/Roboflow là phương án thay thế nếu SHWD không tải được hoặc không phù hợp. Nếu chọn bộ thay thế, phải ghi URL chính xác, phiên bản, định nghĩa nhãn và điều kiện sử dụng. Không ghép hai bộ ngay từ đầu khi chưa kiểm tra ảnh trùng và khác biệt quy tắc gán nhãn.

### 5.2. Quy trình chuẩn bị

| Bước | Công việc | Đầu ra |
|---|---|---|
| 1 | Ghi nguồn tải, phiên bản và điều kiện sử dụng | Hồ sơ nguồn dữ liệu |
| 2 | Đếm ảnh, nhãn, số đối tượng và kích thước box theo lớp | Bảng thống kê ban đầu |
| 3 | Kiểm tra ảnh hỏng, thiếu nhãn, nhãn sai, box ngoài ảnh | Danh sách lỗi và cách xử lý |
| 4 | Tìm ảnh trùng/gần trùng và xác định nhóm cùng cảnh nếu có | Danh sách nhóm ảnh |
| 5 | Chuyển Pascal VOC sang YOLO | Nhãn chuẩn hóa và bảng ánh xạ class ID |
| 6 | Vẽ nhãn lên ảnh, kiểm tra các trường hợp biên và mẫu ngẫu nhiên | Ảnh kiểm tra trực quan |
| 7 | Chia train/validation/test và lưu danh sách cố định | Các split tái lập được |

Định dạng nhãn YOLO dự kiến: `class_id x_center y_center width height`, với tọa độ chuẩn hóa theo kích thước ảnh. Cần kiểm tra tọa độ, class ID và tương ứng ảnh–nhãn sau chuyển đổi.

### 5.3. Chia tập và phòng tránh rò rỉ dữ liệu

- Đề xuất tỷ lệ 70% train, 15% validation, 15% test nếu chưa có cách chia chuẩn cần giữ nguyên.
- Nếu dùng split chính thức của bộ dữ liệu, mô tả và giữ nhất quán thay vì tự chia lại thiếu lý do.
- Chia theo video, chuỗi ảnh hoặc bối cảnh khi xác định được; tránh ảnh gần giống nằm ở các tập khác nhau.
- Kiểm tra phân bố lớp và các điều kiện khó sau khi chia.
- Chỉ augmentation trên train; không đưa biến thể của cùng ảnh sang validation/test.
- Dùng validation để chọn tham số và checkpoint.
- Giữ test độc lập, không điều chỉnh mô hình dựa trên kết quả test.
- Giữ dữ liệu gốc và ghi lại các thay đổi để có thể tái lập.

### 5.4. Video demo ngoài dữ liệu huấn luyện

Nếu có thể, chuẩn bị 2–3 video khác nguồn với dữ liệu huấn luyện, gồm cảnh dễ và cảnh khó. Ghi nguồn video, độ phân giải và đặc điểm bối cảnh.

Video chưa gán nhãn chỉ dùng minh họa định tính. Muốn báo cáo mAP hoặc Recall trên video phải gán nhãn các khung hình được đánh giá và mô tả cách lấy mẫu.

## 6. Công cụ và môi trường

| Thành phần | Phương án đề xuất |
|---|---|
| Ngôn ngữ | Python |
| Huấn luyện và suy luận | Ultralytics YOLOv8 trên PyTorch |
| Xử lý ảnh và video | OpenCV |
| Thống kê và biểu đồ | pandas, Matplotlib hoặc công cụ tương đương |
| Máy huấn luyện | GPU NVIDIA cục bộ hoặc môi trường GPU được nhóm tiếp cận |
| Quản lý phiên bản | Git |
| Ghi cấu hình | YAML, file phiên bản thư viện và log |

Chốt phiên bản thư viện sau khi chạy thử thành công. Ghi hệ điều hành, GPU, VRAM, CUDA, PyTorch và Ultralytics trong hồ sơ thí nghiệm. Chọn batch size và ước lượng tổng thời gian dựa trên chạy thử trên phần cứng thực tế, không dựa vào tốc độ công bố trên máy khác.

## 7. Thiết kế huấn luyện và thí nghiệm

### 7.1. Cấu hình khởi đầu

Ba trọng số khởi tạo là `yolov8n.pt`, `yolov8s.pt` và `yolov8m.pt`; đây là các phiên bản detection chính thức của YOLOv8 [2].

| Tham số | Đề xuất ban đầu |
|---|---|
| Kích thước đầu vào | 640 |
| Ngân sách epoch | Tối đa 100 epoch cho mỗi mô hình |
| Batch size | Thử trên YOLOv8m rồi chọn mức phù hợp GPU |
| Khởi tạo | Trọng số pretrained tương ứng |
| Optimizer, learning rate | Chọn và ghi rõ trước đợt so sánh chính |
| Augmentation | Cùng cấu hình cho ba mô hình trong so sánh chính |
| Seed | Cố định và ghi lại |
| Checkpoint dùng đánh giá | Chọn theo quy tắc dựa trên validation |

Đây là cấu hình khởi đầu, chưa phải cấu hình tối ưu. Chạy thử 3–5 epoch trên một phần train để kiểm tra pipeline, nhãn, bộ nhớ và thời gian trước khi huấn luyện đầy đủ.

### 7.2. Ma trận thí nghiệm

| Mã | Nội dung | Mức độ | Mục đích |
|---|---|---|---|
| E0 | Chạy thử YOLOv8n trên một phần train | Bắt buộc | Kiểm tra pipeline; không dùng làm kết quả cuối |
| E1 | Fine-tune YOLOv8n | Bắt buộc | Mốc so sánh mô hình nhỏ |
| E2 | Fine-tune YOLOv8s | Bắt buộc | So sánh độ chính xác và tốc độ |
| E3 | Fine-tune YOLOv8m | Bắt buộc | So sánh mô hình lớn hơn |
| E4 | Thay đổi một yếu tố dựa trên lỗi validation | Mở rộng | Kiểm tra một hướng cải thiện |
| E5 | Lặp lại với seed khác | Nếu đủ tài nguyên | Khảo sát độ ổn định |

### 7.3. Nguyên tắc so sánh

- E1–E3 dùng cùng dữ liệu, input size, quy tắc augmentation, ngân sách epoch và cách chọn checkpoint.
- Nếu dùng early stopping, áp dụng cùng quy tắc và báo cáo số epoch thực tế.
- Cố gắng dùng cùng batch size; nếu phải thay đổi vì VRAM, ghi rõ và phân tích ảnh hưởng đến tính công bằng.
- Không dùng lựa chọn tự động làm thay đổi optimizer giữa các mô hình mà không ghi nhận.
- Lưu cấu hình thực tế, checkpoint, log và thời gian của từng lần chạy.
- E4 chỉ thay đổi một yếu tố để có thể giải thích kết quả.
- Chốt lựa chọn trên validation trước khi đánh giá test cuối cùng.

Nếu chỉ chạy một seed, phải nêu rõ giới hạn về độ ổn định; không kết luận một chênh lệch nhỏ là có ý nghĩa thống kê.

## 8. Kế hoạch đánh giá

### 8.1. Chỉ số bắt buộc

| Chỉ số | Vai trò |
|---|---|
| Precision | Mức độ chính xác của các phát hiện được đưa ra |
| Recall | Khả năng tìm đủ các đối tượng cần phát hiện |
| mAP@0.5 | Đánh giá detection tại ngưỡng IoU 0,5 |
| mAP@0.5:0.95 | Đánh giá trung bình qua nhiều ngưỡng IoU từ 0,5 đến 0,95 |
| FPS | Tốc độ xử lý trong điều kiện đo được mô tả rõ |

Báo cáo kết quả tổng thể và riêng từng lớp. Precision/Recall phải đi kèm quy tắc tính và ngưỡng liên quan. Chọn ngưỡng confidence cho demo trên validation rồi cố định; không dùng ngưỡng demo cao để lọc mất dự đoán trước khi tính mAP theo quy trình chuẩn.

### 8.2. Bảng kết quả cần điền sau thực nghiệm

| Mô hình | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS mô hình | FPS demo |
|---|---|---|---|---|---|---|
| YOLOv8n | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| YOLOv8s | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| YOLOv8m | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Đính kèm confusion matrix, đường Precision–Recall, đồ thị loss/metric theo epoch và biểu đồ độ chính xác–tốc độ. Bổ sung dung lượng trọng số, thời gian huấn luyện, VRAM nếu đo được.

### 8.3. Quy trình đo tốc độ

1. Dùng cùng máy, thiết bị tính toán, input size, precision và backend.
2. Dùng batch size 1 cho kịch bản video.
3. Warm-up trước khi đo và đồng bộ GPU tại ranh giới đo khi cần.
4. Đo nhiều lần trên cùng đầu vào, ghi số khung hình và báo cáo giá trị trung bình cùng mức biến động nếu có.
5. Phân biệt FPS suy luận mô hình với FPS toàn pipeline.
6. Ghi rõ FPS toàn pipeline có bao gồm đọc video, tiền xử lý, hậu xử lý, vẽ box và ghi tệp hay không.

FPS được tính bằng số khung hình xử lý chia cho thời gian đo tương ứng. Không dùng FPS của video đầu vào hoặc FPS công bố trên phần cứng khác thay cho tốc độ xử lý thực tế.

## 9. Phân tích lỗi

| Điều kiện | Cách xác định | Nội dung phân tích |
|---|---|---|
| Che khuất | Gắn thẻ theo mức độ vùng đầu/mũ bị che, với quy tắc ghi rõ | Bỏ sót, nhầm lớp, định vị box |
| Vật thể nhỏ | Phân nhóm theo kích thước box với ngưỡng và hệ tọa độ cố định | Hiệu quả phát hiện đối tượng ở xa |
| Thiếu sáng | Gắn thẻ theo tiêu chí ánh sáng nhất quán | Bỏ sót hoặc nhận nhầm nền/vật thể |

Đề xuất khảo sát khoảng 50–100 ảnh mỗi nhóm nếu dữ liệu cho phép. Một ảnh có thể thuộc nhiều nhóm; cần công bố sự chồng lặp và số đối tượng thực tế. Chốt quy tắc phân nhóm trước khi xem kết quả test để tránh chỉ chọn ví dụ có lợi cho mô hình.

Mỗi nhóm phân tích cần có:

- Số ảnh, số đối tượng và phân bố lớp.
- Ví dụ phát hiện đúng, bỏ sót, nhầm lớp và phát hiện nhầm.
- Chỉ số phù hợp khi có đủ nhãn; ưu tiên xem Recall của lớp không đội mũ.
- Giải thích nguyên nhân có thể và hướng cải thiện.
- Giới hạn của kết luận nếu mẫu ít hoặc việc gắn thẻ mang tính chủ quan.

Khảo sát validation dùng để đề xuất cải thiện. Khảo sát test dùng để báo cáo kết quả cuối; không tiếp tục chỉnh mô hình dựa trên lỗi test trong cùng đợt đánh giá.

## 10. Demo video

### 10.1. Chức năng tối thiểu

1. Nhận đường dẫn trọng số và video đầu vào.
2. Cho phép đặt ngưỡng confidence đã lựa chọn.
3. Dự đoán trên từng khung hình.
4. Hiển thị bounding box, nhãn và confidence bằng OpenCV.
5. Hiển thị FPS với cách tính được mô tả.
6. Lưu video có kết quả vào thư mục đầu ra.

Có thể hiển thị số vùng đầu không đội mũ trong khung hình hiện tại. Khi chưa có tracking, không gọi đó là tổng số người vi phạm duy nhất trong video.

### 10.2. Tiêu chí hoàn thành

- Người khác có thể chạy demo theo README.
- Tải được mô hình và đọc được video.
- Xử lý hết video và xuất tệp kết quả mở được.
- Kết quả có cả trường hợp dễ và khó để trình bày giới hạn.
- Có video kết quả dự phòng khi máy thuyết trình không có GPU.

Không bắt buộc làm giao diện web. Chỉ thêm giao diện sau khi pipeline và kết quả đã hoàn chỉnh.

## 11. Tiến độ đề xuất

| Tuần | Công việc | Đầu ra | Điều kiện chuyển bước |
|---|---|---|---|
| 1 | Đọc đề, đọc tài liệu/paper, tải và khảo sát dataset | Phạm vi, định nghĩa nhãn, hồ sơ nguồn dữ liệu | Có dữ liệu và chốt được bài toán |
| 2 | Làm sạch, chuyển nhãn, chia tập, cài môi trường, chạy E0 | Pipeline dữ liệu và cấu hình chạy thử | Nhãn kiểm tra đúng, train/predict chạy được |
| 3 | Chạy E1–E3, lưu log và checkpoint | Ba mô hình và kết quả validation | Hoàn thành so sánh cơ sở |
| 4 | Đo tốc độ, phân tích validation, chạy E4 nếu đủ thời gian, chốt cấu hình và đánh giá test | Bảng kết quả cuối, lựa chọn mô hình demo | Test độc lập và số liệu đầy đủ |
| 5 | Hoàn thiện demo, phân tích lỗi test, viết báo cáo | Video và bản báo cáo đầy đủ đầu tiên | Demo chạy được, các nhận xét có minh chứng |
| 6 | Kiểm tra tái lập, rà số liệu, làm slide, tập bảo vệ | Bộ nộp hoàn chỉnh | Chạy lại được theo README |

Nếu thời gian ngắn, giảm phần mở rộng, số seed và công sức làm giao diện trước. Giữ đủ ba mô hình, năm chỉ số, ba nhóm lỗi và demo video.

## 12. Phân công theo mảng

Số thành viên chưa được xác nhận; phân công dưới đây theo mảng trách nhiệm để dễ gộp hoặc chia.

| Mảng | Trách nhiệm | Sản phẩm |
|---|---|---|
| Dữ liệu | Nguồn tải, kiểm tra, chuyển nhãn, chia tập, thống kê | Dataset đã chuẩn hóa và hồ sơ xử lý |
| Mô hình | Môi trường, cấu hình, huấn luyện, checkpoint, log | Ba mô hình và hồ sơ thí nghiệm |
| Đánh giá | Chỉ số, đo FPS, biểu đồ, phân tích lỗi | Bảng kết quả và phần thực nghiệm |
| Demo và tích hợp | Xử lý video, README, đóng gói, tổng hợp tài liệu | Demo và bộ nộp |

Nhóm ít người có thể gộp dữ liệu–đánh giá và mô hình–demo. Mỗi thành viên viết phần báo cáo tương ứng công việc; cả nhóm kiểm tra kết quả và kết luận.

## 13. Cấu trúc thư mục dự kiến

Cấu trúc dưới đây là cấu trúc khởi điểm đề xuất; triển khai thực tế đã bổ sung `scripts/`, `runs/`, `results/` và `weights/`. Trạng thái và cách chạy được cập nhật trong `README.md`.

```text
Noron/
├── KE_HOACH_DU_AN_HELMET_DETECTION.md
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── demo_video.py
├── weights/
├── results/
│   ├── metrics/
│   ├── figures/
│   └── error_analysis/
├── demos/
│   ├── input/
│   └── output/
└── reports/
```

Không đưa dữ liệu lớn, video và checkpoint vào Git thông thường; dùng nơi lưu trữ phù hợp và ghi liên kết tải/hướng dẫn trong README. Giữ cấu hình, danh sách split và số liệu tổng hợp trong quản lý phiên bản.

## 14. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Link dataset không tải được | Kiểm tra sớm; chuyển sang bộ thay thế sau khi xác nhận nhãn và nguồn |
| Nhãn sai hoặc không đúng định nghĩa bài toán | Kiểm tra ảnh trực quan, ghi quy tắc ánh xạ và loại bỏ/sửa lỗi có nhật ký |
| Mất cân bằng lớp | Báo cáo theo lớp; chỉ thử xử lý cân bằng trên train khi validation cho thấy cần thiết |
| Rò rỉ ảnh gần trùng | Kiểm tra trước khi chia, giữ cùng chuỗi/cảnh trong một tập khi có thể |
| GPU thiếu bộ nhớ | Chọn batch sau chạy thử YOLOv8m, ghi mọi thay đổi cấu hình |
| Thời gian hoặc tài nguyên huấn luyện hạn chế | Lưu checkpoint, bỏ phần mở rộng trước và giữ thí nghiệm bắt buộc |
| FPS thấp | Đo rõ nút thắt và chọn mô hình phù hợp từ kết quả thực tế |
| Demo kém trên video mới | Trình bày giới hạn tổng quát hóa, giữ cả ví dụ thành công và thất bại |
| Số liệu báo cáo không khớp | Dùng bảng kết quả gốc làm nguồn chung cho báo cáo và slide |

## 15. Dàn ý báo cáo cuối kỳ

1. **Giới thiệu:** Bối cảnh, mục tiêu, câu hỏi nghiên cứu và phạm vi.
2. **Cơ sở lý thuyết:** Object detection, YOLOv8, fine-tuning và chỉ số đánh giá.
3. **Công trình liên quan:** Các hướng phát hiện mũ bảo hộ và paper được đề bài giới thiệu.
4. **Dữ liệu:** Nguồn, nhãn, phân bố, tiền xử lý và cách chia tập.
5. **Phương pháp:** Pipeline, mô hình, cấu hình và thiết kế thí nghiệm.
6. **Kết quả:** Bảng n/s/m, kết quả theo lớp, biểu đồ và đo tốc độ.
7. **Phân tích lỗi:** Che khuất, vật thể nhỏ, thiếu sáng và giới hạn.
8. **Demo:** Cách vận hành, ví dụ đầu ra và hiệu năng đo được.
9. **Kết luận:** Trả lời câu hỏi nghiên cứu, mô hình lựa chọn và hướng phát triển.
10. **Tài liệu tham khảo và phụ lục:** Nguồn, cấu hình, lệnh chạy và phân công.

Paper “Safety Helmet Detection Based on Improved YOLOv8” trong đề dùng để tham khảo phương pháp và cách viết báo cáo. Đề không bắt buộc tái hiện toàn bộ cải tiến của paper. Khi sử dụng nội dung từ paper, cần đọc bản đầy đủ và trích dẫn phần thực sự đã tham khảo.

## 16. Danh sách kiểm tra trước khi nộp

- [ ] Phạm vi và định nghĩa nhãn được mô tả rõ.
- [ ] Có nguồn dữ liệu, thống kê, nhật ký xử lý và danh sách split.
- [ ] Đã kiểm tra rò rỉ dữ liệu và ảnh trùng/gần trùng.
- [ ] Có mã nguồn, cấu hình và phiên bản môi trường.
- [ ] Có checkpoint, log và kết quả của YOLOv8n/s/m.
- [ ] Có đủ mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS.
- [ ] Có đánh giá test độc lập và kết quả theo lớp.
- [ ] Có quy trình đo FPS và cấu hình phần cứng.
- [ ] Có phân tích che khuất, vật thể nhỏ, thiếu sáng.
- [ ] Có ảnh minh họa đúng/sai và giải thích có căn cứ.
- [ ] Demo xử lý hết video, lưu được đầu ra và có bản dự phòng.
- [ ] README hướng dẫn cài đặt, tải dữ liệu/trọng số và chạy lại.
- [ ] Báo cáo, slide và số liệu gốc thống nhất.
- [ ] Không trình bày số liệu dự kiến như kết quả đã đo.

## 17. Tài liệu tham khảo và nguồn triển khai

### 17.1. Nguồn đã dùng để lập kế hoạch

- **Tài liệu giao đề:** `Helmet_Detection_with_YOLOv8_nguyen_van_tu_sheet.docx`, đã đọc toàn bộ nội dung.
- **[1] SHWD:** [Repository Safety Helmet Wearing Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), dùng để xác nhận mô tả, quy mô công bố và định dạng dữ liệu.
- **[2] Ultralytics:** [Tài liệu YOLOv8](https://docs.ultralytics.com/models/yolov8/), dùng để xác nhận các phiên bản detection và khả năng huấn luyện/đánh giá/suy luận.
- **[3] Kaggle Hard Hat Detection:** [Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection), 5.000 ảnh, nhãn Pascal VOC, metadata Kaggle ghi CC0 Public Domain; dữ liệu ghi nhận tác giả Larxel/MakeML.

### 17.2. Nguồn đề bài giới thiệu để đọc khi triển khai

Các liên kết sau được giữ từ tài liệu giao đề; cần đọc, kiểm tra nội dung và khả năng sử dụng trước khi áp dụng. Không coi mọi kết quả hoặc cách làm trong các nguồn là đã được kiểm chứng trong dự án.

- [Tài liệu Ultralytics](https://docs.ultralytics.com/).
- [Mã nguồn Ultralytics](https://github.com/ultralytics/ultralytics).
- [Paper Safety Helmet Detection Based on Improved YOLOv8](https://www.researchgate.net/publication/378347736_Safety_Helmet_Detection_Based_on_Improved_YOLOv8). Đây là paper, không phải mã nguồn.
- [Repository helmet-detection-yolov8](https://github.com/meryemsakin/helmet-detection-yolov8).
- [Notebook YOLOv8 Safety Helmet Detection](https://www.kaggle.com/code/plasticglass/yolov8-safety-helmet-detection).
- [Paper YOLOv9](https://arxiv.org/abs/2402.13616).
- [Mã nguồn YOLOv9](https://github.com/WongKinYiu/yolov9).

## 18. Các thông tin cần chốt khi bắt đầu

- Hạn nộp và lịch bảo vệ thực tế.
- Số thành viên và người phụ trách từng mảng.
- GPU có thể sử dụng, VRAM và thời lượng truy cập.
- Dataset tải được và phiên bản sẽ dùng.
- Quy định riêng của giảng viên về báo cáo, slide và hình thức nộp.

Các thông tin này dùng để điều chỉnh lịch và nguồn lực; không làm thay đổi các yêu cầu cốt lõi: so sánh n/s/m, đánh giá đủ chỉ số, phân tích lỗi và demo video.

## 19. Tiến độ triển khai thực tế

**Cập nhật ngày 03/11/2026:** Đã tải và chuẩn hóa SHWD với bộ Kaggle Hard Hat Detection. SHWD giữ split chính thức sau khi loại ảnh chứa ba box `dog`; Kaggle có 5.000 ảnh Pascal VOC, metadata Kaggle ghi CC0. Tập hợp có 12.580 ảnh: 8.955 train, 1.356 validation, 2.269 test. Kaggle được chia theo presence lớp với seed 42; không phát hiện ảnh trùng pixel giữa hai nguồn. Sau ánh xạ có 27.997 box đội mũ và 117.299 box không đội mũ. Kiểm tra nhãn mẫu: `results/data_audit/combined_val_samples.jpg`.

Huấn luyện YOLOv8n/s/m hoàn tất với cùng cấu hình: tối đa 10 epoch, batch 1, ảnh 512 px, `workers=0`, `mosaic=0`, patience 7, seed 42, GPU NVIDIA RTX 3050 Laptop 4 GB. Checkpoint được chọn theo mAP50–95 trên validation; test độc lập chỉ chạy sau khi chốt checkpoint.

| Mô hình | Best epoch | Validation mAP50–95 | Test Precision | Test Recall | Test mAP50 | Test mAP50–95 | ms/ảnh |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 8 | 0,3829 | 0,8467 | 0,5760 | 0,6979 | 0,3950 | 14,23 |
| YOLOv8s | 10 | 0,4086 | 0,8726 | 0,5921 | 0,7252 | 0,4250 | 14,12 |
| YOLOv8m | 8 | **0,4389** | 0,8691 | **0,6772** | **0,7779** | **0,4525** | 18,86 |

Mô hình demo được chọn là YOLOv8m theo validation; tệp `weights/helmet_best.pt` trỏ tới best checkpoint. Metric theo lớp, JSON/CSV gốc và cấu hình đánh giá được lưu trong `results/_test_*/`; xem bảng đầy đủ trong [BAO_CAO_KET_QUA.md](BAO_CAO_KET_QUA.md).

Phân tích lỗi test của YOLOv8m tại confidence 0,25 và IoU matching 0,5: recall của `wearing_helmet` là 0,680, còn `not_wearing_helmet` là 0,416. Với box nhỏ hơn 1% diện tích ảnh, recall lần lượt là 0,549 và 0,385; recall tăng rõ ở box vừa/lớn. Các lỗi nổi bật gồm bỏ sót đầu rất nhỏ trong cảnh đông người. Dataset không có nhãn che khuất riêng nên phần che khuất chỉ được xem định tính qua ảnh lỗi; bảng phân tầng và montage nằm tại `results/error_analysis/helmet_best_final_chunked/`.

Đã kiểm thử app Streamlit trực tiếp trên browser tại `http://127.0.0.1:8502`: ảnh validation mẫu cho 10 phát hiện, thời gian suy luận 20,4 ms; clip MP4 H.264 gồm 20 khung hình validation được xử lý thành công, đầu ra phát được ở 640×480, có nút tải xuống và đạt 21,65 FPS toàn pipeline. Clip là fixture để kiểm tra luồng upload/inference/encode; nó không đại diện video tự nhiên hoặc theo dõi đối tượng. Lệnh chạy app và kết quả chi tiết ở `README.md` và [báo cáo kết quả](BAO_CAO_KET_QUA.md).
