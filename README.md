# Đường An — Diabetes research app

Ứng dụng mới thay hoàn toàn app tĩnh cũ, sử dụng trực tiếp model trong `diabetes_model_for_app.zip` (ANN tháng 11/2026), không dùng lại trọng số hoặc bộ tiền xử lý của app cũ.

## Chạy bằng Docker

```sh
docker build -t duong-an .
docker run --rm -p 8000:8000 duong-an
```

Mở http://localhost:8000. Lần dự đoán đầu tiên tải model vào RAM, có thể chậm hơn các lần sau.

## Chạy bằng Python 3.11

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Backend Keras dùng Torch, với Keras/Torch/scikit-learn khớp phiên bản huấn luyện. Docker dùng Torch CPU. Cấu hình Vercel tĩnh cũ đã được gỡ vì không chạy được backend Python. `.openai/hosting.json` là định danh lịch sử, không phải cấu hình triển khai backend mới.

Trên máy phát triển Windows hiện tại, kiểm tra cú pháp Python/JavaScript đã qua nhưng kiểm thử suy luận bị chặn bởi lỗi khởi tạo DLL của Torch (`WinError 1114`). Docker chưa được kiểm thử do máy chưa cài Docker. Không coi đây là bản đã xác minh suy luận đầu cuối; nên chạy bộ kiểm thử bên dưới trên môi trường triển khai.

## API và kiểm tra

- `GET /api/health`: tình trạng tiến trình, không xác nhận model đã được tải.
- `POST /api/predict`: đúng 8 trường trong `model/features.json`, trả score, threshold, predicted_label và model_version.
- `GET /api/docs`: tài liệu API tương tác.
- `pip install httpx` rồi `python -m unittest discover -s tests -v`: kiểm tra hợp đồng API, dữ liệu sai và suy luận model thật.

Tiền xử lý chạy bằng chính `preprocessor.joblib` đi kèm, giữ nguyên thứ tự đặc trưng và ngưỡng `0.8370159268379211`. Model và bộ tiền xử lý được giữ nguyên từ ZIP. Không có cơ sở dữ liệu hoặc lưu hồ sơ. Dữ liệu được truyền đến backend; khi triển khai công khai cần HTTPS.

Điểm model không phải xác suất bệnh cá nhân đã hiệu chuẩn. Đây là phân loại nghiên cứu, không thay thế chẩn đoán. Các chỉ số trên giao diện trích từ metadata do gói model cung cấp, chưa được tái đánh giá độc lập.
