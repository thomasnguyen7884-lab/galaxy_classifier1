
# Galaxy Morphology Mini-Project (Student Pack)

## Cách chạy (local)
1) Cài Python 3.9+ và các gói:
   pip install tensorflow==2.16.1 numpy pillow matplotlib

2) Dự đoán 1 ảnh:
   python predict.py --model galaxy_classifier.keras --classes class_names.npy --image <duong_dan_anh.jpg>

## Ghi chú
- Nếu mô hình bạn lưu là classifier trên embedding, script đã tự nhúng MobileNetV2 để trích đặc trưng trước khi dự đoán.
- Nếu mô hình là end-to-end (nhận ảnh vào trực tiếp), script cũng tự nhận biết.
