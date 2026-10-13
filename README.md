## Cách chạy (local)
1) Cài Python 3.9+ và các gói:
   pip install tensorflow==2.16.1 numpy pillow matplotlib

2) Dự đoán 1 ảnh:
   python predict.py --model galaxy_classifier.keras --classes class_names.npy --image <anh.jpg>

## Ghi chú
- Nếu mô hình bạn lưu là classifier trên embedding, script đã tự nhúng MobileNetV2 để trích đặc trưng trước khi dự đoán.
- Nếu mô hình là end-to-end (nhận ảnh vào trực tiếp), script cũng tự nhận biết.

## How to run (locally)
1) Install Python 3.9+ and the required packages:
   pip install tensorflow==2.16.1 numpy pillow matplotlib
2) Predict a single image:
   python predict.py --model galaxy_classifier.keras --classes class_names.npy --image <image.jpg>

## Notes
- If your saved model is a classifier built on top of embeddings, the script automatically embeds MobileNetV2 to extract features before making a prediction.
- If the model is end-to-end (accepts raw images directly), the script will also detect this automatically.
