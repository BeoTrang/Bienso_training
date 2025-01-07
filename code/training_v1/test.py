from ultralytics import YOLO

# Tải mô hình YOLO
model = YOLO("/training_v1/runs/detect/train5/weights/best.pt")

# Dự đoán ảnh
results = model("C://image//bienso//image//0001.jpg")

# Hiển thị kết quả
results[0].show()  # Truy cập phần tử đầu tiên trong danh sách và gọi phương thức show
