from ultralytics import YOLO

# Tải mô hình YOLO
model = YOLO("D://Project//Python//TGM//Bienso_v2//training_v1//runs//detect//train7//weights//best.pt")

# Dự đoán ảnh
results = model("C://image//bienso//image//0005.jpg")

# Hiển thị kết quả
results[0].show()  # Truy cập phần tử đầu tiên trong danh sách và gọi phương thức show
