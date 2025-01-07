import matplotlib.pyplot as plt
from ultralytics import YOLO

if __name__ == '__main__':
    # Tải mô hình YOLO
    model = YOLO("D://Project//Python//TGM//Bienso_v2//training_v1//runs//detect//train6//weights//best.pt")

    # Kiểm tra mô hình với bộ dữ liệu xác thực (validation)
    results = model.val(data="D://Project//Python//TGM//Bienso_v2//training_v1//data.yaml")  # Thay đường dẫn đến tệp YAML của bạn

    # In kết quả
    print(results)
