from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="D://Project//Python//TGM//Bienso_v2//training_v1//data.yaml",  # Đường dẫn đến tệp YAML
        epochs=200,         # Số lượng epoch
        imgsz=640,         # Kích thước ảnh
        batch=16,          # Kích thước batch
        device=0           # Sử dụng GPU (device=0) hoặc CPU (device='cpu')
    )

if __name__ == "__main__":
    main()
