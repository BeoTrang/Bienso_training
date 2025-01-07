from ultralytics import YOLO
import cv2

# Tải mô hình YOLO
model = YOLO("D://Project//Python//TGM//Bienso_v2//training_v1//runs//detect//train5//weights//best.pt")

input_video_path = 'C://image//bienso//video//0004.mp4'  # Đường dẫn video của bạn
output_video_path = 'C://image//bienso//video//output47.mp4'

# Mở video đầu vào
cap = cv2.VideoCapture(input_video_path)

# Lấy thông tin cơ bản của video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video đầu ra

# Tạo đối tượng ghi video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Kiểm tra xem video có mở được không
if not cap.isOpened():
    print("Lỗi: Không thể mở video:", input_video_path)
else:
    print("Đang xử lý video...")

# Lặp qua từng frame của video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Kết thúc video

    # Chuyển đổi từ BGR sang RGB vì YOLOv5 yêu cầu RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chạy mô hình để nhận diện đối tượng
    results = model(rgb_frame)

    # Kiểm tra nếu có đối tượng nào được phát hiện
    if results:
        # Dùng phương thức `plot()` để vẽ bounding box lên frame
        annotated_frame = results[0].plot()  # `plot()` trả về ảnh đã vẽ bounding box
    else:
        # Nếu không có đối tượng nào, sử dụng frame gốc
        annotated_frame = frame

    # Ghi frame vào video đầu ra
    out.write(annotated_frame)

    # Hiển thị frame (nếu cần)
    # cv2.imshow('Video', annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát sớm
    #     break

# Đóng các luồng video
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video đã được xử lý và lưu tại:", output_video_path)
