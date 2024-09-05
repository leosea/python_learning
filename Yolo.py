from ultralytics import YOLO
import cv2

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano模型

# 读取图像
img = cv2.imread('groupphoto.jpg')

# 进行目标检测
results = model(img)

# 在图像上绘制检测结果
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{model.names[cls]} {conf:.2f}', (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示结果
cv2.imshow('YOLO Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
