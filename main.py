# main.py
import cv2
import math
from modules.detector import ObjectDetector
from modules.estimator import DistanceEstimator

# --- CẤU HÌNH (CONFIG) ---
MODEL_PATH = 'yolov8n.onnx'
REAL_WIDTH_PERSON = 40.0  
FOCAL_LENGTH = 543.75  

class VisionSystem:
    def __init__(self):
        # Khởi tạo các module con
        self.detector = ObjectDetector(model_path=MODEL_PATH)
        self.estimator = DistanceEstimator(FOCAL_LENGTH, REAL_WIDTH_PERSON)
        
        # Mở Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

    def run(self):
        print("System started. Press 'q' to exit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # 1. Gọi module AI để nhận diện
            results = self.detector.detect(frame)

            # 2. Xử lý kết quả
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Lấy class ID (0 là Person)
                    cls_id = int(box.cls[0])
                    
                    if cls_id == 0: # Chỉ xử lý NGƯỜI
                        # Lấy tọa độ bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Tính chiều rộng pixel (w_pixel)
                        w_pixel = x2 - x1
                        
                        # 3. Gọi module Toán để tính khoảng cách
                        distance = self.estimator.estimate(w_pixel)

                        # 4. Logic cảnh báo (STOP/GO)
                        color = (0, 255, 0) # Xanh lá (An toàn)
                        status = "GO"
                        
                        if distance < 1.5:
                            color = (0, 0, 255) # Đỏ (Nguy hiểm)
                            status = "STOP"

                        # 5. Vẽ lên màn hình
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"Person: {distance:.2f}m | {status}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Hiển thị
            cv2.imshow('Robot Vision System (ONNX)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VisionSystem()
    app.run()