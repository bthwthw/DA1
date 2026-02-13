# modules/detector.py
from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.onnx', conf_threshold=0.5):
        print(f"Loading model from: {model_path}...")
        self.model = YOLO(model_path, task='detect') # Load ONNX model
        self.conf = conf_threshold

    def detect(self, frame):
        """
        Nhận diện vật thể trong khung hình.
        Trả về list các kết quả.
        """
        # Stream=True giúp chạy nhanh hơn, không bị lag bộ nhớ
        results = self.model(frame, stream=True, conf=self.conf, verbose=False)
        return results