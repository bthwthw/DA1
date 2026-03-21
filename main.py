# main.py
import cv2
import time
from modules.detector import ObjectDetector
from modules.estimator import DistanceEstimator

MODEL_PATH = 'yolov8n_openvino_model/'
FOCAL_LENGTH = 543.75  
CAMERA_HEIGHT = 1.5  
HORIZON_Y = 240 

class VisionSystem:
    def __init__(self):
        self.detector = ObjectDetector(model_path=MODEL_PATH)
        self.estimator = DistanceEstimator(FOCAL_LENGTH, CAMERA_HEIGHT, HORIZON_Y)
        
        self.cap = cv2.VideoCapture(0)
        
        self.history = {} # {id_vật_thể: khoảng_cách_mét_ở_frame_trước}
        self.fps_assumed = 30.0 # Giả định camera chạy 30 FPS để tính Delta T

    def run(self):
        print("System started. Press 'q' to exit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # goi yolo 
            results = self.detector.track_objects(frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    
                    if cls_id == 0: # class 0 là person
                        if box.id is None:
                            continue
                            
                        obj_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        u, v_bottom = self.detector.get_bottom_center(box)
                        
                        # khoảng cách hiện tại (Z)
                        current_distance = self.estimator.estimate(v_bottom)
                        if current_distance < 0: continue

                        # TÍNH TOÁN TTC VÀ VẬN TỐC
                        ttc = float('inf') # Khởi tạo TTC vô cực (an toàn)
                        status = "GO"
                        color = (0, 255, 0)
                        
                        if obj_id in self.history:
                            prev_distance = self.history[obj_id]
                            
                            # vận tốc tương đối (m/s)
                            v_rel = (prev_distance - current_distance) * self.fps_assumed
                            
                            # Chỉ tính TTC nếu vật thể đang tiến lại gần (v_rel > 0)
                            if v_rel > 0:
                                ttc = current_distance / v_rel
                                
                                # nguy hiểm - cảnh báo 
                                if ttc <= 4.0 or current_distance < 1.5:
                                    color = (0, 0, 255) 
                                    status = f"STOP! TTC: {ttc:.1f}s"
                                else:
                                    color = (0, 255, 255)
                                    status = f"WARN: {ttc:.1f}s"

                        # thêm vô cái history 
                        self.history[obj_id] = current_distance

                        # hiển thị 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{obj_id} | D:{current_distance:.1f}m | {status}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Robot Vision System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VisionSystem()
    app.run()