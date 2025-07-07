import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
from datetime import datetime
from collections import deque
import threading
import time

class YOLOStyleVSLDetector:
    def __init__(self, model_path="../models/model_lstm_30_6.keras", 
                 labels_path="../models/labels_lstm_30_6.npy",
                 max_timesteps=30, min_timesteps=5, confidence_threshold=0.7,
                 log_filename="vsl_predictions.txt"):
        """
        YOLO-style VSL Detector - dự đoán liên tục và ghi log vào file .txt
        """
        # Load model và labels
        self.model = load_model(model_path)
        self.labels = np.load(labels_path)
        self.max_timesteps = max_timesteps
        self.min_timesteps = min_timesteps
        self.confidence_threshold = confidence_threshold
        
        # Rolling buffer - luôn giữ lại frames để dự đoán liên tục
        self.frame_buffer = deque(maxlen=max_timesteps)
        
        # Real-time prediction results
        self.current_prediction = {
            "label": "No hands",
            "confidence": 0.0,
            "timestamp": time.time(),
            "buffer_size": 0
        }
        
        # Threading cho continuous prediction
        self.prediction_thread = None
        self.is_running = False
        self.buffer_lock = threading.Lock()
        self.prediction_lock = threading.Lock()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=2,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Smoothing và tracking
        self.prediction_history = deque(maxlen=10)  # Lưu nhiều prediction để smooth
        self.stable_prediction = "No hands"
        self.last_action_time = 0
        self.action_cooldown = 2.0  # Cooldown 2 giây giữa các action giống nhau
        
        # Statistics
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Logging setup
        self.log_filename = log_filename
        self.log_lock = threading.Lock()
        self.last_logged_prediction = None
        self.setup_logging()
        
    def setup_logging(self):
        """Khởi tạo file log"""
        os.makedirs("logs", exist_ok=True)
        self.log_filepath = os.path.join("logs", self.log_filename)
        
        # Tạo header cho file log nếu chưa tồn tại
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, 'w', encoding='utf-8') as f:
                f.write("=== VSL Detection Log ===\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Format: [TIMESTAMP] LABEL (CONFIDENCE)\n")
                f.write("-" * 50 + "\n")
        
        print(f"Logging to: {self.log_filepath}")
    
    def log_prediction(self, label, confidence):
        """Ghi prediction vào file log"""
        # Tránh ghi lặp lại cùng một prediction
        current_pred = f"{label}_{confidence:.2f}"
        if current_pred == self.last_logged_prediction:
            return
        
        self.last_logged_prediction = current_pred
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Milliseconds
        log_entry = f"[{timestamp}] {label} ({confidence:.3f})\n"
        
        with self.log_lock:
            try:
                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                    f.flush()  # Ensure immediate write
                print(f"Logged: {label} ({confidence:.3f})")
            except Exception as e:
                print(f"Error writing to log: {e}")
    
    def extract_both_hands_landmarks(self, results):
        """Extract landmarks cho cả 2 tay - 126 features"""
        left_hand = [0.0] * 63
        right_hand = [0.0] * 63
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                if handedness.classification[0].label == 'Left':
                    left_hand = coords
                else:
                    right_hand = coords
        
        return left_hand + right_hand
    
    def add_frame(self, landmarks):
        """Thêm frame vào rolling buffer"""
        with self.buffer_lock:
            self.frame_buffer.append(np.array(landmarks, dtype=np.float32))
    
    def predict_continuous(self):
        """Dự đoán liên tục với buffer hiện tại - không cần chờ đủ frames"""
        with self.buffer_lock:
            current_buffer_size = len(self.frame_buffer)
            
            if current_buffer_size < self.min_timesteps:
                return {
                    "label": "Collecting...",
                    "confidence": 0.0,
                    "buffer_size": current_buffer_size,
                    "timestamp": time.time()
                }
            
            # Lấy sequence hiện tại
            sequence = list(self.frame_buffer)
        
        # Xử lý sequence theo kích thước khác nhau
        if len(sequence) < self.max_timesteps:
            # Padding với zeros nếu ít hơn max_timesteps
            padding_size = self.max_timesteps - len(sequence)
            padded_sequence = sequence + [np.zeros(126, dtype=np.float32)] * padding_size
            input_data = np.array(padded_sequence, dtype=np.float32)
        else:
            # Lấy max_timesteps frames gần nhất nếu nhiều hơn
            input_data = np.array(sequence[-self.max_timesteps:], dtype=np.float32)
        
        # Reshape cho model: (1, timesteps, 126)
        input_data = np.expand_dims(input_data, axis=0)
        
        try:
            # Dự đoán
            prediction = self.model.predict(input_data, verbose=0)[0]
            
            # Xử lý kết quả
            max_index = np.argmax(prediction)
            max_label = self.labels[max_index]
            confidence = float(prediction[max_index])
            
            return {
                "label": max_label,
                "confidence": confidence,
                "buffer_size": len(sequence),
                "raw_predictions": prediction,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "label": "Error",
                "confidence": 0.0,
                "buffer_size": len(sequence),
                "timestamp": time.time()
            }
    
    def continuous_prediction_loop(self):
        """Vòng lặp dự đoán liên tục - chạy với tần suất cao"""
        while self.is_running:
            try:
                # Dự đoán liên tục
                prediction_result = self.predict_continuous()
                
                if prediction_result:
                    with self.prediction_lock:
                        self.current_prediction = prediction_result
                        
                        # Smooth prediction với confidence threshold
                        if (prediction_result["confidence"] >= self.confidence_threshold and 
                            prediction_result["label"] not in ["Collecting...", "Error", "non-action"]):
                            
                            self.prediction_history.append({
                                "label": prediction_result["label"],
                                "confidence": prediction_result["confidence"],
                                "time": time.time()
                            })
                        
                        # Tính stable prediction
                        self.update_stable_prediction()
                
                # High frequency prediction - 60 FPS style
                time.sleep(1/60)  # ~16ms delay
                
            except Exception as e:
                print(f"Error in continuous prediction: {e}")
                time.sleep(0.1)
    
    def update_stable_prediction(self):
        """Cập nhật stable prediction dựa trên history"""
        if len(self.prediction_history) < 3:
            return
        
        # Lấy predictions trong 1 giây gần nhất
        current_time = time.time()
        recent_predictions = [
            p for p in self.prediction_history 
            if current_time - p["time"] < 1.0
        ]
        
        if not recent_predictions:
            return
        
        # Tìm prediction xuất hiện nhiều nhất với confidence cao
        from collections import Counter
        label_counts = Counter([p["label"] for p in recent_predictions])
        most_common_label = label_counts.most_common(1)[0]
        
        # Chỉ update nếu xuất hiện ít nhất 3 lần trong 1 giây
        if most_common_label[1] >= 3:
            avg_confidence = np.mean([
                p["confidence"] for p in recent_predictions 
                if p["label"] == most_common_label[0]
            ])
            
            if avg_confidence >= self.confidence_threshold:
                new_prediction = most_common_label[0]
                if new_prediction != self.stable_prediction:
                    self.stable_prediction = new_prediction
                    # Ghi log khi có prediction mới và stable
                    self.log_prediction(new_prediction, avg_confidence)
    
    def start_continuous_prediction(self):
        """Bắt đầu continuous prediction thread"""
        if not self.is_running:
            self.is_running = True
            self.prediction_thread = threading.Thread(
                target=self.continuous_prediction_loop, 
                daemon=True
            )
            self.prediction_thread.start()
    
    def stop_continuous_prediction(self):
        """Dừng continuous prediction"""
        self.is_running = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1)
    
    def get_current_prediction(self):
        """Lấy prediction hiện tại"""
        with self.prediction_lock:
            return self.current_prediction.copy()
    
    def calculate_fps(self):
        """Tính FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
    
    def log_manual_prediction(self, label, confidence):
        """Ghi manual prediction vào log"""
        self.log_prediction(f"MANUAL_{label}", confidence)
    
    def draw_text_with_background(self, frame, text, position, 
                                 font_color=(255, 255, 255), bg_color=(0, 0, 0),
                                 font_scale=0.7, thickness=2):
        """Vẽ text với background để dễ đọc"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Tính kích thước text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Vẽ background
        cv2.rectangle(frame, 
                     (position[0]-5, position[1]-text_size[1]-5),
                     (position[0]+text_size[0]+5, position[1]+5),
                     bg_color, -1)
        
        # Vẽ text
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
        
        return frame
    
    def process_frame(self, frame):
        """Xử lý frame - YOLO style: xử lý mọi frame"""
        if frame is None or frame.size == 0:
            return None
        
        # Calculate FPS
        self.calculate_fps()
        
        # Process với MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Extract landmarks
        landmarks = self.extract_both_hands_landmarks(results)
        has_hands = any(k != 0 for k in landmarks)
        
        if has_hands:
            # Có tay -> thêm vào buffer liên tục
            self.add_frame(landmarks)
        else:
            # Không có tay -> clear buffer sau 1 giây
            with self.buffer_lock:
                if len(self.frame_buffer) > 0:
                    # Giảm dần buffer thay vì clear ngay
                    if len(self.frame_buffer) > 10:
                        for _ in range(5):  # Remove 5 frames
                            if self.frame_buffer:
                                self.frame_buffer.popleft()
        
        # Lấy prediction hiện tại
        current_pred = self.get_current_prediction()
        
        # Hiển thị thông tin real-time
        y_offset = 30
        
        # Main prediction
        if has_hands:
            if current_pred["buffer_size"] < self.min_timesteps:
                main_text = f"Collecting... ({current_pred['buffer_size']}/{self.min_timesteps})"
                color = (0, 255, 255)  # Yellow
            else:
                if current_pred["confidence"] >= self.confidence_threshold:
                    main_text = f"{self.stable_prediction} ({current_pred['confidence']:.2f})"
                    color = (0, 255, 0)  # Green
                    
                    # Auto-log high confidence detections
                    if (current_pred["confidence"] > 0.9 and 
                        self.stable_prediction != "non-action" and
                        time.time() - self.last_action_time > self.action_cooldown):
                        
                        # Log prediction với thread để không block main loop
                        threading.Thread(
                            target=self.log_prediction,
                            args=(f"AUTO_{self.stable_prediction}", current_pred["confidence"]),
                            daemon=True
                        ).start()
                        self.last_action_time = time.time()
                else:
                    main_text = f"Low confidence ({current_pred['confidence']:.2f})"
                    color = (0, 165, 255)  # Orange
        else:
            main_text = "No hands detected"
            color = (128, 128, 128)  # Gray
        
        frame = self.draw_text_with_background(frame, main_text, (10, y_offset), color)
        y_offset += 35
        
        # Buffer info
        buffer_text = f"Buffer: {current_pred['buffer_size']}/{self.max_timesteps}"
        frame = self.draw_text_with_background(frame, buffer_text, (10, y_offset), (255, 255, 255))
        y_offset += 30
        
        # FPS info
        fps_text = f"FPS: {self.current_fps}"
        frame = self.draw_text_with_background(frame, fps_text, (10, y_offset), (255, 255, 255))
        y_offset += 30
        
        # Mode info
        mode_text = "Mode: YOLO-style with TXT Logging"
        frame = self.draw_text_with_background(frame, mode_text, (10, y_offset), (255, 0, 255))
        y_offset += 30
        
        # Log file info
        log_text = f"Log: {self.log_filename}"
        frame = self.draw_text_with_background(frame, log_text, (10, y_offset), (0, 255, 255))
        
        # Vẽ hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Color theo tay: Left=Green, Right=Red
                color = (0, 255, 0) if handedness.classification[0].label == 'Left' else (0, 0, 255)
                
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=color, thickness=2),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                
                # Hiển thị label tay
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, handedness.classification[0].label, 
                           (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def run_yolo_style_detection(self):
        """Chạy detection theo style YOLO - real-time continuous với TXT logging"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        
        # Camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Bắt đầu continuous prediction
        self.start_continuous_prediction()
        
        print("=== YOLO-Style VSL Detection with TXT Logging ===")
        print("Features:")
        print("  ✓ Continuous prediction (no waiting)")
        print("  ✓ Dynamic buffer size (5-30 frames)")
        print("  ✓ Real-time FPS display")
        print("  ✓ Auto-log high confidence detections to TXT")
        print("  ✓ Smooth prediction with history")
        print(f"  ✓ Logging to: {self.log_filepath}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'r' - Reset buffer")
        print("  'l' - Force log current prediction")
        print("  'c' - Clear detection history")
        print("  'o' - Open log file location")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                frame_count += 1
                frame = cv2.flip(frame, 1)  # Mirror effect
                
                # Process frame - YOLO style
                processed_frame = self.process_frame(frame)
                
                if processed_frame is not None:
                    cv2.imshow("YOLO-Style VSL Detection with TXT Logging", processed_frame)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffer
                    with self.buffer_lock:
                        self.frame_buffer.clear()
                    self.prediction_history.clear()
                    self.stable_prediction = "No hands"
                    print("Buffer reset!")
                elif key == ord('l'):
                    # Force log current prediction
                    current_pred = self.get_current_prediction()
                    if current_pred["buffer_size"] >= self.min_timesteps:
                        threading.Thread(
                            target=self.log_manual_prediction,
                            args=(current_pred["label"], current_pred["confidence"]),
                            daemon=True
                        ).start()
                        print("Manual log triggered!")
                elif key == ord('c'):
                    # Clear history
                    self.prediction_history.clear()
                    print("Detection history cleared!")
                elif key == ord('o'):
                    # Open log file location
                    import subprocess
                    import platform
                    try:
                        if platform.system() == "Windows":
                            subprocess.run(f'explorer /select,"{os.path.abspath(self.log_filepath)}"')
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", "-R", os.path.abspath(self.log_filepath)])
                        else:  # Linux
                            subprocess.run(["xdg-open", os.path.dirname(os.path.abspath(self.log_filepath))])
                        print("Opened log file location!")
                    except Exception as e:
                        print(f"Could not open file location: {e}")
                        print(f"Log file is at: {os.path.abspath(self.log_filepath)}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        except Exception as e:
            print(f"Error in detection: {e}")
        
        finally:
            # Cleanup và log kết thúc
            with self.log_lock:
                try:
                    with open(self.log_filepath, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        f.write("-" * 50 + "\n\n")
                except Exception as e:
                    print(f"Error writing end log: {e}")
            
            self.stop_continuous_prediction()
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped!")
            print(f"All predictions logged to: {os.path.abspath(self.log_filepath)}")


# Main execution
if __name__ == "__main__":
    # Khởi tạo YOLO-style detector với TXT logging
    detector = YOLOStyleVSLDetector(
        model_path="../models/model_lstm_30_6.keras",
        labels_path="../models/labels_lstm_30_6.npy",
        max_timesteps=30,      # Maximum buffer size
        min_timesteps=5,       # Minimum frames to start prediction
        confidence_threshold=0.7,  # Lower threshold for more responsive detection
        log_filename="vsl_predictions.txt"  # Tên file log
    )
    
    # Chạy detection theo style YOLO với TXT logging
    detector.run_yolo_style_detection()