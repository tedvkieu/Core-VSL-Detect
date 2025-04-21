import cv2
import mediapipe as mp
import pandas as pd
import os
import time
from datetime import datetime

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam. Vui lòng kiểm tra lại.")
    exit()

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ====================
SAVE_DIR = r"D:\System\Videos\VideoProc_Converter_AI\make_data\R"  # Thư mục lưu file
os.makedirs(SAVE_DIR, exist_ok=True)

collected = []
file_index = 1
recording = False  # Trạng thái thu thập

def extract_both_hands_landmarks(results):
    """Trích xuất landmark từ cả 2 bàn tay, trả về mảng 126 phần tử"""
    left_hand = [0.0] * 63  # default padding  
    right_hand = [0.0] * 63  # default padding
    
    if results.multi_hand_landmarks:
        for i, (landmark, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            if i >= 2:  # Chỉ lấy tối đa 2 tay
                break
                
            label = handedness.classification[0].label
            lm = []
            for point in landmark.landmark:
                lm.extend([point.x, point.y, point.z])
                
            if label == 'Left':
                left_hand = lm
            else:
                right_hand = lm
                
    return left_hand + right_hand  # 126 giá trị

try:
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ webcam. Thoát...")
            break

        frame = cv2.flip(frame, 1)  # Lật ngang để dễ thao tác
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Hiển thị khung hình
        h, w, c = frame.shape
        
        # Vẽ khung landmark tay nếu phát hiện
        if results.multi_hand_landmarks:
            for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                # Màu khác nhau cho tay trái và tay phải
                color = (0, 255, 0) if label == 'Left' else (255, 0, 0)
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS, 
                                      mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                      mp_draw.DrawingSpec(color=(255,255,255), thickness=1))
                
                # Hiển thị nhãn tay
                x, y = int(hand_landmark.landmark[0].x * w), int(hand_landmark.landmark[0].y * h)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Thu thập dữ liệu nếu đang ghi và phát hiện tay
            if recording:
                features = extract_both_hands_landmarks(results)
                collected.append(features)

        # Hiển thị thông tin trạng thái
        elapsed_time = int(time.time() - start_time)
        cv2.putText(frame, f"Mẫu hiện tại: {file_index} - Frames: {len(collected)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Thời gian: {elapsed_time}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        # Hiển thị trạng thái thu thập
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        status_text = "Collecting" if recording else "Stop" 
        cv2.putText(frame, status_text, (w-200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Hiển thị phím tắt
        cv2.putText(frame, "s: Start/Stop | a: Save | q: Quit", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Make data VSL", frame)
        key = cv2.waitKey(1) & 0xFF

        # Xử lý phím tắt
        if key == ord('s'):  # Bắt đầu/dừng thu thập
            recording = not recording
            status = "Collecting" if recording else "Stop"
            print(f"Trạng thái thu thập: {status}")
        
        elif key == ord('a') and len(collected) > 0:
            # Lưu file mẫu
            df = pd.DataFrame(collected)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(SAVE_DIR, f"file_{file_index}_{timestamp}.csv")
            
            # Thêm metadata
            metadata = {
                'date': timestamp,
                'frames': len(collected),
                'label': os.path.basename(SAVE_DIR)
            }
            
            # Lưu dữ liệu và metadata
            with open(file_path.replace('.csv', '_metadata.txt'), 'w', encoding='utf-8') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                    
            df.to_csv(file_path, index=False)
            print(f"✅ Đã lưu {len(collected)} frame vào {file_path}")
            file_index += 1
            collected = []  # reset cho mẫu mới

        elif key == ord('q'):
            print("⛔ Kết thúc thu thập.")
            break

except Exception as e:
    print(f"Lỗi: {e}")
finally:
    # Đảm bảo giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    
    # Kiểm tra nếu còn dữ liệu chưa lưu
    if len(collected) > 0:
        save_option = input("Còn dữ liệu chưa lưu. Bạn có muốn lưu không? (y/n): ")
        if save_option.lower() == 'y':
            df = pd.DataFrame(collected)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(SAVE_DIR, f"file_{file_index}_{timestamp}.csv")
            df.to_csv(file_path, index=False)
            print(f"✅ Đã lưu {len(collected)} frame vào {file_path}")