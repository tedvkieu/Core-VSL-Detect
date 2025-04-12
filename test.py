import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import os

# ========== Cấu hình ==========
MODEL_PATH = "best_gesture_model.h5"  # Đường dẫn đến model đã train
LABEL_MAP_PATH = "label_map.pkl"
TIMESTEPS = 100  # Sử dụng giá trị TIMESTEPS giống với khi train
SMOOTHING_WINDOW = 5  # Số frame để làm mịn kết quả dự đoán
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng độ tin cậy để hiển thị nhãn
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480  # Kích thước khung hình hiển thị

# Cố gắng tìm font hỗ trợ tiếng Việt
FONT_CANDIDATES = [
    "arial.ttf",  # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/Library/Fonts/Arial Unicode.ttf",  # macOS
    "NotoSansCJK-Regular.ttc"  # Google Noto Fonts
]

# ========== Khởi tạo MediaPipe Hands ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ========== Load mô hình và label_map ==========
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Đã load model từ {MODEL_PATH}")
    
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    
    # Đổi ngược để lấy tên nhãn
    label_names = {v: k for k, v in label_map.items()}
    print(f"✅ Các nhãn: {list(label_map.keys())}")
except Exception as e:
    print(f"❌ Lỗi khi load model hoặc label map: {e}")
    exit(1)

# ========== Khởi tạo font tiếng Việt ==========
font = None
for font_path in FONT_CANDIDATES:
    try:
        font = ImageFont.truetype(font_path, 40)
        print(f"✅ Đã load font từ {font_path}")
        break
    except IOError:
        continue

if font is None:
    print("⚠️ Không tìm thấy font hỗ trợ tiếng Việt. Sử dụng font mặc định.")
    font = ImageFont.load_default()

# ========== Hàm xử lý dữ liệu tay ==========
def process_hands(frame):
    """
    Lấy tọa độ các điểm mốc của 2 tay từ frame.
    Trả về mảng 126 phần tử (2 tay x 21 điểm x 3 tọa độ).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    data = np.zeros(126)  # Mặc định tất cả bằng 0

    if results.multi_hand_landmarks:
        # Dictionary để lưu tạm dữ liệu của tay trái và phải
        hand_data = {"Left": None, "Right": None}
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2:  # Chỉ xử lý tối đa 2 tay
                break
                
            # Xác định tay trái hay phải
            handedness = "Left"  # Mặc định là tay trái
            if results.multi_handedness and len(results.multi_handedness) > idx:
                hand_type = results.multi_handedness[idx].classification[0].label
                handedness = "Left" if hand_type == "Left" else "Right"
            
            # Lưu dữ liệu vào dictionary
            hand_data[handedness] = hand_landmarks
            
            # Vẽ các điểm mốc lên frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Đặt dữ liệu vào vị trí đúng trong mảng
        if hand_data["Left"]:
            for i, landmark in enumerate(hand_data["Left"].landmark):
                data[i*3] = landmark.x
                data[i*3 + 1] = landmark.y  
                data[i*3 + 2] = landmark.z
                
        if hand_data["Right"]:
            for i, landmark in enumerate(hand_data["Right"].landmark):
                offset = 63  # Offset cho tay phải: 21 điểm * 3 giá trị = 63
                data[offset + i*3] = landmark.x
                data[offset + i*3 + 1] = landmark.y
                data[offset + i*3 + 2] = landmark.z

    return frame, data, results.multi_hand_landmarks is not None

# ========== Hàm chuẩn hóa dữ liệu ==========
def normalize_data(sequence_array):
    """
    Chuẩn hóa dữ liệu giống như khi train.
    """
    # Chuẩn hóa tọa độ x, y theo phương pháp min-max scaling
    sequence_array[:, ::3] = np.clip(sequence_array[:, ::3], 0, 1)  # x
    sequence_array[:, 1::3] = np.clip(sequence_array[:, 1::3], 0, 1)  # y
    
    return sequence_array

# ========== Hàm hiển thị nhãn tiếng Việt ==========
def draw_text_pil(frame, text, position, text_color=(0, 255, 0, 255), bg_color=(0, 0, 0, 160)):
    """
    Vẽ văn bản tiếng Việt lên frame bằng PIL với background.
    """
    # Chuyển frame sang PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    # Tính kích thước văn bản
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    
    # Vẽ background cho văn bản
    x, y = position
    draw.rectangle(
        [x, y, x + text_width + 10, y + text_height + 10],
        fill=bg_color
    )
    
    # Vẽ văn bản
    draw.text((x + 5, y + 5), text, font=font, fill=text_color)
    
    # Chuyển lại sang OpenCV format
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# ========== Hàm dự đoán ==========
def predict_gesture(sequence_array):
    """
    Dự đoán nhãn từ sequence.
    """
    # Đảm bảo đúng shape
    prediction_input = np.expand_dims(sequence_array, axis=0)
    
    # Dự đoán
    pred = model.predict(prediction_input, verbose=0)
    label_idx = np.argmax(pred, axis=1)[0]
    confidence = pred[0][label_idx]
    
    return label_names.get(label_idx, "Không xác định"), confidence

# ========== Main function ==========
def main():
    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    
    if not cap.isOpened():
        print("❌ Không thể mở webcam")
        return
    
    # Lưu trữ sequence của các frame
    sequence = deque(maxlen=TIMESTEPS)
    
    # Lưu trữ kết quả dự đoán gần đây để làm mịn
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    
    # Biến để theo dõi tay đã xuất hiện chưa
    hands_detected_frames = 0
    
    # Biến để hiển thị kết quả
    current_label = "Đang phát hiện..."
    current_confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc khung hình từ webcam")
            break
        
        # Lật frame để tránh hiệu ứng gương
        frame = cv2.flip(frame, 1)
        
        # Xử lý tay và lấy dữ liệu
        frame, hand_data, has_hands = process_hands(frame)
        
        # Đếm số frame liên tiếp có tay
        if has_hands:
            hands_detected_frames += 1
        else:
            hands_detected_frames = 0
        
        # Thêm dữ liệu vào sequence
        sequence.append(hand_data)
        
        # Chỉ dự đoán khi đủ TIMESTEPS frames và có ít nhất 1 tay
        if len(sequence) == TIMESTEPS and has_hands:
            # Chuyển sequence thành numpy array
            sequence_array = normalize_data(np.array(sequence))
            
            # Dự đoán
            label, confidence = predict_gesture(sequence_array)
            
            # Thêm vào lịch sử dự đoán
            prediction_history.append((label, confidence))
            
            # Tính dự đoán cuối cùng dựa trên lịch sử
            if len(prediction_history) > 0:
                # Đếm số lần xuất hiện của mỗi nhãn
                prediction_counts = {}
                confidence_sum = {}
                
                for p, c in prediction_history:
                    if p not in prediction_counts:
                        prediction_counts[p] = 0
                        confidence_sum[p] = 0
                    prediction_counts[p] += 1
                    confidence_sum[p] += c
                
                # Lấy nhãn xuất hiện nhiều nhất
                max_count = 0
                max_label = "Không xác định"
                max_confidence = 0.0
                
                for p, count in prediction_counts.items():
                    if count > max_count:
                        max_count = count
                        max_label = p
                        max_confidence = confidence_sum[p] / count
                
                # Chỉ cập nhật nếu đủ độ tin cậy
                if max_confidence >= CONFIDENCE_THRESHOLD:
                    current_label = max_label
                    current_confidence = max_confidence
        
        # Hiển thị trạng thái hiện tại
        status_text = f"{current_label} ({current_confidence:.2f})" if current_confidence > 0 else current_label
        
        # Hiển thị thanh tiến trình khi đang thu thập dữ liệu
        if has_hands and len(sequence) < TIMESTEPS:
            progress = f"Đang thu thập dữ liệu: {len(sequence)}/{TIMESTEPS}"
            frame = draw_text_pil(frame, progress, (10, 60), text_color=(255, 255, 0, 255))
            
            # Vẽ thanh tiến trình
            progress_width = int((len(sequence) / TIMESTEPS) * 200)
            cv2.rectangle(frame, (10, 100), (10 + progress_width, 115), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 100), (210, 115), (255, 255, 255), 2)
        
        # Hiển thị nhãn trên frame
        frame = draw_text_pil(frame, status_text, (10, 10))
        
        # Hiển thị khung hình
        cv2.imshow("Nhận dạng cử chỉ tiếng Việt", frame)
        
        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()