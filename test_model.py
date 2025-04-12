import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ========== Cấu hình ==========
MODEL_PATH = "gesture_model_transformer.h5"
LABEL_MAP_PATH = "label_map.pkl"
TIMESTEPS = 50  # Phải khớp với train
FONT_PATH = "arial.ttf"  # Đường dẫn đến font hỗ trợ tiếng Việt
FONT_SIZE = 40
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
CONFIDENCE_THRESHOLD = 0.7
TWO_HAND_LABEL = "Cảm ơn"

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load mô hình và label_map
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)
label_names = {v: k for k, v in label_map.items()}

# Khởi tạo font tiếng Việt
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print("[ERROR] Không tìm thấy font. Sử dụng font mặc định.")
    font = ImageFont.load_default()

# ========== Hàm xử lý dữ liệu tay ==========
def process_hands(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    data = np.zeros(126)
    hands_detected = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            handedness = results.multi_handedness[idx].classification[0].label
            start_idx = 0 if handedness == 'Left' else 63
            for i, landmark in enumerate(hand_landmarks.landmark):
                data[start_idx + i*3] = landmark.x
                data[start_idx + i*3 + 1] = landmark.y
                data[start_idx + i*3 + 2] = landmark.z
            hands_detected += 1
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, data, hands_detected

# ========== Hàm kiểm tra sequence cho nhãn 2 tay ==========
def check_two_hands_sequence(sequence):
    left_hands = sequence[:, :63]
    right_hands = sequence[:, 63:]
    return not (np.all(left_hands == 0) or np.all(right_hands == 0))

# ========== Hàm hiển thị nhãn tiếng Việt ==========
def draw_text_pil(frame, text, position):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text(position, text, font=font, fill=(0, 255, 0, 255))
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# ========== Hàm dự đoán ==========
def predict_gesture(sequence):
    sequence = np.expand_dims(sequence, axis=0)  # Thêm batch dimension
    pred = model.predict(sequence, verbose=0)
    label_idx = np.argmax(pred, axis=1)[0]
    confidence = pred[0][label_idx]
    return label_names.get(label_idx, "Không xác định"), confidence

# ========== Main loop ==========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

sequence = deque(maxlen=TIMESTEPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Không thể đọc từ webcam")
        break

    frame = cv2.flip(frame, 1)  # Lật frame để tránh hiệu ứng gương
    frame, hand_data, hands_detected = process_hands(frame, hands)
    sequence.append(hand_data)

    if len(sequence) == TIMESTEPS:
        sequence_array = np.array(sequence)
        # Chuẩn hóa dữ liệu giống khi train
        sequence_array[:, ::3] = np.clip(sequence_array[:, ::3], 0, 1)  # x
        sequence_array[:, 1::3] = np.clip(sequence_array[:, 1::3], 0, 1)  # y

        # Kiểm tra sequence cho nhãn 'Cảm ơn'
        if check_two_hands_sequence(sequence_array):
            label, confidence = predict_gesture(sequence_array)
            if label == TWO_HAND_LABEL and hands_detected < 2:
                label, confidence = "Không đủ tay", 0.0
        else:
            label, confidence = predict_gesture(sequence_array)
            if label == TWO_HAND_LABEL:
                label, confidence = "Không đủ tay", 0.0

        # Hiển thị nhãn nếu độ tin cậy đủ cao
        if confidence >= CONFIDENCE_THRESHOLD:
            text = f"{label} ({confidence:.2%})"
        else:
            text = "Không xác định"

        frame = draw_text_pil(frame, text, (10, 30))
        
        # Đặt lại sequence sau mỗi dự đoán
        sequence.clear()

    # Hiển thị frame
    cv2.imshow("Real-time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
hands.close()