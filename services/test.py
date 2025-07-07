import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
import os
from datetime import datetime

# Load model và label
model = load_model("models\model_lstm_12_6.h5")
labels = np.load("models\labels_lstm_12_6.npy")
FONT_PATH = "ARIAL.TTF"
font = ImageFont.truetype(FONT_PATH, 32)

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_both_hands_landmarks(results):
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

def save_to_csv(sequence, prediction):
    # Tạo tên file dựa trên thời gian hiện tại
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hand_data_{timestamp}.csv"
    
    # Tạo DataFrame từ sequence
    df = pd.DataFrame(sequence)
    
    # Thêm cột prediction
    df['prediction'] = prediction
    
    # Lưu vào file CSV
    df.to_csv(filename, index=False)
    print(f"Đã lưu dữ liệu vào file: {filename}")

sequence = []
collecting = False
last_prediction = ""
hand_missing_counter = 0
missing_threshold = 10  # số frame liên tục không thấy tay

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        keypoints = extract_both_hands_landmarks(results)

        if any(k != 0 for k in keypoints):
            # Phát hiện tay -> bắt đầu thu frame
            sequence.append(keypoints)
            collecting = True
            hand_missing_counter = 0

        else:
            if collecting:
                hand_missing_counter += 1
                if hand_missing_counter >= missing_threshold:
                    # Tay biến mất -> thực hiện dự đoán ngay
                    if len(sequence) > 0:
                        input_data = np.expand_dims(sequence, axis=0)
                        print("input_data:" , input_data)
                        prediction = model.predict(input_data, verbose=0)[0]
                        max_index = np.argmax(prediction)
                        max_label = labels[max_index]
                        confidence = prediction[max_index]

                        print("Dự đoán:", max_label, "(", confidence, ")")
                        if confidence > 0.9 and max_label != "non-action":
                            last_prediction = max_label
                            # Lưu dữ liệu vào CSV khi có dự đoán hợp lệ
                            save_to_csv(sequence, max_label)
                        else:
                            last_prediction = ""

                    # Reset trạng thái
                    sequence.clear()
                    collecting = False
                    hand_missing_counter = 0

        # Hiển thị nhãn
        display_text = last_prediction if last_prediction else "..."
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), display_text, font=font, fill=(0, 255, 255))
        frame = np.array(img_pil)

        # Vẽ landmarks nếu có
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                color = (0, 255, 0) if handedness.classification[0].label == 'Left' else (255, 0, 0)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=color, thickness=2),
                                       mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1))

        cv2.imshow("VSL Real-time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
