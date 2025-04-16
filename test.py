import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
from PIL import ImageFont, ImageDraw, Image

# Load model và label
model = load_model("model.h5")
labels = np.load("labels.npy")
FONT_PATH = "ARIAL.TTF"
font = ImageFont.truetype(FONT_PATH, 32)  # Kích cỡ chữ
# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


# Hàm trích xuất 126 điểm từ 2 tay
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

    return left_hand + right_hand  # Tổng cộng 126 phần tử

# Lưu trữ 10 frame gần nhất
SEQ_LENGTH = 10
sequence = []

# Mở webcam
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

        show_prediction = False  # Cờ để kiểm tra xem có nên predict không

        # Vẽ landmarks nếu có
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                color = (0, 255, 0) if handedness.classification[0].label == 'Left' else (255, 0, 0)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=color, thickness=2),
                                       mp_draw.DrawingSpec(color=(255,255,255), thickness=1))

            # Nếu có ít nhất 1 tay, mới xử lý keypoints
            keypoints = extract_both_hands_landmarks(results)
            if any(k != 0 for k in keypoints):  # Chỉ thêm nếu có tay thật
                sequence.append(keypoints)
                if len(sequence) > SEQ_LENGTH:
                    sequence.pop(0)
                show_prediction = True  # Bật cờ

        # Dự đoán nếu đã đủ 10 frame và có tay
        if show_prediction and len(sequence) == SEQ_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)  # (1, 10, 126)
            prediction = model.predict(input_data, verbose=0)[0]  # shape: (num_labels,)
            max_index = np.argmax(prediction)
            max_label = labels[max_index]
            confidence = prediction[max_index]

            if confidence > 0.8:
                # Chuyển frame (numpy array) sang PIL để xử lý Unicode
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                
                # Vẽ text tiếng Việt
                text = f"{max_label} ({confidence*100:.1f}%)"
                draw.text((10, 10), text, font=font, fill=(0, 255, 255))

                # Chuyển ngược lại sang OpenCV
                frame = np.array(img_pil)

        cv2.imshow("VSL Real-time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
