import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ⚙️ Cấu hình
MAX_LEN = 300
FEATURE_DIM = 126
WINDOW_SIZE = MAX_LEN

# 🎯 Load model và label
model = tf.keras.models.load_model("model.h5")
labels = np.load("labels.npy")

# 📦 Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 📹 Khởi tạo webcam
cap = cv2.VideoCapture(0)

sequence = []

def extract_hand_landmarks(results):
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    # Pad nếu chỉ có 1 tay
    while len(landmarks) < FEATURE_DIM:
        landmarks.extend([0] * (FEATURE_DIM - len(landmarks)))
    return landmarks[:FEATURE_DIM]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        landmarks = extract_hand_landmarks(results)
        sequence.append(landmarks)

        # chỉ khi đủ độ dài chuỗi mới dự đoán
        if len(sequence) >= WINDOW_SIZE:
            input_seq = np.array(sequence[-WINDOW_SIZE:])

            # reshape và thêm batch
            input_seq = input_seq[np.newaxis, :, :]

            pred = model.predict(input_seq)[0]
            pred_label = labels[np.argmax(pred)]
            confidence = np.max(pred)

            # Hiển thị nhãn lên màn hình
            cv2.putText(image, f"{pred_label} ({confidence:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Hand Action Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
