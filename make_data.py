import cv2
import mediapipe as mp
import pandas as pd
import os

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ====================
SAVE_DIR = r"D:\System\Videos\VideoProc_Converter_AI\data_maked\Cảm ơn"  # Thư mục lưu file
os.makedirs(SAVE_DIR, exist_ok=True)

collected = []
file_index = 1
recording = False  # Bắt đầu thu khi có tay

def extract_both_hands_landmarks(results):
    left_hand = right_hand = [0.0] * 63  # default padding
    for landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        lm = []
        for point in landmark.landmark:
            lm.extend([point.x, point.y, point.z])
        if label == 'Left':
            left_hand = lm
        else:
            right_hand = lm
    return left_hand + right_hand  # 126 giá trị

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # Có tay xuất hiện thì bắt đầu ghi
        features = extract_both_hands_landmarks(results)
        collected.append(features)

        # Vẽ landmark
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Mau hien tai: {file_index} - Frames: {len(collected)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and len(collected) > 0:
        # Lưu file mẫu
        df = pd.DataFrame(collected)
        file_path = os.path.join(SAVE_DIR, f"file_{file_index}.txt")
        df.to_csv(file_path, index=False)
        print(f"✅ Đã lưu {len(collected)} frame vào {file_path}")
        file_index += 1
        collected = []  # reset cho mẫu mới

    elif key == ord('q'):
        print("⛔ Kết thúc thu thập.")
        break

cap.release()
cv2.destroyAllWindows()
