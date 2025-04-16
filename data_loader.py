import cv2
import mediapipe as mp
import pandas as pd
import os

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

label = "HAND_GESTURE"  # Nhãn cho dữ liệu
output_dir = f"data/{label}"  # Thư mục lưu file
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

file_index = 1  # Chỉ số file
lm_list = []  # Danh sách lưu landmarks mỗi frame
frame_count = 0  # Đếm số frame trong file hiện tại
is_collecting = False  # Trạng thái thu thập

def make_landmark_timestep(results):
    global frame_count
    frame_count += 1
    c_lm = [0] * 126  # 21 landmarks x 3 (x,y,z) x 2 tay = 126 phần tử

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2:  # Chỉ xử lý tối đa 2 tay
                break
            # Xác định tay trái hay phải
            handedness = results.multi_handedness[idx].classification[0].label
            offset = 0 if handedness == "Right" else 63  # Right: 0-62, Left: 63-125

            for id, lm in enumerate(hand_landmarks.landmark):
                c_lm[offset + id*3] = lm.x
                c_lm[offset + id*3 + 1] = lm.y
                c_lm[offset + id*3 + 2] = lm.z

    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img

def save_to_file():
    global lm_list, file_index, frame_count
    if lm_list:
        df = pd.DataFrame(lm_list)
        output_file = os.path.join(output_dir, f"{label}_{file_index}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {frame_count} frames to {output_file}")
        lm_list = []
        file_index += 1
        frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện hands
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    # Kiểm tra trạng thái thu thập
    if results.multi_hand_landmarks:  # Có ít nhất một tay
        is_collecting = True
        # Ghi nhận thông số landmarks
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        # Vẽ landmarks lên ảnh
        frame = draw_landmark_on_image(mpDraw, results, frame)
    else:  # Không có tay nào
        is_collecting = False

    # Hiển thị thông tin
    status_text = "Collecting" if is_collecting else "Waiting for hand"
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"File: {file_index}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and is_collecting:  # Nhấn 'a' để lưu file hiện tại và bắt đầu file mới, chỉ khi đang thu thập
        save_to_file()
    elif key == ord('q'):  # Nhấn 'q' để thoát
        save_to_file()  # Lưu file hiện tại trước khi thoát
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()