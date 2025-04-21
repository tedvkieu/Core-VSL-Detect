import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import mediapipe as mp

# Cấu hình
MODEL_PATH = "model.keras"  # Đường dẫn đến mô hình
LABELS_PATH = "labels.npy"  # Đường dẫn đến nhãn
MAX_TIMESTEPS = 50  # Kích thước sliding window
FEATURES = 126  # Số đặc trưng mỗi frame (21 keypoint * 3 * 2 tay)
CONFIDENCE_THRESHOLD = 0.9  # Ngưỡng confidence
MASK_VALUE = -1.0  # Giá trị padding, khớp với huấn luyện

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils  # Để vẽ keypoint

# Tải mô hình và nhãn
try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    print("✅ Đã tải mô hình và nhãn:", labels)
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    exit()

# Hàm trích xuất đặc trưng từ frame
def extract_features(frame, hands, image_width, image_height):
    """
    Trích xuất 126 đặc trưng từ keypoint tay (21 keypoint * 3 (x, y, z) * 2 tay).
    """
    # Chuyển frame sang RGB cho MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Khởi tạo vector đặc trưng
    features = np.full(FEATURES, MASK_VALUE, dtype=np.float32)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) <= 2:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Lấy nhãn tay (trái hoặc phải) từ multi_handedness
            handedness = results.multi_handedness[idx].classification[0].label  # 'Left' hoặc 'Right'
            # Nếu là tay phải, lưu vào nửa đầu (0:63); tay trái, lưu vào nửa sau (63:126)
            offset = 0 if handedness == 'Right' else 63

            # Trích xuất tọa độ keypoint
            for i, landmark in enumerate(hand_landmarks.landmark):
                # Chuẩn hóa x, y dựa trên kích thước khung hình
                x = landmark.x * image_width
                y = landmark.y * image_height
                z = landmark.z  # z là độ sâu tương đối, không chuẩn hóa
                features[offset + i*3:offset + i*3 + 3] = [x, y, z]

    return features

# Khởi tạo sliding window
window = deque(maxlen=MAX_TIMESTEPS)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định
if not cap.isOpened():
    print("❌ Lỗi: Không thể mở webcam")
    exit()

# Giảm độ phân giải webcam để tăng hiệu suất
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Lấy kích thước khung hình
image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Biến để theo dõi FPS
prev_time = time.time()
fps = 0

print("✅ Bắt đầu nhận diện thời gian thực (chế độ gương lật, vẽ keypoint tay). Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lỗi: Không thể đọc frame từ webcam")
        break

    # Lật khung hình ngang (chế độ gương)
    frame = cv2.flip(frame, 1)

    # Tính FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Trích xuất đặc trưng từ frame
    features = extract_features(frame, hands, image_width, image_height)

    # Thêm đặc trưng vào sliding window
    window.append(features)

    # Vẽ keypoint tay lên frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    # Chuẩn bị dữ liệu cho dự đoán
    if len(window) == MAX_TIMESTEPS:
        # Chuyển window thành numpy array
        window_array = np.array(window, dtype=np.float32)

        # Đảm bảo đúng shape (1, MAX_TIMESTEPS, FEATURES)
        input_data = window_array.reshape(1, MAX_TIMESTEPS, FEATURES)

        # Dự đoán
        try:
            predictions = model.predict(input_data, verbose=0)
            confidence = np.max(predictions)
            predicted_label_idx = np.argmax(predictions)
            predicted_label = labels[predicted_label_idx]

            # Hiển thị kết quả nếu confidence > ngưỡng
            if confidence > CONFIDENCE_THRESHOLD:
                label_text = f"{predicted_label} ({confidence:.2f})"
            else:
                label_text = "Không xác định"
        except Exception as e:
            label_text = f"Lỗi dự đoán: {e}"
    else:
        label_text = "Đang thu thập dữ liệu..."

    # Hiển thị thông tin trên frame
    cv2.putText(frame, f"Label: {label_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Realtime Recognition (Mirror Mode, Hand Landmarks)", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
hands.close()
cv2.destroyAllWindows()
print("✅ Đã dừng nhận diện.")