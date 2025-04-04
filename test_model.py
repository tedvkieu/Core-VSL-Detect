import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load mô hình đã train
model = load_model('trained_lstm_model.keras')

# Load scaler và label_to_index
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_to_index.pkl', 'rb') as f:
    label_to_index = pickle.load(f)

# Tạo danh sách nhãn từ label_to_index
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hàm xử lý dữ liệu real-time
def preprocess_landmarks(landmarks_list, max_length=50):
    if not landmarks_list:
        return np.zeros((max_length, 63), dtype=np.float32)
    
    # Chuyển thành mảng numpy
    xyz_data = np.array([[lm.x, lm.y, lm.z] for frame in landmarks_list for lm in frame.landmark])
    xyz_data = xyz_data.reshape(-1, 21, 3)  # Shape: (timesteps, 21, 3)
    
    # Chuẩn hóa
    xyz_flat = xyz_data.reshape(-1, 63)  # Shape: (timesteps, 63)
    xyz_normalized = scaler.transform(xyz_flat)
    
    # Padding
    xyz_padded = pad_sequences([xyz_normalized], maxlen=max_length, padding='post', dtype='float32')[0]
    return xyz_padded

# Mở webcam và dự đoán real-time
cap = cv2.VideoCapture(0)
sequence = []  # Lưu trữ chuỗi frame
max_length = 50  # Phải khớp với max_length khi huấn luyện

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lật khung hình theo chiều ngang để bỏ chế độ gương
    frame = cv2.flip(frame, 1)
    
    # Chuyển ảnh sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Vẽ landmarks lên khung hình đã lật
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Thêm landmarks vào chuỗi
            sequence.append(hand_landmarks)
            
            # Giới hạn độ dài chuỗi
            if len(sequence) > max_length:
                sequence.pop(0)
            
            # Tiền xử lý và dự đoán khi có đủ frame
            if len(sequence) == max_length:
                input_data = preprocess_landmarks(sequence, max_length)
                input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, max_length, 63)
                
                # Dự đoán
                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_label = index_to_label[predicted_index]
                
                # Hiển thị nhãn dự đoán
                cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    else:
        # Nếu không phát hiện tay, dần xóa chuỗi
        if sequence:
            sequence.pop(0)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()