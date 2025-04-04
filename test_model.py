import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# cap.release()

# # Chuyển từ BGR sang RGB vì OpenCV dùng BGR, còn Matplotlib dùng RGB
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Hiển thị bằng matplotlib
# plt.imshow(frame)
# plt.axis("off")
# plt.show()

# Load mô hình đã train
model = load_model('trained_lstm_model.h5')

# Load bộ mã hóa nhãn
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hàm xử lý dữ liệu đầu vào
def preprocess_landmarks(landmarks, max_length=100):
    xyz_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    xyz_data = pad_sequences([xyz_data], maxlen=max_length, padding='post', dtype='float32')
    return xyz_data[0]

# Mở webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển ảnh sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Tiền xử lý tọa độ bàn tay
            input_data = preprocess_landmarks(hand_landmarks)
            input_data = np.expand_dims(input_data, axis=0)  # Reshape thành (1, max_length, 3)
            
            # Dự đoán hành động tay
            prediction = model.predict(input_data)
            # Tạo mảng one-hot từ giá trị dự đoán
            one_hot_prediction = np.zeros((1, len(encoder.categories_[0])))  # Tạo mảng 1D với số cột = số lớp
            one_hot_prediction[0, np.argmax(prediction)] = 1  # Đặt giá trị tại chỉ số của lớp có xác suất cao nhất

            # Chuyển đổi mảng one-hot thành nhãn gốc
            predicted_label = encoder.inverse_transform(one_hot_prediction)[0][0]



            
            # Hiển thị nhãn dự đoán
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()