import os
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split

# Đọc file tọa độ XYZ
def read_xyz_file(file_path):
    """Đọc file tọa độ XYZ và chuyển thành numpy array với kiểm tra lỗi"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 3:  # Đảm bảo đúng định dạng XYZ
                try:
                    values = list(map(float, values))
                    data.append(values)
                except ValueError:
                    print(f"Bỏ qua dòng lỗi trong file: {file_path}")
    return np.array(data, dtype=np.float32) if data else np.zeros((1, 3), dtype=np.float32)

# Đọc dataset
dataset_dir = r'D:\System\Videos\VideoProc_Converter_AI\extract_features'
data = []
labels = []

for file_name in os.listdir(dataset_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(dataset_dir, file_name)
        xyz_data = read_xyz_file(file_path)
        data.append(xyz_data)
        labels.append(file_name.replace('.txt', ''))  # Lấy tên file làm nhãn

# Xử lý dữ liệu: Chuẩn hóa và padding
def normalize_and_pad_sequences(data, max_length=100):
    padded_data = []
    
    # Dồn tất cả dữ liệu lại để fit MinMaxScaler
    all_samples = np.vstack(data)  # Gộp tất cả dữ liệu lại để chuẩn hóa
    scaler = MinMaxScaler()
    scaler.fit(all_samples)  # Fit trên toàn bộ tập dữ liệu

    for sample in data:
        sample_normalized = scaler.transform(sample)  # Áp dụng transform
        sample_padded = pad_sequences([sample_normalized], maxlen=max_length, padding='post', dtype='float32')
        padded_data.append(sample_padded[0])  # Giữ lại mảng đầu tiên từ pad_sequences

    return np.array(padded_data)

data_normalized_padded = normalize_and_pad_sequences(data)
print(f'Normalized and padded data shape: {data_normalized_padded.shape}')

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(data_normalized_padded, labels, test_size=0.2, random_state=42)

# Mã hóa nhãn bằng OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
y_train_encoded = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
y_test_encoded = encoder.transform(np.array(y_test).reshape(-1, 1))

print(f"y_train_encoded shape: {y_train_encoded.shape}")
print(f"y_test_encoded shape: {y_test_encoded.shape}")

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=y_train_encoded.shape[1], activation='softmax'))

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Kiểm tra mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Lưu mô hình
model.save('trained_lstm_model.h5')

# ✅ Sửa lỗi: Lưu OneHotEncoder thay vì label_encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
