import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle

# Đọc file tọa độ XYZ
def read_xyz_file(file_path):
    frames = []
    current_frame = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split()
                if len(values) == 3:
                    current_frame.append(list(map(float, values)))
            else:
                if current_frame:
                    frames.append(current_frame)
                    current_frame = []
    if current_frame:
        frames.append(current_frame)
    return np.array(frames, dtype=np.float32) if frames else np.zeros((1, 21, 3), dtype=np.float32)

# Tải dataset
def load_dataset(dataset_dir):
    data, labels = [], []
    for label_folder in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_folder)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(label_path, file_name)
                    xyz_data = read_xyz_file(file_path)
                    data.append(xyz_data)
                    labels.append(label_folder)
    return data, labels

def mirror_landmarks(sample):
    mirrored_sample = sample.copy()
    mirrored_sample[:, :, 0] = -mirrored_sample[:, :, 0]  # Lật trục X
    return mirrored_sample


# Chuẩn hóa và padding
def normalize_and_pad_sequences(data, max_length=50):
    all_samples = np.vstack([sample.reshape(-1, 63) for sample in data])
    scaler = MinMaxScaler()
    scaler.fit(all_samples)
    
    padded_data = []
    for sample in data:
        sample_flat = sample.reshape(-1, 63)
        sample_normalized = scaler.transform(sample_flat)
        sample_padded = pad_sequences([sample_normalized], maxlen=max_length, padding='post', dtype='float32')[0]
        padded_data.append(sample_padded)
    
    return np.array(padded_data), scaler

# Tăng cường dữ liệu - giảm mức độ nhiễu để tránh overfitting
def augment_data(sample):
    noise = np.random.normal(0, 0.01, sample.shape)  # Giảm nhiễu từ 0.05 xuống 0.01
    augmented = sample + noise
    return np.clip(augmented, 0, 1)

# Đường dẫn
dataset_dir = r'D:\System\Videos\VideoProc_Converter_AI\extract_features_ex'

# Tải dữ liệu

data, labels = load_dataset(dataset_dir)
print(f"Total samples: {len(data)}, Unique labels: {len(set(labels))}")
print(f"Sample shape: {data[0].shape}")

# Tăng cường dữ liệu - giảm số lượng để model có thể học tốt hơn patterns gốc
# Tạo dữ liệu mirrored
mirrored_data = [mirror_landmarks(sample) for sample in data]
data.extend(mirrored_data)
labels.extend(labels)  # Nhãn không đổi vì chỉ lật hình

print(f"After mirroring: {len(data)} samples")
augmented_data = []
for _ in range(1):  # Giảm xuống chỉ còn 1 lần tăng cường thay vì 3
    augmented_data.extend([augment_data(sample) for sample in data])
data.extend(augmented_data)
labels.extend(labels * 1)
print(f"After augmentation: {len(data)} samples")

# Chuẩn hóa và padding
max_length = 50
data_normalized_padded, scaler = normalize_and_pad_sequences(data, max_length)
print(f"Data shape: {data_normalized_padded.shape}")

# Mã hóa nhãn
unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = to_categorical([label_to_index[label] for label in labels], num_classes=len(unique_labels))

# Xây dựng mô hình - tăng capacity và giảm dropout
model = Sequential()
model.add(LSTM(units=128, input_shape=(max_length, 63), return_sequences=True))  # tăng từ 64 lên 128
model.add(Dropout(0.1))  # giảm từ 0.3 xuống 0.1 để tránh underfitting dữ liệu train
model.add(LSTM(units=64, return_sequences=False))  # tăng từ 32 lên 64
model.add(Dropout(0.1))  # giảm từ 0.3 xuống 0.1
model.add(Dense(units=128, activation='relu'))  # tăng từ 64 lên 128
model.add(Dropout(0.1))  # giảm từ 0.3 xuống 0.1
model.add(Dense(units=len(unique_labels), activation='softmax'))

# Compile với learning rate cao hơn để học nhanh hơn
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Huấn luyện - bỏ validation split và early stopping để học toàn bộ dữ liệu
history = model.fit(
    data_normalized_padded, y_encoded,
    epochs=150,  # tăng epochs lên để đảm bảo học đủ
    batch_size=16,  # giảm batch size để học chi tiết hơn
    verbose=1
)

# Kiểm tra độ chính xác trên tập train
train_loss, train_accuracy = model.evaluate(data_normalized_padded, y_encoded)
print(f"Training accuracy: {train_accuracy:.4f}")

# Lưu mô hình - sử dụng cả 2 định dạng để đảm bảo tương thích
model.save('trained_lstm_model.h5')  # Định dạng H5 có thể tương thích tốt hơn
model.save('trained_lstm_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_to_index.pkl', 'wb') as f:
    pickle.dump(label_to_index, f)

# Kiểm tra dự đoán trên một số mẫu đã train
predictions = model.predict(data_normalized_padded)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_encoded, axis=1)

# Tính tỷ lệ dự đoán đúng
correct = np.sum(predicted_classes == true_classes)
print(f"Correct predictions on training data: {correct}/{len(true_classes)} ({correct/len(true_classes)*100:.2f}%)")

# Hiển thị một số dự đoán để kiểm tra
index_to_label = {idx: label for label, idx in label_to_index.items()}
print("\nDự đoán trên một số mẫu:")
for i in range(min(10, len(data))):
    predicted = index_to_label[predicted_classes[i]]
    actual = index_to_label[true_classes[i]]
    print(f"Mẫu {i}: Dự đoán = {predicted}, Thực tế = {actual}, {'✓' if predicted == actual else '✗'}")