import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle

# Đọc file tọa độ XYZ với xử lý lỗi tốt hơn
def read_xyz_file(file_path):
    frames = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        values = list(map(float, line.split(',')))
                        if len(values) == 63:
                            frame = np.array(values).reshape((21, 3))
                            frames.append(frame)
                    except ValueError:
                        print(f"[!] Dòng lỗi trong {file_path}")
                        continue
        return np.array(frames, dtype=np.float32) if frames else np.zeros((1, 21, 3), dtype=np.float32)
    except Exception as e:
        print(f"[Lỗi] Không đọc được file {file_path}: {e}")
        return np.zeros((1, 21, 3), dtype=np.float32)


# Tải dataset với kiểm tra thư mục
def load_dataset(dataset_dir):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Thư mục không tồn tại: {dataset_dir}")
    
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

# Lật gương tọa độ
def mirror_landmarks(sample):
    mirrored_sample = sample.copy()
    mirrored_sample[:, :, 0] = -mirrored_sample[:, :, 0]  # Lật trục X
    return mirrored_sample

# Chuẩn hóa và padding dữ liệu
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

# Tăng cường dữ liệu với nhiễu nhỏ
def augment_data(sample):
    noise = np.random.normal(0, 0.01, sample.shape)
    augmented = sample + noise
    return np.clip(augmented, 0, 1)

# Đường dẫn dataset
dataset_dir = r'D:\System\Videos\VideoProc_Converter_AI\data_maked'

# Tải và xử lý dữ liệu
try:
    data, labels = load_dataset(dataset_dir)
    print(f"Total samples: {len(data)}, Unique labels: {len(set(labels))}")
    if data:
        print(f"Sample shape: {data[0].shape}")
except Exception as e:
    print(f"Lỗi khi tải dataset: {e}")
    exit()

# Tăng cường dữ liệu
mirrored_data = [mirror_landmarks(sample) for sample in data]
data.extend(mirrored_data)
labels.extend(labels)
print(f"After mirroring: {len(data)} samples")

augmented_data = []
for _ in range(1):  # Chỉ tăng cường 1 lần
    augmented_data.extend([augment_data(sample) for sample in data])
data.extend(augmented_data)
labels.extend(labels)
print(f"After augmentation: {len(data)} samples")

# Chuẩn hóa và padding
max_length = 50
data_normalized_padded, scaler = normalize_and_pad_sequences(data, max_length)
print(f"Data shape: {data_normalized_padded.shape}")

# Mã hóa nhãn
unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = to_categorical([label_to_index[label] for label in labels], num_classes=len(unique_labels))

# Xây dựng mô hình
model = Sequential()
model.add(LSTM(units=128, input_shape=(max_length, 63), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=len(unique_labels), activation='softmax'))

# Compile mô hình
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Huấn luyện mô hình
history = model.fit(
    data_normalized_padded, y_encoded,
    epochs=150,
    batch_size=16,
    verbose=1
)

# Đánh giá trên tập train
train_loss, train_accuracy = model.evaluate(data_normalized_padded, y_encoded)
print(f"Training accuracy: {train_accuracy:.4f}")

# Lưu mô hình và các bộ mã hóa
model.save('trained_lstm_model.h5')
model.save('trained_lstm_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_to_index.pkl', 'wb') as f:
    pickle.dump(label_to_index, f)

# Dự đoán và kiểm tra
predictions = model.predict(data_normalized_padded)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_encoded, axis=1)

correct = np.sum(predicted_classes == true_classes)
print(f"\n✅ Correct predictions on training data: {correct}/{len(true_classes)} ({correct/len(true_classes)*100:.2f}%)")

# Hiển thị một số dự đoán
index_to_label = {idx: label for label, idx in label_to_index.items()}
print("\n🔎 Dự đoán trên một số mẫu đầu tiên:")
for i in range(min(10, len(data))):
    predicted = index_to_label[predicted_classes[i]]
    actual = index_to_label[true_classes[i]]
    print(f"Mẫu {i}: Dự đoán = {predicted}, Thực tế = {actual}, {'✓' if predicted == actual else '✗'}")

# ================================
# ✅ KIỂM TRA RIÊNG NHÃN "Ê"
# ================================
target_label = "Ê"
label_index = label_to_index.get(target_label, None)

if label_index is not None:
    total_e_samples = np.sum(true_classes == label_index)
    correct_for_e = np.sum((predicted_classes == true_classes) & (true_classes == label_index))

    print(f"\n🔍 Kiểm tra nhãn '{target_label}':")
    print(f"Số mẫu nhãn '{target_label}': {total_e_samples}")
    print(f"Số dự đoán đúng: {correct_for_e}")
    if total_e_samples > 0:
        accuracy_for_e = (correct_for_e / total_e_samples) * 100
        print(f"🎯 Độ chính xác riêng cho nhãn '{target_label}': {accuracy_for_e:.2f}%")
    else:
        print(f"⚠️ Không có mẫu nhãn '{target_label}' trong tập dữ liệu.")

    print("\n📋 Chi tiết từng mẫu nhãn 'Ê':")
    for i in range(len(data)):
        if true_classes[i] == label_index:
            predicted = index_to_label[predicted_classes[i]]
            actual = index_to_label[true_classes[i]]
            print(f"Mẫu {i}: Dự đoán = {predicted}, Thực tế = {actual}, {'✓' if predicted == actual else '✗'}")
else:
    print(f"\n❌ Không tìm thấy nhãn '{target_label}' trong từ điển nhãn.")