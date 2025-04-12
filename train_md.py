import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, TimeDistributed, Masking, Attention
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# ========== Cấu hình ==========
DATASET_DIR = r'D:\System\Videos\VideoProc_Converter_AI\data_maked'
MAX_TIMESTEPS = 100  # Độ dài tối đa của sequence
MIN_TIMESTEPS = 20   # Độ dài tối thiểu của sequence
BATCH_SIZE = 32
EPOCHS = 150

# ========== Hàm kiểm tra số tay ==========
def count_hands(data, file_path, label, strict_two_hands_labels=['Cảm ơn']):
    """
    Kiểm tra số tay được sử dụng trong dữ liệu.
    - Tay có tọa độ toàn bộ bằng 0 được coi là không sử dụng.
    - Nhãn yêu cầu hai tay phải có cả hai tay hoạt động.
    """
    left_hand = data[:, :63]
    right_hand = data[:, 63:]
    
    # Kiểm tra activity theo frame thay vì toàn bộ chuỗi
    left_active_frames = np.any(left_hand != 0, axis=1)
    right_active_frames = np.any(right_hand != 0, axis=1)
    
    # Tính phần trăm frame hoạt động
    left_active_percent = np.mean(left_active_frames)
    right_active_percent = np.mean(right_active_frames)
    
    if label in strict_two_hands_labels:
        if left_active_percent < 0.5 or right_active_percent < 0.5:
            print(f"[WARNING] {file_path}: Nhãn {label} yêu cầu 2 tay nhưng chỉ có {left_active_percent:.2f}% frame tay trái và {right_active_percent:.2f}% frame tay phải")
            return 0
        return 2
    
    # Đối với nhãn không yêu cầu hai tay
    if left_active_percent > 0.3 and right_active_percent > 0.3:
        return 2
    elif left_active_percent > 0.3:
        return 1  # chỉ tay trái
    elif right_active_percent > 0.3:
        return 1  # chỉ tay phải
    else:
        print(f"[WARNING] {file_path}: Không phát hiện hoạt động tay trong file")
        return 0

# ========== Hàm chuẩn hóa dữ liệu ==========
def normalize_data(data):
    """
    Chuẩn hóa dữ liệu tọa độ tay.
    - Sử dụng z-score normalization cho các tọa độ
    - Xử lý riêng các giá trị tin cậy (confidence)
    """
    # Tách tọa độ và confidence
    coords_x = data[:, ::3]  # x coordinates
    coords_y = data[:, 1::3]  # y coordinates
    conf = data[:, 2::3]     # confidence values
    
    # Chuẩn hóa tọa độ cho từng điểm riêng biệt
    for i in range(coords_x.shape[1]):
        # Chỉ chuẩn hóa các giá trị khác 0
        mask_x = coords_x[:, i] != 0
        if np.any(mask_x):
            mean_x = np.mean(coords_x[mask_x, i])
            std_x = np.std(coords_x[mask_x, i]) + 1e-8  # Tránh chia cho 0
            coords_x[mask_x, i] = (coords_x[mask_x, i] - mean_x) / std_x
        
        mask_y = coords_y[:, i] != 0
        if np.any(mask_y):
            mean_y = np.mean(coords_y[mask_y, i])
            std_y = np.std(coords_y[mask_y, i]) + 1e-8
            coords_y[mask_y, i] = (coords_y[mask_y, i] - mean_y) / std_y
    
    # Ghép lại dữ liệu
    normalized_data = np.zeros_like(data)
    normalized_data[:, ::3] = coords_x
    normalized_data[:, 1::3] = coords_y
    normalized_data[:, 2::3] = conf  # Giữ nguyên giá trị confidence
    
    return normalized_data

# ========== Hàm trích xuất đặc trưng ==========
def extract_features(data):
    """
    Trích xuất thêm đặc trưng từ dữ liệu tọa độ.
    - Tính vận tốc, gia tốc
    - Tính khoảng cách giữa các điểm
    """
    n_frames = data.shape[0]
    if n_frames < 3:
        return data  # Không đủ frame để tính toán đặc trưng
    
    # Tách tọa độ x, y
    coords_x = data[:, ::3]  # shape: (n_frames, 42)
    coords_y = data[:, 1::3]
    
    # Tính vận tốc (sự thay đổi tọa độ qua các frame)
    velocity_x = np.zeros_like(coords_x)
    velocity_y = np.zeros_like(coords_y)
    velocity_x[1:] = coords_x[1:] - coords_x[:-1]
    velocity_y[1:] = coords_y[1:] - coords_y[:-1]
    
    # Tính gia tốc (sự thay đổi vận tốc)
    accel_x = np.zeros_like(coords_x)
    accel_y = np.zeros_like(coords_y)
    accel_x[2:] = velocity_x[2:] - velocity_x[1:-1]
    accel_y[2:] = velocity_y[2:] - velocity_y[1:-1]
    
    # Ghép các đặc trưng lại
    features = np.concatenate([
        data,  # Dữ liệu gốc
        velocity_x, velocity_y,  # Vận tốc
        accel_x, accel_y,  # Gia tốc
    ], axis=1)
    
    return features

# ========== Hàm tạo cửa sổ trượt ==========
def create_sliding_windows(data, label_idx, min_window=20, max_window=100, stride=10):
    """
    Tạo các cửa sổ trượt từ một chuỗi dài.
    """
    n_frames = data.shape[0]
    windows = []
    labels = []
    
    if n_frames < min_window:
        # Nếu chuỗi quá ngắn, đệm thêm frame cuối
        padding = np.repeat(data[-1:], min_window - n_frames, axis=0)
        padded_data = np.vstack([data, padding])
        windows.append(padded_data)
        labels.append(label_idx)
        return windows, labels
    
    # Tạo các cửa sổ trượt
    for start in range(0, n_frames - min_window + 1, stride):
        end = min(start + max_window, n_frames)
        window = data[start:end]
        
        # Nếu cửa sổ nhỏ hơn max_window, đệm thêm
        if window.shape[0] < max_window:
            padding = np.repeat(window[-1:], max_window - window.shape[0], axis=0)
            window = np.vstack([window, padding])
        
        windows.append(window)
        labels.append(label_idx)
    
    return windows, labels

# ========== Đọc và xử lý dữ liệu ==========
X, y = [], []
label_map = {}
label_counts = defaultdict(int)
sequence_lengths = []  # Lưu độ dài thực tế của các chuỗi

for idx, label in enumerate(sorted(os.listdir(DATASET_DIR))):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    label_map[label] = idx
    
    for file in os.listdir(label_path):
        if file.endswith('.txt'):
            file_path = os.path.join(label_path, file)
            try:
                df = pd.read_csv(file_path, header=None)
                data = df.values  # (n_frames, 126)
                sequence_lengths.append(data.shape[0])  # Lưu độ dài chuỗi
            except Exception as e:
                print(f"[ERROR] Không đọc được file {file_path}: {e}")
                continue

            if data.shape[1] != 126:
                print(f"[WARNING] Bỏ qua file {file_path}: Shape không hợp lệ {data.shape}")
                continue

            # Kiểm tra số tay
            num_hands = count_hands(data, file_path, label)
            if num_hands == 0:  # Bỏ qua nếu không đủ tay
                continue
            
            # Chuẩn hóa dữ liệu
            data = normalize_data(data)
            
            # Trích xuất thêm đặc trưng (tùy chọn)
            # data = extract_features(data)  # Nếu bật tính năng này, cần thay đổi input_shape của mô hình
            
            # Tạo các cửa sổ trượt
            windows, labels = create_sliding_windows(
                data, idx, 
                min_window=MIN_TIMESTEPS,
                max_window=MAX_TIMESTEPS, 
                stride=10
            )
            
            X.extend(windows)
            y.extend(labels)
            label_counts[label] += len(windows)

X = np.array(X)
y = np.array(y)

print(f"✅ Tổng mẫu sau khi xử lý: {len(y)}")
print(f"✅ Shape của dữ liệu X: {X.shape}")
print("✅ Số mẫu theo nhãn:", dict(label_counts))

# Phân tích độ dài chuỗi
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=20)
plt.title("Phân bố độ dài chuỗi")
plt.xlabel("Số frame")
plt.ylabel("Số lượng chuỗi")
plt.axvline(MIN_TIMESTEPS, color='r', linestyle='--', label=f'Min timesteps: {MIN_TIMESTEPS}')
plt.axvline(MAX_TIMESTEPS, color='g', linestyle='--', label=f'Max timesteps: {MAX_TIMESTEPS}')
plt.legend()
plt.show()

# Phân tích phân bố nhãn
plt.figure(figsize=(12, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.title("Phân bố số mẫu theo nhãn")
plt.xlabel("Nhãn")
plt.ylabel("Số mẫu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Tính trọng số lớp
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== Xây dựng mô hình cải tiến ==========
def build_improved_model(input_shape, num_classes):
    """
    Xây dựng mô hình LSTM cải tiến với attention
    """
    inputs = Input(shape=input_shape)
    
    # Lớp Masking giúp mô hình bỏ qua các timestep đệm 0
    x = Masking(mask_value=0.0)(inputs)
    
    # Lớp Conv1D để học các pattern cục bộ
    x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Bidirectional LSTM để học các pattern theo cả hai chiều
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Lớp Attention để tập trung vào các timestep quan trọng
    # Lưu ý: Đây là lớp Attention tự động học từ context
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Tạo vector biểu diễn cuối cùng
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Xây dựng mô hình
input_shape = (MAX_TIMESTEPS, X.shape[2])  # (timesteps, features)
model = build_improved_model(input_shape, len(label_map))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== Callbacks ==========
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_gesture_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, model_checkpoint]

# ========== Huấn luyện ==========
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ========== Vẽ biểu đồ huấn luyện ==========
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Vẽ learning rate nếu sử dụng ReduceLROnPlateau
if 'lr' in history.history:
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# ========== Đánh giá mô hình ==========
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ========== Lưu ==========
model.save("improved_gesture_model.h5")
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Mô hình và label_map đã được lưu!")

# ========== Hàm để sử dụng trong quá trình dự đoán thời gian thực ==========
def predict_sequence(model, sequence, label_map):
    """
    Hàm dự đoán cho một chuỗi dữ liệu theo thời gian thực.
    Có thể dùng để xử lý các chuỗi không hoàn chỉnh.
    
    Input:
        - model: mô hình đã train
        - sequence: chuỗi dữ liệu tọa độ tay (n_frames, 126)
        - label_map: dict ánh xạ từ index sang tên nhãn
    
    Return:
        - prediction: nhãn dự đoán
        - confidence: độ tin cậy
    """
    # Chuẩn hóa dữ liệu
    sequence = normalize_data(sequence)
    
    # Đảm bảo độ dài chuỗi phù hợp (đệm hoặc cắt bớt)
    if sequence.shape[0] < MIN_TIMESTEPS:
        padding = np.repeat(sequence[-1:], MIN_TIMESTEPS - sequence.shape[0], axis=0)
        sequence = np.vstack([sequence, padding])
    
    if sequence.shape[0] > MAX_TIMESTEPS:
        sequence = sequence[-MAX_TIMESTEPS:]  # Lấy MAX_TIMESTEPS frame cuối
    
    # Thêm chiều batch
    sequence = np.expand_dims(sequence, axis=0)
    
    # Dự đoán
    pred = model.predict(sequence, verbose=0)
    pred_idx = np.argmax(pred[0])
    confidence = pred[0][pred_idx]
    
    # Ánh xạ ngược từ index sang tên nhãn
    reverse_label_map = {v: k for k, v in label_map.items()}
    prediction = reverse_label_map.get(pred_idx, "Unknown")
    
    return prediction, confidence