import os
import numpy as np
import pandas as pd
import tensorflow as tf  # Thêm import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ========== Cấu hình ==========
DATASET_DIR = r'D:\System\Videos\VideoProc_Converter_AI\data_maked'
TIMESTEPS = 50
BATCH_SIZE = 32
EPOCHS = 100
NUM_HEADS = 4
FF_DIM = 128

# ========== Hàm kiểm tra số tay ==========
def count_hands(data, file_path, label, strict_two_hands_labels=['Cảm ơn']):
    left_hand = data[:, :63]
    right_hand = data[:, 63:]
    left_active = not np.all(left_hand == 0)
    right_active = not np.all(right_hand == 0)
    
    if label in strict_two_hands_labels:
        if not (left_active and right_active):
            print(f"[WARNING] {file_path}: Nhãn {label} yêu cầu 2 tay nhưng chỉ có {'tay trái' if left_active else 'tay phải' if right_active else 'không tay'}")
            return 0
        return 2
    return 2 if left_active and right_active else 1

# ========== Hàm xử lý sequence từ file ==========
def process_file(data, timesteps, file_path, label):
    if data.shape[0] < timesteps:
        print(f"[WARNING] {file_path}: Số frame ({data.shape[0]}) nhỏ hơn TIMESTEPS ({timesteps})")
        return None
    
    # Chuẩn hóa dữ liệu
    data[:, ::3] = np.clip(data[:, ::3], 0, 1)  # x
    data[:, 1::3] = np.clip(data[:, 1::3], 0, 1)  # y
    
    # Cắt hoặc đệm
    if data.shape[0] > timesteps:
        data = data[:timesteps]
    elif data.shape[0] < timesteps:
        padding = np.repeat(data[-1:], timesteps - data.shape[0], axis=0)
        data = np.vstack([data, padding])
    
    return data

# ========== Đọc dữ liệu ==========
X, y = [], []
label_map = {}
label_counts = defaultdict(int)

for idx, label in enumerate(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    label_map[label] = idx

    for file in os.listdir(label_path):
        if file.endswith('.txt'):
            file_path = os.path.join(label_path, file)
            try:
                df = pd.read_csv(file_path, header=None)
                data = df.values
            except Exception as e:
                print(f"[ERROR] Không đọc được file {file_path}: {e}")
                continue

            if data.shape[1] != 126:
                print(f"[WARNING] Bỏ qua file {file_path}: Shape không hợp lệ {data.shape}")
                continue

            num_hands = count_hands(data, file_path, label)
            if num_hands == 0:
                continue

            sequence = process_file(data, TIMESTEPS, file_path, label)
            if sequence is None:
                continue

            X.append(sequence)
            y.append(idx)
            label_counts[label] += 1

X = np.array(X)  # (samples, TIMESTEPS, 126)
y = np.array(y)

print(f"✅ Tổng mẫu: {len(y)}")
print("✅ Số mẫu theo nhãn:", dict(label_counts))

# Kiểm tra phân bố nhãn
plt.figure(figsize=(10, 6))
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

# ========== Xây dựng mô hình CNN + Transformer ==========
def transformer_block(inputs, num_heads, ff_dim, dropout=0.3):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)
    
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    out2 = Add()([out1, ffn_output])
    out2 = LayerNormalization(epsilon=1e-6)(out2)
    return out2

inputs = Input(shape=(TIMESTEPS, 126))
x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = Dropout(0.3)(x)

x = transformer_block(x, num_heads=NUM_HEADS, ff_dim=FF_DIM)
x = transformer_block(x, num_heads=NUM_HEADS, ff_dim=FF_DIM)

x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(label_map), activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== Huấn luyện ==========
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# ========== Vẽ biểu đồ huấn luyện ==========
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ========== Lưu ==========
model.save("gesture_model_transformer.h5")
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Mô hình và label_map đã được lưu!")