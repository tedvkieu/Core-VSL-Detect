import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# ⚙️ Cấu hình
DATA_DIR = r"D:\System\Videos\VideoProc_Converter_AI\make_data"
MAX_LEN = 300  # độ dài tối đa 1 chuỗi, pad nếu ngắn hơn
FEATURE_DIM = 126

# 📥 Đọc dữ liệu
X, y = [], []
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir): continue

    for file in os.listdir(label_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(label_dir, file), header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                data = df.values.astype(np.float32)

                # skip nếu file quá ngắn
                if data.shape[0] < 10: continue

                # pad hoặc cắt cho đủ độ dài MAX_LEN
                if len(data) < MAX_LEN:
                    pad_len = MAX_LEN - len(data)
                    data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    data = data[:MAX_LEN]

                X.append(data)
                y.append(label)
            except Exception as e:
                print(f"❌ Lỗi file {file}: {e}")

X = np.array(X)  # shape (samples, MAX_LEN, 126)
y = np.array(y)
print("✅ Dữ liệu đã đọc:", X.shape)

# 🏷️ Encode nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 🔀 Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 🎯 Positional Encoding layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# 🧠 Mô hình Transformer đơn giản
def build_model():
    inputs = tf.keras.Input(shape=(MAX_LEN, FEATURE_DIM))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Dense(128)(x)
    x = PositionalEncoding(MAX_LEN, 128)(x)

    # Multi-head Self-Attention + FFN
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# 🏋️ Huấn luyện
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 💾 Lưu lại
model.save("transformer_model.h5")
np.save("labels.npy", label_encoder.classes_)
print("✅ Đã lưu mô hình và nhãn:", label_encoder.classes_)
