import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Cấu hình
DATA_DIR = r"D:\System\Videos\VideoProc_Converter_AI\make_data"
TIMESTEPS = 10  # Số frame liên tục

X, y = [], []

# Đọc dữ liệu từ tất cả các thư mục nhãn
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(label_dir, file), header=None)
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna()

                if len(df) < TIMESTEPS:
                    print(f"Bỏ qua file {file} do quá ít dòng hợp lệ")
                    continue

                values = df.values.astype(np.float32)

                for i in range(TIMESTEPS, len(values)):
                    window = values[i-TIMESTEPS:i]
                    if window.shape == (TIMESTEPS, 126):
                        X.append(window)
                        y.append(label)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý file {file}: {e}")


X = np.array(X, dtype=np.float32)

y = np.array(y)

print(f"Tổng số mẫu: {len(X)} - Dạng dữ liệu: {X.shape}")


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save model và nhãn
model.save("model.h5")
np.save("labels.npy", label_encoder.classes_)

print("✅ Đã lưu model và nhãn:", label_encoder.classes_)
