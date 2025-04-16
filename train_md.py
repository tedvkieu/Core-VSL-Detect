import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Masking, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import json
import time
import joblib

# Thiết lập các thông số
class Config:
    DATA_DIR = r"D:\System\Videos\VideoProc_Converter_AI\make_data"  # Thư mục chứa dữ liệu VSL
    MODEL_DIR = r"D:\Project\Intern-Project\VSL-detect\models"     # Thư mục lưu mô hình
    RANDOM_STATE = 42                                                # Giá trị ngẫu nhiên cố định
    TEST_SIZE = 0.2                                                  # Tỷ lệ dữ liệu test
    VALIDATION_SIZE = 0.1                                            # Tỷ lệ dữ liệu validation (từ tập train)
    BATCH_SIZE = 32                                                  # Kích thước batch
    EPOCHS = 50                                                      # Số epoch tối đa
    SEQUENCE_LENGTH = 64                                             # Độ dài tối đa của chuỗi
    LEARNING_RATE = 0.001                                            # Tốc độ học
    USE_BIDIRECTIONAL = True                                         # Sử dụng LSTM/GRU 2 chiều
    MODEL_TYPE = "LSTM"                                              # LSTM hoặc GRU
    UNITS = [128, 64]                                                # Số units cho các lớp LSTM/GRU
    DROPOUT_RATE = 0.3                                               # Tỷ lệ dropout
    USE_ATTENTION = False                                            # Sử dụng attention mechanism

# Hàm tạo thư mục nếu chưa tồn tại
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Hàm tải và tiền xử lý dữ liệu
def load_and_preprocess_data(config):
    print("Bắt đầu tải và tiền xử lý dữ liệu...")
    
    # Tìm tất cả các file CSV trong thư mục dữ liệu
    all_files = []
    all_labels = []
    
    # Duyệt qua tất cả các thư mục con (mỗi thư mục là một nhãn)
    for label_dir in os.listdir(config.DATA_DIR):
        label_path = os.path.join(config.DATA_DIR, label_dir)
        if os.path.isdir(label_path):
            csv_files = glob.glob(os.path.join(label_path, "*.csv"))
            all_files.extend(csv_files)
            all_labels.extend([label_dir] * len(csv_files))
    
    print(f"Tìm thấy {len(all_files)} file CSV từ {len(set(all_labels))} nhãn khác nhau")
    
    # Mã hóa nhãn thành số nguyên
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    
    # Lưu label encoder để sử dụng khi dự đoán
    joblib.dump(label_encoder, os.path.join(config.MODEL_DIR, 'label_encoder.pkl'))
    
    # Chia thành tập train và test theo file (để tránh data leakage)
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, encoded_labels, test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, stratify=encoded_labels
    )
    
    print(f"Chia thành {len(train_files)} file train và {len(test_files)} file test")
    
    # Hàm đọc và xử lý file CSV
    def process_file(file_path, max_length=config.SEQUENCE_LENGTH):
        try:
            df = pd.read_csv(file_path)
            # Nếu file quá dài, lấy max_length frames
            if len(df) > max_length:
                df = df.iloc[:max_length]
            # Nếu file quá ngắn, padding bằng 0
            data = df.values
            if len(data) < max_length:
                padding = np.zeros((max_length - len(data), data.shape[1]))
                data = np.vstack([data, padding])
            return data
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return None
    
    # Tải và xử lý dữ liệu train
    X_train = []
    y_train = []
    for i, file_path in enumerate(train_files):
        data = process_file(file_path)
        if data is not None:
            X_train.append(data)
            y_train.append(train_labels[i])
    
    # Tải và xử lý dữ liệu test
    X_test = []
    y_test = []
    for i, file_path in enumerate(test_files):
        data = process_file(file_path)
        if data is not None:
            X_test.append(data)
            y_test.append(test_labels[i])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Hình dạng dữ liệu train: {X_train.shape}, {y_train.shape}")
    print(f"Hình dạng dữ liệu test: {X_test.shape}, {y_test.shape}")
    
    # Chuẩn hóa dữ liệu
    # Reshape để chuẩn hóa
    original_shape = X_train.shape
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Reshape lại về dạng ban đầu
    X_train = X_train_flat.reshape(original_shape)
    X_test = X_test_flat.reshape(X_test.shape)
    
    # Lưu scaler để sử dụng khi dự đoán
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    
    # Chuyển nhãn sang dạng one-hot encoding
    n_classes = len(label_encoder.classes_)
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    
    # Chia tập validation từ tập train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.VALIDATION_SIZE, 
        random_state=config.RANDOM_STATE, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Chia validation: Train={X_train.shape}, Val={X_val.shape}")
    
    # Lưu thông tin về các lớp
    class_info = {
        'num_classes': n_classes,
        'class_names': list(label_encoder.classes_),
        'class_indices': {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    }
    
    with open(os.path.join(config.MODEL_DIR, 'class_info.json'), 'w') as f:
        json.dump(class_info, f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_info

# Xây dựng mô hình LSTM/GRU
def build_model(input_shape, num_classes, config):
    model = Sequential()
    
    # Thêm lớp Masking để xử lý padding
    model.add(Masking(mask_value=0., input_shape=input_shape))
    
    # Thêm các lớp LSTM/GRU
    for i, units in enumerate(config.UNITS):
        return_sequences = i < len(config.UNITS) - 1  # Chỉ lớp cuối cùng có return_sequences=False
        
        if config.MODEL_TYPE == "LSTM":
            if config.USE_BIDIRECTIONAL:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
        else:  # GRU
            if config.USE_BIDIRECTIONAL:
                model.add(Bidirectional(GRU(units, return_sequences=return_sequences)))
            else:
                model.add(GRU(units, return_sequences=return_sequences))
        
        # Thêm BatchNormalization và Dropout sau mỗi lớp
        model.add(BatchNormalization())
        model.add(Dropout(config.DROPOUT_RATE))
    
    # Lớp đầu ra với softmax
    model.add(Dense(num_classes, activation='softmax'))
    
    # Biên dịch mô hình
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

# Huấn luyện mô hình
def train_model(model, X_train, y_train, X_val, y_val, config):
    print("Bắt đầu huấn luyện mô hình...")
    
    # Tạo thư mục logs
    log_dir = os.path.join(config.MODEL_DIR, "logs", time.strftime("%Y%m%d-%H%M%S"))
    ensure_dir(log_dir)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, "best_model.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu mô hình cuối cùng
    model.save(os.path.join(config.MODEL_DIR, "final_model.h5"))
    
    # Lưu lịch sử huấn luyện
    with open(os.path.join(config.MODEL_DIR, "training_history.json"), 'w') as f:
        json.dump({
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }, f)
    
    return history, model

# Đánh giá mô hình
def evaluate_model(model, X_test, y_test, class_info, config):
    print("Đánh giá mô hình trên tập test...")
    
    # Đánh giá trên tập test
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # In báo cáo phân loại
    class_names = class_info['class_names']
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Vẽ đồ thị độ chính xác và mất mát
    with open(os.path.join(config.MODEL_DIR, "training_history.json"), 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'training_history.png'))
    plt.close()
    
    # Lưu kết quả đánh giá
    evaluation_results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'classification_report': classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
    }
    
    with open(os.path.join(config.MODEL_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f)
    
    return evaluation_results

# Hàm dự đoán với dữ liệu mới
def predict_new_sequence(model_path, scaler_path, label_encoder_path, sequence_data, max_length=64):
    # Tải mô hình
    model = load_model(model_path)
    
    # Tải scaler và label encoder
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Xử lý dữ liệu đầu vào
    if len(sequence_data) > max_length:
        sequence_data = sequence_data[:max_length]
    
    # Padding nếu cần
    if len(sequence_data) < max_length:
        padding = np.zeros((max_length - len(sequence_data), sequence_data.shape[1]))
        sequence_data = np.vstack([sequence_data, padding])
    
    # Chuẩn hóa
    sequence_data = scaler.transform(sequence_data)
    
    # Thêm chiều batch
    sequence_data = np.expand_dims(sequence_data, axis=0)
    
    # Dự đoán
    prediction = model.predict(sequence_data)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = prediction[0][predicted_class_index]
    
    return predicted_class, confidence, prediction[0]

# Hàm chính
def main():
    config = Config()
    
    # Tạo thư mục lưu mô hình
    ensure_dir(config.MODEL_DIR)
    
    # Tải và tiền xử lý dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test, class_info = load_and_preprocess_data(config)
    
    # Lưu cấu hình
    with open(os.path.join(config.MODEL_DIR, 'config.json'), 'w') as f:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        json.dump(config_dict, f)
    
    # Xây dựng mô hình
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes, config)
    
    # Huấn luyện mô hình
    history, model = train_model(model, X_train, y_train, X_val, y_val, config)
    
    # Đánh giá mô hình
    evaluation_results = evaluate_model(model, X_test, y_test, class_info, config)
    
    print("\nQuá trình huấn luyện và đánh giá hoàn tất!")
    print(f"Mô hình và kết quả đã được lưu tại: {config.MODEL_DIR}")
    
    # Tạo ví dụ về cách sử dụng mô hình để dự đoán
    print("\nVí dụ về cách sử dụng mô hình để dự đoán:")
    print("""
    # Sử dụng hàm predict_new_sequence để dự đoán dữ liệu mới
    predicted_class, confidence, probabilities = predict_new_sequence(
        model_path=os.path.join(config.MODEL_DIR, "best_model.h5"),
        scaler_path=os.path.join(config.MODEL_DIR, "scaler.pkl"),
        label_encoder_path=os.path.join(config.MODEL_DIR, "label_encoder.pkl"),
        sequence_data=your_new_data  # Dữ liệu cần dự đoán
    )
    print(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.4f}")
    """)

if __name__ == "__main__":
    main()