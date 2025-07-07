import asyncio
import numpy as np
from keras.models import load_model

# Load model và label chỉ 1 lần
model = load_model(r"D:\Project\Intern-Project\VSL-detect\models\model_lstm_5_7.keras")  # Sửa đuôi .keras
labels = np.load(r"D:\Project\Intern-Project\VSL-detect\models\labels_lstm_5_7.npy")


async def predict_sequence_from_frames(frames):
    # Đảm bảo frames hợp lệ
    if frames and isinstance(frames, list) and all(len(f) == 126 for f in frames) and len(frames) <= 100:
        sequence = np.array(frames, dtype=np.float32)
        input_data = np.expand_dims(sequence, axis=0)  # (1, timesteps, 126)
        
        # Dự đoán từ mô hình
        prediction = model.predict(input_data, verbose=0)[0]
        
        # Xử lý kết quả
        max_index = np.argmax(prediction)
        max_label = labels[max_index]
        confidence = prediction[max_index]
        
        if confidence < 0.85:
            return {
                "label": "unknown",
                "confidence": float(confidence),
                "message": "Chưa hiểu hành động"
            }
        else:
            return {
                "label": max_label,
                "confidence": float(confidence),
                "message": max_label
            }
    else:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": "Dữ liệu không hợp lệ"
        }
