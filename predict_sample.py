import asyncio
import numpy as np
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandGesturePredictor:
    def __init__(self, model_path, scaler_path, encoder_path):
        self.TIMESTEPS = 15  # Phải khớp với cấu hình khi train
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_artifacts(model_path, scaler_path, encoder_path)
        
    def _load_artifacts(self, model_path, scaler_path, encoder_path):
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = load_model(model_path)

            logger.info(f"Loading scaler from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            logger.info(f"Loading label encoder from {encoder_path}")
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            logger.info("✅ Artifacts loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            raise
    
    async def predict_sequence_from_frames(self, frames, confidence_threshold=0.85):
        try:
            if not frames or not isinstance(frames, list):
                return self._error_response("Dữ liệu không hợp lệ - frames phải là list")
            
            if not all(len(f) == 126 for f in frames):
                return self._error_response("Mỗi frame cần có 126 features")

            sequence = np.array(frames, dtype=np.float32)
            if len(sequence) < self.TIMESTEPS:
                return self._error_response(f"Cần ít nhất {self.TIMESTEPS} frames")

            sequence_scaled = self.scaler.transform(sequence)
            input_sequence = sequence_scaled[-self.TIMESTEPS:]
            input_data = np.expand_dims(input_sequence, axis=0)

            prediction = self.model.predict(input_data, verbose=0)[0]
            max_index = np.argmax(prediction)
            max_label = self.label_encoder.inverse_transform([max_index])[0]
            confidence = float(prediction[max_index])

            if confidence < confidence_threshold:
                return {
                    "label": "unknown",
                    "confidence": confidence
                }
            else:
                return {
                    "label": max_label,
                    "confidence": confidence
                }
                
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return self._error_response(f"Lỗi: {str(e)}")
    
    def _error_response(self, message):
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": message
        }


# Khởi tạo predictor toàn cục
try:
    global_predictor = HandGesturePredictor(
        model_path=r"D:\Project\Intern-Project\VSL-detect\models\model_lstm_optimized_6_7.keras",
        scaler_path=r"D:\Project\Intern-Project\VSL-detect\models\scaler_optimized_6_7.pkl", 
        encoder_path=r"D:\Project\Intern-Project\VSL-detect\models\label_encoder_optimized_6_7.pkl"
    )
except Exception as e:
    logger.error(f"❌ Không thể khởi tạo predictor: {e}")
    global_predictor = None


# Hàm public gọi xuống từ main.py
async def predict_sequence_from_frames(frames):
    if global_predictor is None:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": "Model chưa sẵn sàng"
        }
    return await global_predictor.predict_sequence_from_frames(frames)
