import asyncio
import numpy as np
from keras.models import load_model
import tensorflow as tf
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor

# Tối ưu TensorFlow
tf.config.experimental.enable_op_determinism()

# Load model và label chỉ 1 lần với optimizations
class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            print("🔄 Loading model...")
            
            # Load model với optimization
            self.model = load_model(
                r"D:\Project\Intern-Project\VSL-detect\models\model_lstm_5_7.keras",
                compile=False  # Không compile lại để tăng tốc
            )
            
            # Tối ưu model cho inference
            self.model.compile(optimizer='adam')  # Minimal compile
            
            self.labels = np.load(r"D:\Project\Intern-Project\VSL-detect\models\labels_lstm_5_7.npy")
            
            # Thread pool cho CPU intensive tasks
            self.executor = ThreadPoolExecutor(max_workers=2)
            
            # Warm up model
            self._warmup_model()
            
            self._initialized = True
            print("✅ Model loaded và optimized")
    
    def _warmup_model(self):
        """Warm up model để tránh cold start"""
        try:
            dummy_input = np.random.random((1, 30, 126)).astype(np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            print("🔥 Model warmed up")
        except Exception as e:
            print(f"⚠️ Warmup warning: {e}")
    
    @lru_cache(maxsize=128)
    def _cached_preprocess(self, frames_hash):
        """Cache preprocessing results"""
        # This would need actual frames data, just placeholder
        pass
    
    def predict_batch(self, frames_batch):
        """Predict cho batch frames - tối ưu cho multiple clients"""
        try:
            # Convert to numpy array
            batch_array = np.array(frames_batch, dtype=np.float32)
            
            # Batch prediction - nhanh hơn nhiều prediction đơn lẻ
            predictions = self.model.predict(batch_array, verbose=0)
            
            results = []
            for i, prediction in enumerate(predictions):
                max_index = np.argmax(prediction)
                max_label = self.labels[max_index]
                confidence = float(prediction[max_index])
                
                if confidence < 0.9:
                    result = {
                        "label": "unknown",
                        "confidence": confidence,
                        "message": "Chưa hiểu hành động"
                    }
                else:
                    result = {
                        "label": max_label,
                        "confidence": confidence,
                        "message": max_label
                    }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return [{"label": "error", "confidence": 0.0, "message": str(e)} 
                   for _ in frames_batch]

# Global model manager
model_manager = ModelManager()

async def predict_sequence_from_frames(frames):
    """Original function - tối ưu cho single prediction"""
    loop = asyncio.get_event_loop()
    
    # Validate input
    if not (frames and isinstance(frames, list) and 
            all(len(f) == 126 for f in frames) and len(frames) <= 100):
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": "Dữ liệu không hợp lệ"
        }
    
    try:
        # Run prediction in thread pool để không block event loop
        result = await loop.run_in_executor(
            model_manager.executor,
            _sync_predict_single,
            frames
        )
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            "label": "error",
            "confidence": 0.0,
            "message": f"Prediction error: {str(e)}"
        }

def _sync_predict_single(frames):
    """Synchronous prediction function"""
    sequence = np.array(frames, dtype=np.float32)
    input_data = np.expand_dims(sequence, axis=0)  # (1, timesteps, 126)
    
    # Predict
    prediction = model_manager.model.predict(input_data, verbose=0)[0]
    
    # Process result
    max_index = np.argmax(prediction)
    max_label = model_manager.labels[max_index]
    confidence = float(prediction[max_index])
    
    if confidence < 0.9:
        return {
            "label": "unknown",
            "confidence": confidence,
            "message": "Chưa hiểu hành động"
        }
    else:
        return {
            "label": max_label,
            "confidence": confidence,
            "message": max_label
        }

async def predict_batch_frames(frames_batch):
    """Batch prediction function - tối ưu cho continuous prediction"""
    loop = asyncio.get_event_loop()
    
    try:
        # Validate batch
        valid_batch = []
        for frames in frames_batch:
            if (frames and isinstance(frames, list) and 
                all(len(f) == 126 for f in frames) and len(frames) <= 100):
                valid_batch.append(frames)
            else:
                valid_batch.append(None)  # Placeholder for invalid data
        
        # Run batch prediction in thread pool
        results = await loop.run_in_executor(
            model_manager.executor,
            _sync_predict_batch,
            valid_batch
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return [{"label": "error", "confidence": 0.0, "message": str(e)} 
               for _ in frames_batch]

def _sync_predict_batch(frames_batch):
    """Synchronous batch prediction"""
    results = []
    valid_frames = []
    valid_indices = []
    
    # Separate valid frames
    for i, frames in enumerate(frames_batch):
        if frames is not None:
            valid_frames.append(frames)
            valid_indices.append(i)
        else:
            results.append({
                "label": "unknown",
                "confidence": 0.0,
                "message": "Dữ liệu không hợp lệ"
            })
    
    # Batch predict valid frames
    if valid_frames:
        batch_results = model_manager.predict_batch(valid_frames)
        
        # Insert results back to correct positions
        for i, result in enumerate(batch_results):
            results.insert(valid_indices[i], result)
    
    return results

# Additional utility functions
def preprocess_batch(frames_batch):
    """Preprocess batch of frames for optimization"""
    processed = []
    for frames in frames_batch:
        if frames and len(frames) > 0:
            # Normalize frames
            normalized = np.array(frames, dtype=np.float32)
            # Add any preprocessing logic here
            processed.append(normalized.tolist())
        else:
            processed.append([])
    return processed

@lru_cache(maxsize=256)
def get_label_confidence_cached(prediction_tuple):
    """Cache frequent predictions để tăng tốc"""
    prediction = np.array(prediction_tuple)
    max_index = np.argmax(prediction)
    max_label = model_manager.labels[max_index]
    confidence = float(prediction[max_index])
    return max_label, confidence