import asyncio
import websockets
import json
import os
import csv
import time
import numpy as np
from collections import deque
from services.continuous_predict import predict_sequence_from_frames

connected_clients = set()

# File log
CSV_FILE = "frames_log.csv"

# Nếu file chưa tồn tại, ghi header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = [i for i in range(126)]
        writer.writerow(header)

class ContinuousPredictionManager:
    def __init__(self):
        self.client_buffers = {}  # Lưu buffer cho mỗi client
        self.client_last_predictions = {}  # Cache kết quả prediction cuối
        self.client_prediction_times = {}  # Track thời gian prediction
        self.prediction_queue = asyncio.Queue()  # Queue xử lý batch prediction
        self.batch_processor_task = None
        
        # Cấu hình tối ưu
        self.MIN_PREDICTION_INTERVAL = 0.3  # 300ms minimum giữa các prediction
        self.BATCH_SIZE = 4  # Xử lý 4 client cùng lúc
        self.CONFIDENCE_THRESHOLD = 0.9
        self.DUPLICATE_THRESHOLD = 0.95  # Threshold để coi là duplicate
        
    async def start_batch_processor(self):
        """Khởi động batch processor để xử lý nhiều prediction cùng lúc"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            
    async def stop_batch_processor(self):
        """Dừng batch processor"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
            self.batch_processor_task = None
    
    async def _batch_processor(self):
        """Xử lý batch prediction để tối ưu GPU/CPU"""
        pending_predictions = []
        
        while True:
            try:
                # Collect batch
                while len(pending_predictions) < self.BATCH_SIZE:
                    try:
                        item = await asyncio.wait_for(self.prediction_queue.get(), timeout=0.1)
                        pending_predictions.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if pending_predictions:
                    # Process batch
                    await self._process_prediction_batch(pending_predictions)
                    pending_predictions.clear()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Lỗi trong batch processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_prediction_batch(self, batch):
        """Xử lý batch predictions"""
        try:
            # Tách data và metadata
            frames_batch = [item['frames'] for item in batch]
            
            # Batch prediction (nếu model hỗ trợ)
            results = await self._batch_predict(frames_batch)
            
            # Gửi kết quả về cho từng client
            for i, item in enumerate(batch):
                websocket = item['websocket']
                client_id = item['client_id']
                result = results[i] if i < len(results) else None
                
                try:
                    await self._send_result(websocket, client_id, result)
                except Exception as e:
                    print(f"Gửi kết quả cho client {client_id} thất bại: {e}")


        except Exception as e:
            print(f"❌ Lỗi batch prediction: {e}")
    
    async def _batch_predict(self, frames_batch):
        """Predict cho batch frames"""
        results = []
        
        # Xử lý tuần tự (có thể tối ưu thành parallel nếu model hỗ trợ)
        for frames in frames_batch:
            try:
                result = await asyncio.wait_for(
                    predict_sequence_from_frames(frames), 
                    timeout=3
                )
                results.append(result)
            except asyncio.TimeoutError:
                results.append({
                    "label": "timeout",
                    "confidence": 0.0,
                    "message": "Prediction timeout"
                })
            except Exception as e:
                results.append({
                    "label": "error",
                    "confidence": 0.0,
                    "message": f"Prediction error: {str(e)}"
                })
        
        return results
    
    async def handle_continuous_prediction(self, websocket, client_id, frames, timestamp):
        """Xử lý prediction liên tục với tối ưu"""
        current_time = time.time()
        
        # Throttle prediction per client
        last_time = self.client_prediction_times.get(client_id, 0)
        if current_time - last_time < self.MIN_PREDICTION_INTERVAL:
            return  # Skip nếu quá gần prediction trước
        
        # Update buffer cho client
        self.client_buffers[client_id] = frames
        self.client_prediction_times[client_id] = current_time
        
        # Thêm vào queue để xử lý batch
        await self.prediction_queue.put({
            'websocket': websocket,
            'client_id': client_id,
            'frames': frames,
            'timestamp': timestamp
        })
    
    async def _send_result(self, websocket, client_id, result):
        """Gửi kết quả với duplicate detection"""
        try:
            label = result["label"]
            confidence = result["confidence"]
            
            # Kiểm tra duplicate
            last_prediction = self.client_last_predictions.get(client_id)
            if (last_prediction and 
                last_prediction["label"] == label and 
                abs(last_prediction["confidence"] - confidence) < 0.05):
                return  # Skip duplicate
            
            # Cache prediction mới
            self.client_last_predictions[client_id] = {
                "label": label,
                "confidence": confidence,
                "timestamp": time.time()
            }
            
            # Gửi kết quả
            if confidence > self.CONFIDENCE_THRESHOLD:
                await websocket.send(json.dumps({
                    "status": "success",
                    "label": label,
                    "confidence": confidence,
                    "type": "continuous_prediction",
                    "timestamp": time.time()
                }))
                print(f"✅ Client {client_id}: {label} ({confidence:.2f})")
            else:
                await websocket.send(json.dumps({
                    "status": "low_confidence",
                    "label": label,
                    "confidence": confidence,
                    "type": "continuous_prediction"
                }))
                
        except Exception as e:
            print(f"❌ Lỗi gửi kết quả: {e}")
    
    def cleanup_client(self, client_id):
        """Dọn dẹp data của client khi disconnect"""
        self.client_buffers.pop(client_id, None)
        self.client_last_predictions.pop(client_id, None)
        self.client_prediction_times.pop(client_id, None)

# Global manager
prediction_manager = ContinuousPredictionManager()

async def handler(websocket):
    client_id = id(websocket)  # Unique ID cho mỗi client
    connected_clients.add(websocket)
    print(f"🟢 Client {client_id} đã kết nối")
    
    # Start batch processor nếu chưa chạy
    await prediction_manager.start_batch_processor()

    try:
        async for message in websocket:
            data = json.loads(message)
            frames = data.get("frames")
            msg_type = data.get("type", "prediction")
            timestamp = data.get("timestamp", time.time())

            if frames and isinstance(frames, list) and all(len(f) == 126 for f in frames):
                
                if msg_type == "continuous_prediction":
                    # Xử lý prediction liên tục
                    await prediction_manager.handle_continuous_prediction(
                        websocket, client_id, frames, timestamp
                    )
                else:
                    # Xử lý prediction thông thường (legacy)
                    try:
                        result = await asyncio.wait_for(
                            predict_sequence_from_frames(frames), 
                            timeout=5
                        )
                        
                        label = result["label"]
                        confidence = float(result["confidence"])

                        if confidence > 0.9:
                            print(f"✅ Dự đoán: {label} ({confidence:.2f})")
                            await websocket.send(json.dumps({
                                "status": "success",
                                "label": label,
                                "confidence": confidence
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "label": label,
                                "confidence": confidence
                            }))
                            print(f"⚠️ Không chắc chắn {label} ({confidence:.2f})")
                            
                    except asyncio.TimeoutError:
                        print("❌ Predict quá lâu, bỏ qua.")
                        await websocket.send(json.dumps({
                            "status": "timeout",
                            "label": "timeout",
                            "confidence": 0.0
                        }))
            else:
                print("❌ Dữ liệu không hợp lệ. Mỗi frame phải có 126 phần tử.")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔴 Client {client_id} đã ngắt kết nối")
    except Exception as e:
        print(f"❌ Lỗi xử lý client {client_id}: {e}")
    finally:
        connected_clients.remove(websocket)
        prediction_manager.cleanup_client(client_id)
        
        # Stop batch processor nếu không còn client
        if not connected_clients:
            await prediction_manager.stop_batch_processor()

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("🚀 WebSocket server đang chạy tại ws://localhost:8765")
        print("📊 Batch processing enabled cho continuous prediction")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())