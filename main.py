import asyncio
import websockets
import json
import os
import csv
from services.predict import predict_sequence_from_frames

connected_clients = set()


# File log
CSV_FILE = "frames_log.csv"

# Nếu file chưa tồn tại, ghi header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = [i for i in range(126)]
        writer.writerow(header)



async def handler(websocket):
    connected_clients.add(websocket)
    print("🟢 Client đã kết nối")

    try:
        async for message in websocket:
            data = json.loads(message)
            frames = data.get("frames")
            # /save_frames_to_csv(frames, "unknown")

            if frames and isinstance(frames, list) and all(len(f) == 126 for f in frames):

                try:
                    result = await asyncio.wait_for(predict_sequence_from_frames(frames), timeout=5)
                    label = result["label"]
                    confidence = result["confidence"]
                except asyncio.TimeoutError:
                    print("❌ Predict quá lâu, bỏ qua.")



                confidence = float(confidence)


                if confidence > 0.9:
                    print(f"✅ Dự đoán: {label} ({confidence:.2f})")

                    # Gửi kết quả về cho client
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
                    print(f"⚠️ Không chắc chắn {label} hoặc 'non-action' ({confidence:.2f})")
            else:
                print("❌ Dữ liệu không hợp lệ. Mỗi frame phải có 126 phần tử.")
    except websockets.exceptions.ConnectionClosed:
        print("🔴 Client đã ngắt kết nối")
    finally:
        connected_clients.remove(websocket)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("🚀 WebSocket server đang chạy tại ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
