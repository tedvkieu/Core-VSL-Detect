import asyncio
import websockets
import json
from predict import predict_sequence_from_frames

connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    print("🟢 Client đã kết nối")

    try:
        async for message in websocket:
            data = json.loads(message)
            frames = data.get("frames")

            if frames and isinstance(frames, list) and all(len(f) == 126 for f in frames):

                label, confidence = predict_sequence_from_frames(frames)

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
                    print(f"⚠️ Không chắc chắn hoặc 'non-action' ({confidence:.2f})")
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
