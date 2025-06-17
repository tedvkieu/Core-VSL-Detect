# import asyncio
# import numpy as np
# from keras.models import load_model

# # Load model và label chỉ 1 lần
# model = load_model("models/model_Conv1D_7_6_2.h5")
# labels = np.load("models/model_Conv1D_7_6_2.npy")


# async def predict_sequence_from_frames(frames):
#     if not frames or not isinstance(frames, list) or not all(len(f) == 126 for f in frames):
#         return {
#             "label": "unknown",
#             "confidence": 0.0,
#             "message": "Dữ liệu không hợp lệ"
#         }

#     # ✅ Chuẩn hóa về đúng 90 frame
#     frames = pad_or_trim_frames(frames, target_len=90, dim=126)

#     sequence = np.array(frames, dtype=np.float32)
#     input_data = np.expand_dims(sequence, axis=0)  # shape: (1, 90, 126)

#     prediction = model.predict(input_data, verbose=0)[0]
#     max_index = np.argmax(prediction)
#     confidence = float(prediction[max_index])
#     max_label = labels[max_index]

#     if confidence >= 0.9:
#         return {
#             "label": max_label,
#             "confidence": confidence,
#             "message": max_label
#         }
#     else:
#         return {
#             "label": "unknown",
#             "confidence": confidence,
#             "message": "Chưa hiểu hành động"
#         }




# def pad_or_trim_frames(frames, target_len=90, dim=126):
#     """
#     Đưa frames về đúng độ dài target_len:
#     - Nếu thiếu thì thêm các frame toàn 0 ở cuối
#     - Nếu dư thì cắt bớt frame ở đầu và cuối để cân bằng
#     """
#     current_len = len(frames)

#     if current_len < target_len:
#         # Thiếu → bổ sung frame toàn 0 ở cuối
#         pad_length = target_len - current_len
#         zero_frame = [0.0] * dim
#         frames += [zero_frame] * pad_length

#     elif current_len > target_len:
#         # Dư → cắt đều từ đầu và cuối
#         excess = current_len - target_len
#         cut_start = excess // 2
#         cut_end = excess - cut_start
#         frames = frames[cut_start: current_len - cut_end]

#     print("da cat frame ")

#     return frames


import asyncio
import numpy as np
from keras.models import load_model

# Load model và labels một lần duy nhất
model = load_model("models/model_conv1d_26_4.h5")
labels = np.load("models/labels_conv1d_26_4.npy")

SEQUENCE_LENGTH = 90  # Độ dài chuỗi chuẩn theo model

async def predict_sequence_from_frames(frames):
    """
    Hàm dự đoán hành động từ chuỗi các khung (frames) đầu vào.
    Input: List các frame (mỗi frame gồm 126 tọa độ pose vector)
    Output: Kết quả label và độ tin cậy (confidence)
    """
    if not frames or not isinstance(frames, list):
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": "Dữ liệu không hợp lệ"
        }

    # Lọc những frame sai định dạng
    frames = [f for f in frames if isinstance(f, (list, np.ndarray)) and len(f) == 126]

    if len(frames) < 10:  # quá ít frame để dự đoán
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": "Không đủ dữ liệu để dự đoán"
        }

    # Chuẩn hóa độ dài về đúng 90 frames (hoặc SEQUENCE_LENGTH)
    if len(frames) < SEQUENCE_LENGTH:
        # Padding: lặp lại frame cuối cùng
        pad_len = SEQUENCE_LENGTH - len(frames)
        frames += [frames[-1]] * pad_len
    elif len(frames) > SEQUENCE_LENGTH:
        # Cắt bớt ở cuối
        frames = frames[:SEQUENCE_LENGTH]

    # Định dạng input cho model
    sequence = np.array(frames)  # (90, 126)
    input_data = np.expand_dims(sequence, axis=0)  # (1, 90, 126)

    # Dự đoán
    prediction = model.predict(input_data, verbose=0)[0]
    max_index = np.argmax(prediction)
    confidence = prediction[max_index]
    max_label = labels[max_index]

    # Trả kết quả
    if confidence < 0.9:
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
