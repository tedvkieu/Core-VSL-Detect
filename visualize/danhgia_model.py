import os
import pandas as pd

# Thay đổi đường dẫn nếu cần
DATASET_DIR = r"D:\System\Videos\VideoProc_Converter_AI\data_maked"

label_frame_dims = {}

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    feature_dims = set()

    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            file_path = os.path.join(label_path, file)
            try:
                df = pd.read_csv(file_path, header=None)
                feature_dims.add(df.shape[1])
            except Exception as e:
                print(f"[ERROR] File {file_path} bị lỗi: {e}")

    label_frame_dims[label] = feature_dims

for label, dims in label_frame_dims.items():
    print(f"{label}: {dims}")
