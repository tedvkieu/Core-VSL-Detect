import os
import numpy as np
import pandas as pd

DATASET_DIR = r'D:\System\Videos\VideoProc_Converter_AI\data_maked'
LABEL = "Cảm ơn"

label_path = os.path.join(DATASET_DIR, LABEL)
invalid_files = []

for file in os.listdir(label_path):
    if file.endswith('.txt'):
        file_path = os.path.join(label_path, file)
        try:
            df = pd.read_csv(file_path, header=None)
            data = df.values  # (n_frames, 126)
            left_hand = data[:, :63]
            right_hand = data[:, 63:]
            if np.all(left_hand == 0) or np.all(right_hand == 0):
                invalid_files.append(file_path)
        except Exception as e:
            print(f"[ERROR] Không đọc được file {file_path}: {e}")

print(f"Các file không hợp lệ cho nhãn {LABEL}: {len(invalid_files)}")
for f in invalid_files[:5]:  # In 5 file đầu tiên
    print(f" - {f}")