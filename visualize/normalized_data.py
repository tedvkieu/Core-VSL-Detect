import os
import pandas as pd
import numpy as np

def normalize_hand(landmarks):
    wrist = landmarks[0]
    relative = landmarks - wrist
    ref_point = landmarks[12]
    scale = np.linalg.norm(ref_point - wrist)
    if scale > 0:
        relative /= scale
    return relative

def normalize_frame(row):
    row = np.array(row).reshape(2, 21, 3)
    left_hand = normalize_hand(row[0])
    right_hand = normalize_hand(row[1])
    combined = np.concatenate([left_hand, right_hand], axis=0)
    return combined.flatten()

def normalize_csv_file(input_path, output_path):
    df = pd.read_csv(input_path, header=None, skiprows=1)
    normalized_data = []

    for _, row in df.iterrows():
        try:
            norm_row = normalize_frame(row.values)
            normalized_data.append(norm_row)
        except:
            continue  # bỏ qua dòng lỗi nếu thiếu dữ liệu

    norm_df = pd.DataFrame(normalized_data)
    norm_df.to_csv(output_path, header=False, index=False)

def normalize_all_labels(input_root, output_root):
    for label_folder in os.listdir(input_root):
        label_path = os.path.join(input_root, label_folder)
        if os.path.isdir(label_path):
            output_label_path = os.path.join(output_root, label_folder)
            os.makedirs(output_label_path, exist_ok=True)

            for file_name in os.listdir(label_path):
                if file_name.endswith('.csv'):
                    input_file = os.path.join(label_path, file_name)
                    output_file = os.path.join(output_label_path, file_name)

                    normalize_csv_file(input_file, output_file)
                    print(f"✔ Đã xử lý: {label_folder}/{file_name}")

# Thư mục gốc
input_folder = r"D:\System\Videos\VideoProc_Converter_AI\make_data"
output_folder = r"D:\System\Videos\VideoProc_Converter_AI\optimize_data"

# Thực thi
normalize_all_labels(input_folder, output_folder)
