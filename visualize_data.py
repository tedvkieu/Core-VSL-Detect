import os
import pandas as pd

# Cấu hình thư mục chứa dữ liệu
DATA_DIR = r"D:\System\Videos\VideoProc_Converter_AI\make_data"
min_frame = 30
max_frame = 150

deleted_files = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(label_dir, file)
            try:
                df = pd.read_csv(file_path, header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                frame_count = len(df)

                if frame_count < min_frame or frame_count > max_frame:
                    os.remove(file_path)
                    deleted_files.append((label, file, frame_count))
                    print(f"🗑️ Đã xoá {file} ({frame_count} frames) trong nhãn {label}")
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {file}: {e}")

print(f"\n✅ Đã xoá {len(deleted_files)} file không đạt yêu cầu.")
