import os
import numpy as np
import pandas as pd

DATA_DIR = r"D:\System\Videos\VideoProc_Converter_AI\data_maked"
FIXED_LENGTH = 90

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(label_dir, file)
            try:
                df = pd.read_csv(file_path, header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                
                current_len = len(df)
                if current_len == FIXED_LENGTH:
                    continue  # Không cần chỉnh sửa

                elif current_len > FIXED_LENGTH:
                    # Cắt ở giữa chuỗi
                    start = (current_len - FIXED_LENGTH) // 2
                    df_new = df.iloc[start:start + FIXED_LENGTH]

                else:  # current_len < FIXED_LENGTH
                    # Pad thêm dòng 0 ở cuối
                    pad_len = FIXED_LENGTH - current_len
                    padding = pd.DataFrame(np.zeros((pad_len, df.shape[1])), columns=df.columns)
                    df_new = pd.concat([df, padding], ignore_index=True)

                df_new.to_csv(file_path, index=False, header=False)
                print(f"✅ Đã chuẩn hóa: {file} ({current_len} → {FIXED_LENGTH})")

            except Exception as e:
                print(f"❌ Lỗi xử lý {file}: {e}")
