import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn dữ liệu
DATA_DIR = r"D:\System\Videos\VideoProc_Converter_AI\make_data"

labels = []
num_frames = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith('.csv'):
            path = os.path.join(label_dir, file)
            try:
                df = pd.read_csv(path, header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                labels.append(label)
                num_frames.append(len(df))
            except Exception as e:
                print(f"Lỗi đọc {file}: {e}")

# 📌 1. Biểu đồ số mẫu mỗi nhãn
plt.figure(figsize=(10, 5))
sns.countplot(x=labels)
plt.title("Số mẫu mỗi nhãn")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📌 2. Biểu đồ độ dài sample (số frame mỗi file)
plt.figure(figsize=(10, 5))
sns.histplot(num_frames, bins=20, kde=True)
plt.title("Phân bố số frame mỗi file")
plt.xlabel("Số frame")
plt.ylabel("Số lượng file")
plt.tight_layout()
plt.show()
