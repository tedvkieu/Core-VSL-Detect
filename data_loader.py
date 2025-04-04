import os
import numpy as np

def load_data_from_folder(folder):
    """
    Tải dữ liệu đặc trưng từ thư mục. Mỗi file trong thư mục chứa dữ liệu cho một frame.
    """
    data = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                # Đọc dữ liệu điểm mốc từ file
                with open(file_path, 'r') as f:
                    landmarks = []
                    for line in f:
                        coordinates = list(map(float, line.strip().split(', ')))
                        landmarks.append(coordinates)
                    data.append(np.array(landmarks))
                    labels.append(label_folder)
    
    return np.array(data), np.array(labels)

def preprocess_data(data, labels):
    """
    Tiền xử lý dữ liệu (chuẩn hóa, padding, ...)
    """
    # Tiền xử lý đơn giản (tùy chỉnh thêm nếu cần)
    data = data / np.max(np.abs(data), axis=0)  # Chuẩn hóa dữ liệu
    return data, labels
