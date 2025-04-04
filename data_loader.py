import os
import numpy as np

def load_data_from_folder(folder):

    data = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
           
                with open(file_path, 'r') as f:
                    landmarks = []
                    for line in f:
                        coordinates = list(map(float, line.strip().split(', ')))
                        landmarks.append(coordinates)
                    data.append(np.array(landmarks))
                    labels.append(label_folder)
    
    return np.array(data), np.array(labels)

def preprocess_data(data, labels):

    data = data / np.max(np.abs(data), axis=0)
    return data, labels
