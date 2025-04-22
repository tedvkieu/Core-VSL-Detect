import numpy as np
from keras.models import load_model

# Load model và label chỉ 1 lần
model = load_model("model_22_4.h5")
labels = np.load("labels_22_4.npy")

def predict_sequence_from_frames(frames):
   
    sequence = np.array(frames)
    input_data = np.expand_dims(sequence, axis=0)  # (1, timesteps, 126)
    prediction = model.predict(input_data, verbose=0)[0]

    max_index = np.argmax(prediction)
    max_label = labels[max_index]
    confidence = prediction[max_index]
    return max_label, confidence
