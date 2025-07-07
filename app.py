from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import mediapipe as mp
import base64
from datetime import datetime
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

# Load model và label giống như test.py
model = load_model("models/model_lstm_20_6.h5")
labels = np.load("models/labels_lstm_20_6.npy")

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_both_hands_landmarks(results):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if handedness.classification[0].label == 'Left':
                left_hand = coords
            else:
                right_hand = coords
    return left_hand + right_hand

def draw_hands_on_frame(frame, results):
    """Vẽ landmarks của tay lên frame"""
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            color = (0, 255, 0) if handedness.classification[0].label == 'Left' else (255, 0, 0)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=color, thickness=2),
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1))
    return frame

def save_to_csv(sequence, prediction):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hand_data_{timestamp}.csv"
    
    df = pd.DataFrame(sequence)
    df['prediction'] = prediction
    df.to_csv(filename, index=False)
    print(f"Đã lưu dữ liệu vào file: {filename}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        frame_data = data['frame']
        sequence = data.get('sequence', [])
        collecting = data.get('collecting', False)
        hand_missing_counter = data.get('hand_missing_counter', 0)
        missing_threshold = 10
        
        # Decode base64 image
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip frame horizontally like in test.py
        frame = cv2.flip(frame, 1)
        
        # Process frame
        results = hands.process(frame)
        keypoints = extract_both_hands_landmarks(results)
        
        prediction_result = "no_hands"
        confidence = 0.0
        
        # Logic giống với test.py
        if any(k != 0 for k in keypoints):
            # Phát hiện tay -> bắt đầu thu frame
            sequence.append(keypoints)
            collecting = True
            hand_missing_counter = 0
            prediction_result = "collecting"
        else:
            if collecting:
                hand_missing_counter += 1
                if hand_missing_counter >= missing_threshold:
                    # Tay biến mất -> thực hiện dự đoán ngay
                    if len(sequence) > 0:
                        # Logic dự đoán giống test.py
                        input_data = np.expand_dims(sequence, axis=0)
                        prediction = model.predict(input_data, verbose=0)[0]
                        max_index = np.argmax(prediction)
                        max_label = labels[max_index]
                        confidence = prediction[max_index]

                        print("Dự đoán:", max_label, "(", confidence, ")")
                        if confidence > 0.9 and max_label != "non-action":
                            prediction_result = max_label
                            # Lưu dữ liệu vào CSV khi có dự đoán hợp lệ
                            save_to_csv(sequence, max_label)
                        else:
                            prediction_result = "unknown"
                    else:
                        prediction_result = "no_hands"

                    # Reset trạng thái
                    sequence = []
                    collecting = False
                    hand_missing_counter = 0
                else:
                    prediction_result = "collecting"
        
        # Draw hands on frame
        frame_with_hands = draw_hands_on_frame(frame, results)
        
        # Convert frame back to base64 for display
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'confidence': float(confidence),
            'sequence': sequence,
            'collecting': collecting,
            'hand_missing_counter': hand_missing_counter,
            'frame_with_hands': f"data:image/jpeg;base64,{frame_base64}"
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/save_sequence', methods=['POST'])
def save_sequence():
    try:
        data = request.get_json()
        sequence = data['sequence']
        prediction = data['prediction']
        
        save_to_csv(sequence, prediction)
        
        return jsonify({
            'success': True,
            'message': 'Sequence saved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("App is starting...")
    app.run(debug=True, host='0.0.0.0', port=5000) 