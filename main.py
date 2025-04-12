# Đảm bảo import thư viện cần thiết
import os
import cv2
import mediapipe as mp
def extract_hand_landmarks(video_path, label, video_name, label_output_dir):
    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Đọc video
    cap = cv2.VideoCapture(video_path)
    landmarks_data = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                for idx, lm in enumerate(landmarks.landmark):
                    # Lưu frame_index, landmark_index, x, y, z
                    landmarks_data.append((frame_index, idx, lm.x, lm.y, lm.z))
        else:
            # Thêm vector zero nếu không phát hiện tay
            for idx in range(21):  # MediaPipe hand có 21 điểm
                landmarks_data.append((frame_index, idx, 0, 0, 0))

        frame_index += 1

    cap.release()

    # Tạo tên file dựa trên tên video
    txt_file_path = os.path.join(label_output_dir, f"{video_name}.txt")
    with open(txt_file_path, 'w') as f:
        for data in landmarks_data:
            # Ghi ra dạng: frame_index landmark_index x y z
            f.write(f"{data[0]} {data[1]} {data[2]} {data[3]} {data[4]}\n")

def process_videos(dataset_dir, output_dir):
    # Tạo thư mục đầu ra chính nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Duyệt qua các thư mục nhãn
    for label_folder in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_folder)
        if os.path.isdir(label_path):
            # Tạo thư mục con cho nhãn trong output_dir
            label_output_dir = os.path.join(output_dir, label_folder)
            if not os.path.exists(label_output_dir):
                os.makedirs(label_output_dir)

            # Duyệt qua các video trong thư mục nhãn
            for video in os.listdir(label_path):
                if video.endswith(('.mp4', '.avi')):  # Kiểm tra định dạng video
                    video_path = os.path.join(label_path, video)
                    # Lấy tên video không có phần mở rộng
                    video_name = os.path.splitext(video)[0]
                    print(f"Đang xử lý video: {video} (nhãn: {label_folder})")
                    extract_hand_landmarks(video_path, label_folder, video_name, label_output_dir)
                    print(f"Đã xử lý xong video: {video}")

# Đường dẫn thư mục
dataset_dir = r'D:\System\Videos\VideoProc_Converter_AI\vsl'
output_dir = r'D:\System\Videos\VideoProc_Converter_AI\extract_video'



# Chạy chương trình
process_videos(dataset_dir, output_dir)