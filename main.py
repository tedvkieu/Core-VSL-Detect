import cv2
import mediapipe as mp
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Đường dẫn tới folder chứa video
video_folder = r'D:\System\Videos\VideoProc_Converter_AI\vsl'

# Tạo thư mục lưu frame nếu chưa có
output_folder = r'D:\System\Videos\VideoProc_Converter_AI\vsl-label'
#Kiểm tra nếu thư mục output chưa tồn tại, tạo mới
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Lặp qua từng thư mục nhãn trong bộ dữ liệu
# for label_folder in os.listdir(video_folder):
#     label_path = os.path.join(video_folder, label_folder)
    
#     if os.path.isdir(label_path):  # Kiểm tra nếu là thư mục (nhãn)
#         # Tạo thư mục cho nhãn trong thư mục output nếu chưa có
#         label_output_folder = os.path.join(output_folder, label_folder)
#         if not os.path.exists(label_output_folder):
#             os.makedirs(label_output_folder)

#         # Lặp qua từng video trong thư mục nhãn
#         for video_file in os.listdir(label_path):
#             video_path = os.path.join(label_path, video_file)
#             cap = cv2.VideoCapture(video_path)
            
#             # Kiểm tra nếu video mở thành công
#             if not cap.isOpened():
#                 print(f"Error opening video file: {video_path}")
#                 continue

#             frame_count = 0
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 # Lưu frame dưới dạng ảnh vào thư mục nhãn trong folder_OP
#                 frame_filename = f"{video_file}_frame_{frame_count}.jpg"
#                 cv2.imwrite(os.path.join(label_output_folder, frame_filename), frame)
#                 frame_count += 1

#             cap.release()

# cv2.destroyAllWindows()


def extract_hand_landmarks(video_path, label, output_dir):
    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Đọc video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    landmarks_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi ảnh sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                frame_landmarks = []
                for lm in landmarks.landmark:
                    frame_landmarks.append((lm.x, lm.y, lm.z))  # Tọa độ 3D của từng landmark
                landmarks_data.append(frame_landmarks)

        frame_count += 1

    cap.release()

    # Lưu dữ liệu vào file .txt
    txt_file_path = os.path.join(output_dir, f"{label}.txt")
    with open(txt_file_path, 'w') as f:
        for frame_landmarks in landmarks_data:
            for lm in frame_landmarks:
                f.write(f"{lm[0]} {lm[1]} {lm[2]}\n")  # Ghi tọa độ x, y, z vào file
            f.write("\n")

def process_videos(dataset_dir, output_dir):
    # Duyệt qua từng folder trong dataset
    for label_folder in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_folder)
        if os.path.isdir(label_path):
            # Duyệt qua các video trong folder
            for video in os.listdir(label_path):
                if video.endswith(('.mp4', '.avi')):  # Kiểm tra định dạng video
                    video_path = os.path.join(label_path, video)
                    extract_hand_landmarks(video_path, label_folder, output_dir)

# Thư mục chứa dataset của bạn
dataset_dir =  r'D:\System\Videos\VideoProc_Converter_AI\vsl'
# Thư mục nơi sẽ lưu các file .txt
output_dir = r'D:\System\Videos\VideoProc_Converter_AI\extract_features'

process_videos(dataset_dir, output_dir)
