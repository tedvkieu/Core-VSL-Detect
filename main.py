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

def extract_hand_landmarks(video_path, label, video_name, label_output_dir):
    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Đọc video
    cap = cv2.VideoCapture(video_path)
    landmarks_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                frame_landmarks = []
                for lm in landmarks.landmark:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
                landmarks_data.append(frame_landmarks)
        else:
            # Thêm vector zero nếu không phát hiện tay
            landmarks_data.append([(0, 0, 0)] * 21)

    cap.release()

    # Tạo tên file dựa trên tên video (không cần thêm label vào tên file nữa vì đã có thư mục nhãn)
    txt_file_path = os.path.join(label_output_dir, f"{video_name}.txt")
    with open(txt_file_path, 'w') as f:
        for frame_landmarks in landmarks_data:
            for lm in frame_landmarks:
                f.write(f"{lm[0]} {lm[1]} {lm[2]}\n")
            f.write("\n")  # Phân cách giữa các frame

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
                    extract_hand_landmarks(video_path, label_folder, video_name, label_output_dir)

# Đường dẫn thư mục
dataset_dir = r'D:\System\Videos\VideoProc_Converter_AI\vsl'
output_dir = r'D:\System\Videos\VideoProc_Converter_AI\extract_features_ex'

# Chạy chương trình
process_videos(dataset_dir, output_dir)