import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import savgol_filter
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib

def load_data_from_directory(directory_path):
    """
    Đọc tất cả các file CSV trong thư mục và gắn nhãn theo tên thư mục
    
    Parameters:
    directory_path (str): Đường dẫn đến thư mục chứa file CSV
    
    Returns:
    tuple: (X_data, y_data, filenames)
    """
    all_data = []
    all_labels = []
    all_filenames = []
    
    # Danh sách các thư mục chứa dữ liệu (mỗi thư mục là một nhãn)
    label_dirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    print(f"Tìm thấy {len(label_dirs)} lớp: {label_dirs}")
    
    for label_idx, label_name in enumerate(label_dirs):
        label_path = os.path.join(directory_path, label_name)
        csv_files = glob.glob(os.path.join(label_path, "*.csv"))
        
        print(f"Đang đọc {len(csv_files)} file từ lớp '{label_name}'")
        
        for file_path in csv_files:
            try:
                # Đọc file CSV
                df = pd.read_csv(file_path)
                
                # Kiểm tra xem dữ liệu có đúng định dạng không
                if df.shape[1] != 126:  # 126 landmark features
                    print(f"Bỏ qua file {file_path} do số features không đúng: {df.shape[1]}")
                    continue
                
                # Thêm vào danh sách
                all_data.append(df.values)
                all_labels.extend([label_idx] * len(df))
                all_filenames.extend([os.path.basename(file_path)] * len(df))
                
            except Exception as e:
                print(f"Lỗi khi đọc file {file_path}: {e}")
    
    # Ghép tất cả dữ liệu
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    # Tạo từ điển ánh xạ index -> tên nhãn
    label_map = {idx: name for idx, name in enumerate(label_dirs)}
    
    return X, y, all_filenames, label_map

def check_missing_values(X):
    """Kiểm tra giá trị thiếu trong dữ liệu"""
    missing_count = np.isnan(X).sum()
    total_missing = missing_count.sum()
    
    print(f"Tổng số giá trị thiếu: {total_missing}")
    if total_missing > 0:
        missing_features = np.where(missing_count > 0)[0]
        print(f"Các features có giá trị thiếu: {missing_features}")
        print(f"Số lượng giá trị thiếu theo feature: {missing_count[missing_features]}")
    
    return total_missing

def handle_missing_values(X, strategy='mean'):
    """Xử lý giá trị thiếu trong dữ liệu"""
    if strategy == 'mean':
        # Thay thế bằng giá trị trung bình của cột
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    elif strategy == 'zero':
        # Thay thế bằng 0
        X = np.nan_to_num(X)
    elif strategy == 'interpolate':
        # Sử dụng nội suy
        df = pd.DataFrame(X)
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        X = df.values
    
    return X

def normalize_data(X, method='minmax', save_scaler=True, scaler_path=None):
    """
    Chuẩn hóa dữ liệu
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    method (str): Phương pháp chuẩn hóa ('minmax' hoặc 'standard')
    save_scaler (bool): Có lưu scaler không
    scaler_path (str): Đường dẫn lưu scaler
    
    Returns:
    numpy.ndarray: Dữ liệu đã chuẩn hóa
    """
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Phương pháp chuẩn hóa không hợp lệ. Chọn 'minmax' hoặc 'standard'")
    
    # Chuẩn hóa dữ liệu
    X_scaled = scaler.fit_transform(X)
    
    # Lưu scaler nếu cần
    if save_scaler and scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Đã lưu scaler tại: {scaler_path}")
    
    return X_scaled, scaler

def smooth_landmarks(X, window_length=7, polyorder=2):
    """
    Làm mịn dữ liệu landmark bằng bộ lọc Savitzky-Golay
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    window_length (int): Độ dài cửa sổ làm mịn
    polyorder (int): Bậc đa thức
    
    Returns:
    numpy.ndarray: Dữ liệu đã làm mịn
    """
    # Kiểm tra tham số
    if window_length % 2 == 0:
        window_length += 1  # window_length phải là số lẻ
    
    if window_length > X.shape[0]:
        print(f"Cảnh báo: window_length ({window_length}) lớn hơn số lượng mẫu ({X.shape[0]})")
        window_length = min(X.shape[0] - (1 if X.shape[0] % 2 == 0 else 0), 5)
        print(f"Tự động điều chỉnh window_length = {window_length}")
    
    if polyorder >= window_length:
        polyorder = window_length - 1
        print(f"Tự động điều chỉnh polyorder = {polyorder}")
    
    # Áp dụng bộ lọc cho từng feature
    X_smoothed = np.zeros_like(X)
    for i in range(X.shape[1]):
        try:
            X_smoothed[:, i] = savgol_filter(X[:, i], window_length, polyorder)
        except Exception as e:
            print(f"Lỗi khi làm mịn feature {i}: {e}")
            X_smoothed[:, i] = X[:, i]  # Giữ nguyên nếu có lỗi
    
    return X_smoothed

def split_data_by_gesture(X, y, filenames):
    """
    Tách dữ liệu thành các cử chỉ dựa trên filename
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    y (numpy.ndarray): Nhãn
    filenames (list): Tên file tương ứng với mỗi mẫu
    
    Returns:
    dict: Từ điển chứa dữ liệu theo cử chỉ
    """
    gestures = {}
    unique_files = np.unique(filenames)
    
    for file in unique_files:
        indices = np.where(np.array(filenames) == file)[0]
        gestures[file] = {
            'X': X[indices],
            'y': y[indices]
        }
    
    return gestures

def balance_dataset(X, y, random_state=42):
    """
    Cân bằng dữ liệu sử dụng SMOTE
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    y (numpy.ndarray): Nhãn
    random_state (int): Giá trị ngẫu nhiên
    
    Returns:
    tuple: (X_balanced, y_balanced)
    """
    # Đếm số lượng mẫu mỗi lớp
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Phân bố ban đầu: {dict(zip(unique_labels, counts))}")
    
    # Áp dụng SMOTE
    try:
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Kiểm tra lại phân bố
        unique_labels, counts = np.unique(y_balanced, return_counts=True)
        print(f"Phân bố sau khi cân bằng: {dict(zip(unique_labels, counts))}")
        
        return X_balanced, y_balanced
    except Exception as e:
        print(f"Lỗi khi cân bằng dữ liệu: {e}")
        return X, y

def feature_selection_pca(X, n_components=0.95):
    """
    Giảm chiều dữ liệu sử dụng PCA
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    n_components (float/int): Số lượng components hoặc tỷ lệ phương sai cần giữ lại
    
    Returns:
    tuple: (X_pca, pca_model)
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_
    
    print(f"Số lượng components sau PCA: {pca.n_components_}")
    print(f"Tổng phương sai giải thích được: {sum(explained_variance):.4f}")
    
    return X_pca, pca

def visualize_data_distribution(X, y, label_map=None):
    """
    Hiển thị phân bố dữ liệu
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    y (numpy.ndarray): Nhãn
    label_map (dict): Ánh xạ từ chỉ số nhãn sang tên nhãn
    """
    # PCA để giảm xuống 2D để biểu diễn
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.5)
    
    if label_map:
        legend_labels = [label_map[i] for i in np.unique(y)]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    
    plt.title('Phân bố dữ liệu sau khi giảm chiều với PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def save_processed_data(X, y, output_path, metadata=None):
    """
    Lưu dữ liệu đã tiền xử lý
    
    Parameters:
    X (numpy.ndarray): Dữ liệu đầu vào
    y (numpy.ndarray): Nhãn
    output_path (str): Đường dẫn lưu file
    metadata (dict): Thông tin bổ sung
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez(output_path, X=X, y=y)
    print(f"Đã lưu dữ liệu tại: {output_path}")
    
    if metadata:
        metadata_path = output_path.replace('.npz', '_metadata.json')
        pd.DataFrame([metadata]).to_json(metadata_path, orient='records')
        print(f"Đã lưu metadata tại: {metadata_path}")

def preprocess_vsl_data(input_directory, output_directory, normalize_method='minmax', smooth_data=True,
                       balance_classes=True, use_pca=True, pca_components=0.95):
    """
    Thực hiện tiền xử lý dữ liệu VSL
    
    Parameters:
    input_directory (str): Thư mục chứa dữ liệu gốc
    output_directory (str): Thư mục lưu dữ liệu đã xử lý
    normalize_method (str): Phương pháp chuẩn hóa ('minmax' hoặc 'standard')
    smooth_data (bool): Có làm mịn dữ liệu không
    balance_classes (bool): Có cân bằng dữ liệu không
    use_pca (bool): Có sử dụng PCA không
    pca_components (float/int): Số lượng components hoặc tỷ lệ phương sai cần giữ lại
    """
    print(f"===== TIỀN XỬ LÝ DỮ LIỆU VSL =====")
    
    # Bước 1: Đọc dữ liệu
    print(f"\n1. Đọc dữ liệu từ {input_directory}")
    X, y, filenames, label_map = load_data_from_directory(input_directory)
    print(f"Đã đọc {X.shape[0]} mẫu với {X.shape[1]} features")
    print(f"Số lượng nhãn: {len(np.unique(y))}")
    
    # Bước 2: Kiểm tra và xử lý giá trị thiếu
    print(f"\n2. Kiểm tra giá trị thiếu")
    missing_count = check_missing_values(X)
    if missing_count > 0:
        print("Xử lý giá trị thiếu bằng phương pháp nội suy...")
        X = handle_missing_values(X, strategy='interpolate')
    
    # Bước 3: Làm mịn dữ liệu nếu cần
    if smooth_data:
        print(f"\n3. Làm mịn dữ liệu")
        X = smooth_landmarks(X)
        print(f"Đã làm mịn dữ liệu với bộ lọc Savitzky-Golay")
    
    # Bước 4: Chuẩn hóa dữ liệu
    print(f"\n4. Chuẩn hóa dữ liệu với phương pháp {normalize_method}")
    scaler_path = os.path.join(output_directory, f'scaler_{normalize_method}.pkl')
    X, scaler = normalize_data(X, method=normalize_method, save_scaler=True, scaler_path=scaler_path)
    
    # Bước 5: Phân tích và giảm chiều dữ liệu nếu cần
    if use_pca:
        print(f"\n5. Giảm chiều dữ liệu với PCA")
        X, pca_model = feature_selection_pca(X, n_components=pca_components)
        pca_path = os.path.join(output_directory, 'pca_model.pkl')
        joblib.dump(pca_model, pca_path)
        print(f"Đã lưu mô hình PCA tại: {pca_path}")
    
    # Bước 6: Cân bằng dữ liệu nếu cần
    if balance_classes:
        print(f"\n6. Cân bằng dữ liệu")
        X, y = balance_dataset(X, y)
    
    # Bước 7: Hiển thị phân bố dữ liệu sau xử lý
    print(f"\n7. Hiển thị phân bố dữ liệu")
    visualize_data_distribution(X, y, label_map)
    
    # Bước 8: Lưu dữ liệu đã xử lý
    print(f"\n8. Lưu dữ liệu đã xử lý")
    output_path = os.path.join(output_directory, 'processed_data.npz')
    
    metadata = {
        'features': X.shape[1],
        'samples': X.shape[0],
        'classes': len(np.unique(y)),
        'class_names': label_map,
        'normalize_method': normalize_method,
        'smoothed': smooth_data,
        'balanced': balance_classes,
        'pca_applied': use_pca,
        'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_processed_data(X, y, output_path, metadata)
    
    print(f"\n===== HOÀN THÀNH TIỀN XỬ LÝ =====")
    print(f"Dữ liệu đã được lưu tại: {output_path}")

if __name__ == "__main__":
    # Cấu hình đường dẫn thư mục
    input_dir = r"D:\System\Videos\VideoProc_Converter_AI\make_data"
    output_dir = r"D:\System\Videos\VideoProc_Converter_AI\processed_data"
    
    # Thực hiện tiền xử lý
    preprocess_vsl_data(
        input_directory=input_dir,
        output_directory=output_dir,
        normalize_method='minmax',  # 'minmax' hoặc 'standard'
        smooth_data=True,
        balance_classes=True,
        use_pca=True,
        pca_components=0.95
    )