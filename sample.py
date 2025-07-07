import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
import seaborn as sns

# ==================== CONFIGURATION ====================
DATA_DIR = r"/kaggle/input/vsl-2-7/make_data"
TIMESTEPS = 15
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
NOISE_FACTOR = 0.005  # For data augmentation

print("🚀 Starting optimized LSTM training...")
print(f"TensorFlow version: {tf.__version__}")

# ==================== DATA PREPROCESSING ====================
def preprocess_sequence(df, scaler=None):
    """Preprocess a single CSV file"""
    # Convert to numeric and drop NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    # Standardize data
    if scaler is None:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = scaler.transform(df)
    
    return df_scaled, scaler

def augment_sequence(sequence, noise_factor=0.005):
    """Add noise for data augmentation"""
    noise = np.random.normal(0, noise_factor, sequence.shape)
    return sequence + noise

def create_windows(data, timesteps, augment=True, noise_factor=0.005):
    """Create sliding windows from sequence data"""
    X_windows = []
    
    for i in range(timesteps, len(data)):
        window = data[i - timesteps:i]
        if window.shape == (timesteps, 126):
            X_windows.append(window)
            
            # Data augmentation
            if augment and np.random.random() < 0.3:  # 30% chance
                augmented_window = augment_sequence(window, noise_factor)
                X_windows.append(augmented_window)
    
    return X_windows

# ==================== LOAD AND PROCESS DATA ====================
X, y = [], []
global_scaler = None
processed_files = 0
total_files = 0

print("📊 Loading and processing data...")

# Count total files first
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_dir):
        total_files += len([f for f in os.listdir(label_dir) if f.endswith('.csv')])

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    
    label_windows = []
    
    for file in os.listdir(label_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(label_dir, file), header=None, skiprows=1)
                
                if len(df) < TIMESTEPS:
                    print(f"⚠️  Skipping {file}: insufficient data ({len(df)} < {TIMESTEPS})")
                    continue
                
                # Preprocess data
                df_scaled, scaler = preprocess_sequence(df, global_scaler)
                
                # Save the first scaler as global scaler
                if global_scaler is None:
                    global_scaler = scaler
                
                # Create windows
                windows = create_windows(df_scaled, TIMESTEPS, augment=True, noise_factor=NOISE_FACTOR)
                label_windows.extend(windows)
                
                processed_files += 1
                if processed_files % 10 == 0:
                    print(f"📈 Processed {processed_files}/{total_files} files")
                    
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
    
    # Add all windows for this label
    X.extend(label_windows)
    y.extend([label] * len(label_windows))
    
    print(f"✅ Label '{label}': {len(label_windows)} windows")

# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n📋 Dataset Summary:")
print(f"   Total samples: {len(X)}")
print(f"   Data shape: {X.shape}")
print(f"   Labels: {np.unique(y)}")
print(f"   Samples per label: {[(label, np.sum(y == label)) for label in np.unique(y)]}")

# ==================== LABEL ENCODING ====================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = len(label_encoder.classes_)

print(f"📝 Number of classes: {num_classes}")

# ==================== TRAIN/TEST SPLIT ====================
# Use stratified split to maintain class distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y_encoded))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]

print(f"🔄 Train/Test Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ==================== CLASS WEIGHTS ====================
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))
print(f"⚖️  Class weights: {class_weight_dict}")

# ==================== MODEL ARCHITECTURE ====================
def build_optimized_model(input_shape, num_classes):
    """Build an optimized CNN-LSTM model"""
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # LSTM layers
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Dense layers with regularization
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.5),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Build model
print("🏗️  Building optimized model...")
model = build_optimized_model((X_train.shape[1], X_train.shape[2]), num_classes)

# Print model summary
model.summary()

# ==================== COMPILE MODEL ====================
optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# ==================== CALLBACKS ====================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model_2_7.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]

# ==================== TRAINING ====================
print("🎯 Starting training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ==================== SAVE ARTIFACTS ====================
print("💾 Saving model and artifacts...")

# Save model
model.save("model_lstm_optimized_2_7.keras")

# Save test data
np.save("X_test_optimized_2_7.npy", X_test)
np.save("y_test_optimized_2_7.npy", y_test)
np.save("labels_optimized_2_7.npy", label_encoder.classes_)

# Save label encoder
with open("label_encoder_optimized_2_7.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save scaler
with open("scaler_optimized_2_7.pkl", "wb") as f:
    pickle.dump(global_scaler, f)

print("✅ All artifacts saved successfully!")

# ==================== EVALUATION ====================
print("📊 Evaluating model...")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\n📋 Classification Report:")
report = classification_report(y_true_classes, y_pred_classes, 
                             target_names=label_encoder.classes_)
print(report)

# ==================== VISUALIZATION ====================
def plot_training_history(history):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='red')
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Precision
    ax3.plot(history.history['precision'], label='Train Precision', color='blue')
    ax3.plot(history.history['val_precision'], label='Val Precision', color='red')
    ax3.set_title('Model Precision', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True)
    
    # Recall
    ax4.plot(history.history['recall'], label='Train Recall', color='blue')
    ax4.plot(history.history['val_recall'], label='Val Recall', color='red')
    ax4.set_title('Model Recall', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.show()

# Plot results
print("📈 Generating visualizations...")
plot_training_history(history)
plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_)

# ==================== FINAL SUMMARY ====================
final_accuracy = max(history.history['val_accuracy'])
final_loss = min(history.history['val_loss'])

print(f"\n🎉 Training Complete!")
print(f"   Best Validation Accuracy: {final_accuracy:.4f}")
print(f"   Best Validation Loss: {final_loss:.4f}")
print(f"   Total Training Time: {len(history.history['loss'])} epochs")
print(f"   Model saved as: model_lstm_optimized_2_7.keras")

# ==================== PREDICTION FUNCTION ====================
def predict_new_sample(model, scaler, label_encoder, sample_data):
    """
    Predict on new sample data
    
    Args:
        model: trained model
        scaler: fitted StandardScaler
        label_encoder: fitted LabelEncoder
        sample_data: raw CSV data (numpy array) - shape should be (frames, 126)
    """
    # Preprocess
    sample_scaled = scaler.transform(sample_data)
    
    # Create windows
    if len(sample_scaled) >= TIMESTEPS:
        # Take the last TIMESTEPS as input
        X_sample = sample_scaled[-TIMESTEPS:].reshape(1, TIMESTEPS, -1)
        
        # Predict
        prediction = model.predict(X_sample)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        
        # Decode label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_label, confidence
    else:
        return None, 0.0

# ==================== COMPATIBLE PREDICTION FUNCTION ====================
async def predict_sequence_from_frames(frames, model, scaler, label_encoder):
    """
    Predict function compatible with your backend
    
    Args:
        frames: List of frames, each frame has 126 features [x1,y1,z1,x2,y2,z2,...] for 2 hands
        model: trained model
        scaler: fitted StandardScaler  
        label_encoder: fitted LabelEncoder
    """
    try:
        # Validate input
        if not frames or not isinstance(frames, list):
            return {"label": "unknown", "confidence": 0.0, "message": "Dữ liệu không hợp lệ"}
        
        # Check frame format
        if not all(len(f) == 126 for f in frames):
            return {"label": "unknown", "confidence": 0.0, "message": "Mỗi frame cần có 126 features"}
        
        # Convert to numpy array
        sequence = np.array(frames, dtype=np.float32)
        
        # Check if we have enough frames
        if len(sequence) < TIMESTEPS:
            return {"label": "unknown", "confidence": 0.0, "message": f"Cần ít nhất {TIMESTEPS} frames"}
        
        # Preprocess with scaler (IMPORTANT: this was missing in original code)
        sequence_scaled = scaler.transform(sequence)
        
        # Take the last TIMESTEPS frames for prediction
        if len(sequence_scaled) > TIMESTEPS:
            input_sequence = sequence_scaled[-TIMESTEPS:]
        else:
            input_sequence = sequence_scaled
            
        # Reshape for model input: (1, timesteps, 126)
        input_data = np.expand_dims(input_sequence, axis=0)
        
        # Predict
        prediction = model.predict(input_data, verbose=0)[0]
        
        # Get results
        max_index = np.argmax(prediction)
        max_label = label_encoder.inverse_transform([max_index])[0]
        confidence = prediction[max_index]
        
        # Confidence threshold
        if confidence < 0.85:
            return {
                "label": "unknown", 
                "confidence": float(confidence),
                "message": "Chưa hiểu hành động"
            }
        else:
            return {
                "label": max_label,
                "confidence": float(confidence), 
                "message": max_label
            }
            
    except Exception as e:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "message": f"Lỗi prediction: {str(e)}"
        }

print("\n🔮 Prediction function ready!")
print("   Use predict_new_sample() to predict on new data")
print("   Example: predict_new_sample(model, global_scaler, label_encoder, your_data)")