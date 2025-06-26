import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import h5py
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import tempfile
import shutil
import traceback

# Import the model classes from the notebook
import torch.nn as nn
import lightning as L
import torchmetrics

# Try to import optional dependencies
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO not available: {e}")
    YOLO_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global configuration (same as in notebook)
YOLO_MODEL_NAME = 'yolov8n.pt'
YOLO_CONF_THRESHOLD = 0.4
HAND_DETECTION_TARGET_CLASSES = [0]
MAX_NUM_HANDS_MEDIAPIPE = 2
MIN_DETECTION_CONF_MEDIAPIPE = 0.5
MIN_TRACKING_CONF_MEDIAPIPE = 0.5
NUM_FRAMES_PER_VIDEO = 30
NUM_KEYPOINTS_PER_HAND = 21
NUM_COORDS_PER_KEYPOINT = 2
INPUT_SIZE = MAX_NUM_HANDS_MEDIAPIPE * NUM_KEYPOINTS_PER_HAND * NUM_COORDS_PER_KEYPOINT

# Global variables for model and data
trained_model = None
data_module = None
yolo_detector = None
mp_hands = None
mp_drawing = None

# LSTMClassifier class (same as in notebook)
class LSTMClassifier(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

    def _common_step(self, batch, batch_idx):
        x, y_true = batch
        logits = self(x)
        loss = self.criterion(logits, y_true)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y_true

    def training_step(self, batch, batch_idx):
        loss, preds, y_true = self._common_step(batch, batch_idx)
        acc = self.train_acc(preds, y_true)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y_true = self._common_step(batch, batch_idx)
        acc = self.val_acc(preds, y_true)
        f1 = self.val_f1(preds, y_true)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, preds, y_true = self._common_step(batch, batch_idx)
        acc = self.test_acc(preds, y_true)
        f1 = self.test_f1(preds, y_true)
        self.log('test_loss', loss, logger=True)
        self.log('test_acc', acc, logger=True)
        self.log('test_f1', f1, logger=True)
        return {'loss': loss, 'preds': preds, 'targets': y_true}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.2,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
            },
        }

def load_model_and_data():
    """Load the trained model and data module"""
    global trained_model, data_module, yolo_detector, mp_hands, mp_drawing
    
    # Check if MediaPipe is available
    if not MEDIAPIPE_AVAILABLE:
        print("❌ MediaPipe is not available. Please install it: pip install mediapipe")
        return False
    
    # Check if YOLO is available
    if not YOLO_AVAILABLE:
        print("❌ YOLO is not available. Please install it: pip install ultralytics")
        return False
    
    try:
        # Load YOLO detector
        yolo_detector = YOLO(YOLO_MODEL_NAME)
        print("✅ YOLO detector loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLO detector: {e}")
        yolo_detector = None
    
    # Initialize MediaPipe
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        print("✅ MediaPipe initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe: {e}")
        return False
    
    # Load the best model
    model_path = Path("./model_yolo/best-action-model-epoch=31-val_f1=0.92.ckpt")
    if not model_path.exists():
        # Try alternative model
        model_path = Path("./model_yolo/best-action-model-epoch=13-val_f1=0.88-v3.ckpt")
    
    if model_path.exists():
        try:
            trained_model = LSTMClassifier.load_from_checkpoint(str(model_path))
            trained_model.eval()
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    else:
        print("❌ No trained model found!")
        return False
    
    # Load data module to get class names
    hdf5_file_path = Path("./data-yolo/model.h5")
    if hdf5_file_path.exists():
        try:
            with h5py.File(hdf5_file_path, 'r') as hf:
                data_module = type('DataModule', (), {})()
                data_module.label_map = json.loads(hf.attrs['label_map'])
                data_module.num_classes = len(data_module.label_map)
                data_module.inv_label_map = {v: k for k, v in data_module.label_map.items()}
            print("✅ Data module loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load data module: {e}")
            return False
    else:
        print("❌ HDF5 file not found!")
        return False
    
    return True

def extract_normalized_hand_kps(landmarks, image_width, image_height):
    """Extract and flatten (x, y) coordinates from MediaPipe landmarks"""
    if not landmarks:
        return np.zeros(NUM_KEYPOINTS_PER_HAND * NUM_COORDS_PER_KEYPOINT)
    
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y])
    
    return np.array(coords)

def process_frame_for_keypoints(frame, hands_solution, yolo_model=None):
    """Process a single frame to extract keypoints"""
    frame_annotated = frame.copy()
    image_height, image_width = frame.shape[:2]
    
    # Initialize keypoints for this frame
    keypoints_frame_hands = [np.zeros(NUM_KEYPOINTS_PER_HAND * NUM_COORDS_PER_KEYPOINT) for _ in range(MAX_NUM_HANDS_MEDIAPIPE)]
    
    # YOLO ROI detection (optional)
    yolo_rois = []
    if yolo_model is not None:
        try:
            results = yolo_model.predict(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) in HAND_DETECTION_TARGET_CLASSES:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            yolo_rois.append([int(x1), int(y1), int(x2), int(y2)])
                            cv2.rectangle(frame_annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        except Exception as e:
            print(f"YOLO detection failed: {e}")
    
    # Process regions (ROIs or full frame)
    regions_to_process = []
    if yolo_rois:
        for roi in yolo_rois:
            x1, y1, x2, y2 = roi
            regions_to_process.append((frame[y1:y2, x1:x2], x1, y1))
    else:
        regions_to_process.append((frame, 0, 0))
    
    # Process each region with MediaPipe
    for region, offset_x, offset_y in regions_to_process:
        if region.size == 0:
            continue
            
        # Convert to RGB for MediaPipe
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = hands_solution.process(region_rgb)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= MAX_NUM_HANDS_MEDIAPIPE:
                    break
                
                # Convert region-relative coordinates to full-frame coordinates
                temp_full_frame_landmarks = []
                for lm in hand_landmarks.landmark:
                    full_x = (lm.x * region.shape[1] + offset_x) / image_width
                    full_y = (lm.y * region.shape[0] + offset_y) / image_height
                    temp_full_frame_landmarks.append(type('Landmark', (), {'x': full_x, 'y': full_y})())
                
                # Extract keypoints
                hand_kps = extract_normalized_hand_kps(temp_full_frame_landmarks, image_width, image_height)
                keypoints_frame_hands[hand_idx] = hand_kps
                
                # Draw landmarks on annotated frame
                for lm in temp_full_frame_landmarks:
                    x, y = int(lm.x * image_width), int(lm.y * image_height)
                    cv2.circle(frame_annotated, (x, y), 3, (0, 0, 255), -1)
    
    # Concatenate all hand keypoints
    final_keypoints_for_frame = np.concatenate(keypoints_frame_hands)
    
    return final_keypoints_for_frame, frame_annotated

def process_video_for_inference(video_path):
    """Process a video file for inference - matching the notebook code"""
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "Could not open video file"
    
    video_keypoints_list = []
    processed_frames = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS_MEDIAPIPE,
        min_detection_confidence=MIN_DETECTION_CONF_MEDIAPIPE,
        min_tracking_confidence=MIN_TRACKING_CONF_MEDIAPIPE
    ) as hands_solution:
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            keypoints_single_frame, frame_annotated = process_frame_for_keypoints(
                frame, hands_solution, yolo_detector
            )
            
            video_keypoints_list.append(keypoints_single_frame)
            processed_frames.append(frame_annotated)
            frame_idx += 1
    
    cap.release()
    
    if not video_keypoints_list:
        return None, "No keypoints extracted from video"
    
    print(f"Extracted {len(video_keypoints_list)} frames")
    
    # Convert to numpy array - matching notebook code
    video_keypoints_np = np.array(video_keypoints_list, dtype=np.float32)
    current_num_frames = video_keypoints_np.shape[0]
    
    # Pad or truncate to fixed length - matching notebook code
    if current_num_frames > NUM_FRAMES_PER_VIDEO:
        # Truncate from middle
        start = (current_num_frames - NUM_FRAMES_PER_VIDEO) // 2
        processed_sequence = video_keypoints_np[start:start + NUM_FRAMES_PER_VIDEO]
        processed_frames = processed_frames[start:start + NUM_FRAMES_PER_VIDEO]
    elif current_num_frames < NUM_FRAMES_PER_VIDEO:
        # Pad with zeros
        padding_frames = NUM_FRAMES_PER_VIDEO - current_num_frames
        padding = np.zeros((padding_frames, INPUT_SIZE), dtype=np.float32)
        processed_sequence = np.concatenate((video_keypoints_np, padding), axis=0)
    else:
        processed_sequence = video_keypoints_np
    
    print(f"Final sequence shape: {processed_sequence.shape}")
    return processed_sequence, processed_frames

def predict_sign(video_path):
    """Predict the sign from a video file - matching notebook code"""
    if trained_model is None:
        return None, "Model not loaded"
    
    try:
        # Process video
        processed_sequence, processed_frames = process_video_for_inference(video_path)
        if processed_sequence is None:
            return None, processed_frames  # processed_frames contains error message
        
        # Convert to tensor - matching notebook code
        sequence_tensor = torch.from_numpy(processed_sequence).float().unsqueeze(0)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        trained_model.to(device)
        trained_model.eval()
        sequence_tensor = sequence_tensor.to(device)
        
        # Make prediction - matching notebook code
        with torch.no_grad():
            logits = trained_model(sequence_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class_idx = predicted_idx.item()
            prediction_confidence = confidence.item()
        
        # Get class name
        predicted_class_name = data_module.inv_label_map.get(predicted_class_idx, "Unknown")
        
        # Get all class probabilities
        all_probabilities = {}
        for i in range(data_module.num_classes):
            class_name_i = data_module.inv_label_map.get(i, f"Class_{i}")
            all_probabilities[class_name_i] = probabilities[0][i].item()
        
        print(f"Predicted: {predicted_class_name} (Confidence: {prediction_confidence:.2f})")
        
        return {
            'predicted_class': predicted_class_name,
            'confidence': prediction_confidence,
            'all_probabilities': all_probabilities
        }, None
        
    except Exception as e:
        print(f"Error in predict_sign: {e}")
        traceback.print_exc()
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload with better error handling"""
    try:
        print("Upload request received")
        
        if 'video' not in request.files:
            print("No 'video' field in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            print("No filename provided")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Received file: {file.filename}")
        
        # Check file type
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
        
        # Check file size
        if len(file.read()) > app.config['MAX_CONTENT_LENGTH']:
            file.seek(0)  # Reset file pointer
            return jsonify({'error': 'File too large. Max size: 16MB'}), 400
        
        file.seek(0)  # Reset file pointer
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Predict the sign
            print("Starting prediction...")
            result, error = predict_sign(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
                print("Uploaded file cleaned up")
            except:
                pass
            
            if error:
                print(f"Prediction error: {error}")
                return jsonify({'error': error}), 400
            
            print("Prediction successful")
            return jsonify(result)
            
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/record', methods=['POST'])
def record_video():
    # This would handle recorded video data
    # For now, we'll return a placeholder
    return jsonify({'error': 'Recording functionality not implemented yet'}), 501

if __name__ == '__main__':
    print("Loading model and data...")
    if load_model_and_data():
        print("Model and data loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model and data. Please check the model files.") 