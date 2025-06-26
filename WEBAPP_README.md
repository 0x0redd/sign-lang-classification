# Sign Language Recognition Web Application

A modern web application for real-time sign language recognition using deep learning. This application allows users to upload videos or record them directly through the browser to get instant predictions of sign language gestures.

## Features

- 🎥 **Video Upload**: Drag and drop or browse to upload video files
- 📹 **Live Recording**: Record videos directly through your webcam
- 🤖 **AI Recognition**: Uses trained LSTM model for accurate sign language recognition
- 📊 **Confidence Scores**: Shows prediction confidence and probabilities for all classes
- 🎨 **Modern UI**: Beautiful, responsive design with real-time feedback
- 📱 **Mobile Friendly**: Works on desktop and mobile devices

## Supported Sign Classes

The application recognizes the following sign language gestures:
- **clavier** (keyboard)
- **disque_dur** (hard drive)
- **ordinateur** (computer)
- **souris** (mouse)

## Prerequisites

Before running the web application, ensure you have:

1. **Trained Model**: The LSTM model must be trained using the notebook (`model - FINAL .ipynb`)
2. **Required Files**:
   - `yolov8n.pt` (YOLO model for hand detection)
   - `data-yolo/model.h5` (processed dataset with class mappings)
   - At least one trained model checkpoint in `model_yolo/` directory

## Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd sign-lang-classification
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify required files exist**:
   ```bash
   python run_app.py
   ```
   This will check if all required files are present.

## Usage

### Quick Start

1. **Run the startup script**:
   ```bash
   python run_app.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Use the application**:
   - **Upload Video**: Click "Choose File" or drag and drop a video file
   - **Record Video**: Click "Record Video" to use your webcam
   - **View Results**: See the predicted sign and confidence scores

### Alternative: Direct Flask Run

If you prefer to run Flask directly:

```bash
python app.py
```

## How It Works

1. **Video Processing**: The uploaded/recorded video is processed frame by frame
2. **Hand Detection**: YOLOv8 detects regions of interest (people/hands)
3. **Keypoint Extraction**: MediaPipe extracts 21 hand landmarks per hand
4. **Feature Processing**: Keypoints are normalized and formatted for the model
5. **AI Prediction**: The LSTM model predicts the sign language gesture
6. **Results Display**: Shows the predicted class and confidence scores

## Technical Details

### Model Architecture
- **LSTM Classifier**: 2-layer LSTM with 256 hidden units
- **Input**: 84 features per frame (2 hands × 21 landmarks × 2 coordinates)
- **Output**: 4 classes with confidence scores
- **Training**: PyTorch Lightning with AdamW optimizer

### Video Processing
- **Frame Rate**: Processes all frames from the video
- **Sequence Length**: Pads/truncates to 30 frames
- **Hand Detection**: Supports up to 2 hands simultaneously
- **Normalization**: Coordinates normalized to [0, 1] range

### Web Technologies
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Video Recording**: MediaRecorder API
- **File Upload**: Drag & drop with progress feedback

## File Structure

```
sign-lang-classification/
├── app.py                 # Main Flask application
├── run_app.py            # Startup script with file checks
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface
├── uploads/             # Temporary upload directory
├── model_yolo/          # Trained model checkpoints
├── data-yolo/           # Processed dataset
├── yolov8n.pt           # YOLO model
└── SignLanguageDataset/ # Raw video dataset
```

## Troubleshooting

### Common Issues

1. **"Model not loaded" error**:
   - Ensure you've trained the model using the notebook
   - Check that model checkpoint files exist in `model_yolo/`

2. **"No video file provided" error**:
   - Make sure you're uploading a valid video file (MP4, AVI, MOV)
   - Check file size (max 16MB)

3. **Camera access denied**:
   - Allow camera permissions in your browser
   - Try using HTTPS (required for camera access on some browsers)

4. **Import errors**:
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

### Performance Tips

- **GPU Usage**: The app automatically uses GPU if available
- **Video Quality**: Lower resolution videos process faster
- **Browser**: Use Chrome or Firefox for best compatibility
- **Memory**: Close other applications if processing large videos

## Development

### Adding New Sign Classes

1. **Add videos** to `SignLanguageDataset/` in new class folders
2. **Retrain the model** using the notebook
3. **Update the web interface** if needed

### Customizing the UI

The web interface is in `templates/index.html`. You can modify:
- Colors and styling in the `<style>` section
- Layout and functionality in the `<script>` section
- Add new features like batch processing or history

### API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload and process video file
- `POST /record`: Record video from webcam (placeholder)

## License

This project is part of a sign language recognition system. Please refer to the main project license.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all required files are present
3. Check the console output for error messages
4. Ensure your environment meets the prerequisites

---

**Note**: This web application requires the trained model from the main project. Make sure to run the training notebook first before using the web interface. 