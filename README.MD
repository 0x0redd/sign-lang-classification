# Document: Video Action/Sign Recognition Pipeline using Keypoints and Sequence Models

**Author:** (Derived from Notebook `model - yolo .ipynb`)
**Date:** (Based on execution counts, likely recent)

---

**Table of Contents:**

1.  [General Idea and Overview](#1-general-idea-and-overview)
    *   [1.1. Problem Statement](#11-problem-statement)
    *   [1.2. Pipeline Approach](#12-pipeline-approach)
    *   [1.3. Key Technologies and Definitions](#13-key-technologies-and-definitions)
2.  [Etape 1: Setup and Configuration](#2-etape-1-setup-and-configuration)
    *   [2.1. Library Installation and Imports](#21-library-installation-and-imports)
    *   [2.2. Global Configuration Parameters](#22-global-configuration-parameters)
    *   [2.3. Reproducibility](#23-reproducibility)
3.  [Etape 2: Dataset Preparation (Optional: Dummy Data)](#3-etape-2-dataset-preparation-optional-dummy-data)
    *   [3.1. Expected Dataset Structure](#31-expected-dataset-structure)
    *   [3.2. Dummy Video Dataset Creation](#32-dummy-video-dataset-creation)
4.  [Etape 3: Preprocessing - Keypoint Extraction](#4-etape-3-preprocessing---keypoint-extraction)
    *   [4.1. Initialization of Detectors (YOLO & MediaPipe)](#41-initialization-of-detectors-yolo--mediapipe)
    *   [4.2. Helper Function: `extract_normalized_hand_kps`](#42-helper-function-extract_normalized_hand_kps)
    *   [4.3. Core Helper Function: `process_frame_for_keypoints`](#43-core-helper-function-process_frame_for_keypoints)
    *   [4.4. Main Preprocessing Loop: `preprocess_videos_to_hdf5`](#44-main-preprocessing-loop-preprocess_videos_to_hdf5)
    *   [4.5. HDF5 Data Storage](#45-hdf5-data-storage)
5.  [Etape 4: Data Loading and Preparation for Training](#5-etape-4-data-loading-and-preparation-for-training)
    *   [5.1. `KeypointSequenceDataset` (PyTorch Dataset)](#51-keypointsequencedataset-pytorch-dataset)
    *   [5.2. `KeypointDataModule` (PyTorch Lightning DataModule)](#52-keypointdatamodule-pytorch-lightning-datamodule)
    *   [5.3. Data Splitting](#53-data-splitting)
6.  [Etape 5: Sequence Model Definition](#6-etape-5-sequence-model-definition)
    *   [6.1. `LSTMClassifier` (PyTorch Lightning Module)](#61-lstmclassifier-pytorch-lightning-module)
    *   [6.2. Model Architecture (LSTM)](#62-model-architecture-lstm)
    *   [6.3. Loss Function, Metrics, and Optimizer](#63-loss-function-metrics-and-optimizer)
7.  [Etape 6: Model Training](#7-etape-6-model-training)
    *   [7.1. Callbacks (Checkpointing, Early Stopping, LR Monitoring)](#71-callbacks-checkpointing-early-stopping-lr-monitoring)
    *   [7.2. Logger (CSVLogger)](#72-logger-csvlogger)
    *   [7.3. PyTorch Lightning Trainer Setup](#73-pytorch-lightning-trainer-setup)
    *   [7.4. Training Execution](#74-training-execution)
8.  [Etape 7: Evaluation](#8-etape-7-evaluation)
    *   [8.1. Loading the Best Model](#81-loading-the-best-model)
    *   [8.2. Plotting Training Curves](#82-plotting-training-curves)
    *   [8.3. Test Set Evaluation](#83-test-set-evaluation)
    *   [8.4. Classification Report and Confusion Matrix](#84-classification-report-and-confusion-matrix)
9.  [Etape 8: Inference on a New Video](#9-etape-8-inference-on-a-new-video)
    *   [9.1. Video Selection and Keypoint Extraction](#91-video-selection-and-keypoint-extraction)
    *   [9.2. Prediction](#92-prediction)
    *   [9.3. Visualization](#93-visualization)
10. [Conclusion](#10-conclusion)

---

## 1. General Idea and Overview

### 1.1. Problem Statement
The goal is to recognize actions or sign language gestures from video input. This involves processing video frames, extracting relevant features, and then using a sequence model to classify the entire video sequence into predefined action/sign categories.

### 1.2. Pipeline Approach
The notebook implements a multi-stage pipeline:
1.  **Preprocessing:** This is a crucial step where raw video data is transformed into a more manageable and feature-rich format.
    *   **Frame-wise Processing:** Videos are broken down into individual frames.
    *   **Region of Interest (ROI) Detection (Optional):** YOLOv8 is used to detect persons or hands within each frame. This helps focus the keypoint extraction on relevant areas, potentially improving accuracy and efficiency.
    *   **Keypoint Extraction:** MediaPipe Hands is then applied (either to the ROIs or the full frame) to extract 2D keypoints (landmarks) for up to two hands.
    *   **Normalization:** Keypoints are normalized to be independent of image size and position.
    *   **Sequence Creation:** For each video, a fixed number of frames (`NUM_FRAMES_PER_VIDEO`) is selected (by padding or truncating), and their corresponding keypoint vectors are concatenated to form a sequence.
2.  **Dataset Creation:** The extracted keypoint sequences and their corresponding labels are stored in an HDF5 file for efficient loading during training.
3.  **Sequence Modeling:** An LSTM-based neural network is trained to learn patterns from these keypoint sequences and classify them. The notebook mentions that GRU or Transformer models could be substituted.
4.  **Evaluation & Inference:** The trained model is evaluated on a test set, and its performance is demonstrated by making predictions on a sample video.

### 1.3. Key Technologies and Definitions
*   **YOLO (You Only Look Once):** A real-time object detection system. In this notebook, `yolov8n.pt` (a small, fast YOLOv8 model) is used for detecting regions of interest (hands or persons).
*   **MediaPipe:** A framework by Google for building multimodal (e.g., video, audio) applied machine learning pipelines. `mp.solutions.hands` is specifically used here for robust hand tracking and 21 keypoint (landmark) extraction per hand.
*   **Keypoints (Landmarks):** Specific points of interest on an object (e.g., finger joints, palm center). Each keypoint has coordinates (x, y, and sometimes z and visibility). Here, 2D (x, y) coordinates are used.
*   **Normalization:** The process of scaling keypoint coordinates (usually between 0 and 1) relative to the frame dimensions. This makes the model invariant to video resolution and the absolute position of the hands in the frame.
*   **Sequence:** An ordered set of data points. In this context, a sequence of keypoint vectors, where each vector represents the hand keypoints from one video frame.
*   **HDF5 (Hierarchical Data Format 5):** A file format designed to store and organize large amounts of numerical data. It's suitable for storing the extracted keypoint sequences.
*   **LSTM (Long Short-Term Memory):** A type of Recurrent Neural Network (RNN) architecture that is well-suited for learning from sequential data, as it can capture long-range dependencies.
*   **PyTorch:** An open-source machine learning library.
*   **PyTorch Lightning:** A lightweight PyTorch wrapper that organizes PyTorch code, making it easier to train and scale models, handle hardware, and implement best practices with less boilerplate.
*   **Epoch:** One complete pass through the entire training dataset.
*   **Batch Size:** The number of training examples utilized in one iteration (one forward/backward pass).
*   **Learning Rate:** A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
*   **Loss Function:** Measures how well the model's predictions match the true labels (e.g., `CrossEntropyLoss` for classification).
*   **Optimizer:** An algorithm that adjusts the model's weights to minimize the loss function (e.g., `AdamW`).
*   **Metrics:** Quantitative measures of model performance (e.g., accuracy, F1-score).

## 2. Etape 1: Setup and Configuration

### 2.1. Library Installation and Imports
*   **Action:** The first code cell uses `%pip install` to ensure all necessary Python libraries are available.
*   **Libraries Installed:** `ultralytics`, `mediapipe`, `opencv-python`, `numpy`, `torch`, `torchvision`, `torchaudio`, `tqdm`, `scikit-learn`, `matplotlib`, `h5py`, `lightning`, `pandas`, `ipywidgets`.
*   **Imports:** Standard Python modules (`os`, `shutil`, etc.) and the installed libraries are imported for use throughout the notebook.
*   **PyTorch Setting:** `torch.set_float32_matmul_precision('medium')` is set for potentially faster matrix multiplications on compatible hardware.

### 2.2. Global Configuration Parameters
*   **Action:** A dedicated code cell defines various global parameters that control the pipeline's behavior.
*   **Path Definitions:**
    *   `RAW_VIDEO_DATA_DIR`: Path to the directory containing raw video files, organized into subdirectories by class.
    *   `PROCESSED_DATA_DIR`: Directory to save the processed HDF5 file.
    *   `H5_FILENAME`: Name of the HDF5 file (e.g., "model.h5").
    *   `MODEL_SAVE_DIR`: Directory to save trained model checkpoints.
*   **Preprocessing Parameters:**
    *   `YOLO_MODEL_NAME`: Specifies the YOLO model to use (e.g., 'yolov8n.pt').
    *   `YOLO_CONF_THRESHOLD`: Minimum confidence for YOLO detections.
    *   `HAND_DETECTION_TARGET_CLASSES`: COCO class index for "person" (used as ROI for MediaPipe if a dedicated hand detector isn't used).
    *   `MAX_NUM_HANDS_MEDIAPIPE`, `MIN_DETECTION_CONF_MEDIAPIPE`, `MIN_TRACKING_CONF_MEDIAPIPE`: MediaPipe Hands configuration.
    *   `NUM_FRAMES_PER_VIDEO`: Fixed number of frames per video sequence (padding/truncation applied).
    *   `NUM_KEYPOINTS_PER_HAND`: 21 for MediaPipe hands.
    *   `NUM_COORDS_PER_KEYPOINT`: 2 (for x, y).
    *   `INPUT_SIZE`: Calculated as `MAX_NUM_HANDS_MEDIAPIPE * NUM_KEYPOINTS_PER_HAND * NUM_COORDS_PER_KEYPOINT` (e.g., 2 * 21 * 2 = 84 features per frame).
*   **Model & Training Parameters:**
    *   `SEQUENCE_MODEL_TYPE`: Set to "LSTM".
    *   LSTM Hyperparameters: `HIDDEN_SIZE_LSTM`, `NUM_LAYERS_LSTM`, `DROPOUT_LSTM`.
    *   Common Training Parameters: `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `EPOCHS`, `PATIENCE_EARLY_STOPPING`.
*   **Directory Creation:** The specified directories are created if they don't already exist.

### 2.3. Reproducibility
*   **Action:** `RANDOM_SEED` is set, and this seed is applied to `random`, `numpy`, `torch`, and `pytorch_lightning.seed_everything()` to ensure that experiments can be reproduced with the same results.
*   `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` is set for deterministic operations on CUDA (GPU) if available.

## 3. Etape 2: Dataset Preparation (Optional: Dummy Data)

### 3.1. Expected Dataset Structure
The notebook expects videos to be organized in `RAW_VIDEO_DATA_DIR` as follows:
RAW_VIDEO_DATA_DIR /
├── class1_name /
│ ├── video1.mp4
│ ├── video2.avi
├── class2_name /
│ ├── video3.mp4
└── ...
### 3.2. Dummy Video Dataset Creation
*   **Purpose:** If `RAW_VIDEO_DATA_DIR` is empty, this section creates a small, synthetic dataset for demonstration purposes.
*   **Action:** The `create_dummy_video` function generates simple MP4 videos containing moving text.
*   The script checks if `RAW_VIDEO_DATA_DIR` has any subdirectories. If not, it creates `DUMMY_CLASSES` (e.g., "sign_A", "sign_B") and populates them with a few dummy videos.
*   **User Note:** Users are explicitly instructed to replace this with their actual video dataset or point `RAW_VIDEO_DATA_DIR` correctly.

## 4. Etape 3: Preprocessing - Keypoint Extraction

### 4.1. Initialization of Detectors (YOLO & MediaPipe)
*   **MediaPipe Hands:** `mp_hands = mp.solutions.hands` and `mp_drawing = mp.solutions.drawing_utils` are initialized.
*   **YOLO Detector:** `yolo_detector = YOLO(YOLO_MODEL_NAME)` loads the specified YOLOv8 model. Error handling is included in case the model fails to load.

### 4.2. Helper Function: `extract_normalized_hand_kps`
*   **Purpose:** To extract and flatten (x, y) coordinates from a list of MediaPipe landmark objects.
*   **Input:** A list of landmark objects (each with `.x` and `.y` attributes, already normalized by MediaPipe to image dimensions) and image width/height (though not strictly used in this version as landmarks are pre-normalized).
*   **Output:** A flat NumPy array of `[x1, y1, x2, y2, ..., xN, yN]` for a single hand.

### 4.3. Core Helper Function: `process_frame_for_keypoints`
This function (`process_frame_for_keypoints_fixed`, which overwrites the original placeholder) is the heart of the feature extraction per frame.
*   **Input:** A single video frame (NumPy array), the initialized MediaPipe `hands_solution`, and optionally the `yolo_model`.
*   **Steps:**
    1.  **YOLO ROI Detection (Optional):**
        *   If `yolo_model` is provided, it runs `yolo_model.predict()` on the frame to detect objects specified by `HAND_DETECTION_TARGET_CLASSES` (e.g., "person").
        *   Bounding boxes (`x1, y1, x2, y2`) for these detections are extracted and stored as `yolo_rois`. These ROIs are drawn on an annotated copy of the frame.
    2.  **Region Selection for MediaPipe:**
        *   If `yolo_rois` are found, the frame is cropped to these ROIs. MediaPipe Hands will be run on these smaller regions.
        *   If no YOLO ROIs, or if YOLO is not used, the entire frame is processed by MediaPipe.
    3.  **MediaPipe Hands Processing:**
        *   For each selected region (cropped ROI or full frame):
            *   The region is converted to RGB (MediaPipe expects RGB).
            *   `hands_solution.process(img_region_rgb)` is called to detect hands and their landmarks within that region.
    4.  **Keypoint Extraction and Normalization:**
        *   If hands are detected (`results.multi_hand_landmarks`):
            *   For each detected hand in the region:
                *   The landmarks (`lm_roi.x`, `lm_roi.y`) are initially relative to the processed region.
                *   These region-relative coordinates are converted to absolute pixel coordinates within the full frame by considering the region's offset (`offset_x`, `offset_y`).
                *   These absolute full-frame pixel coordinates are then normalized by dividing by `image_width` and `image_height` to get values between 0 and 1. These are stored in `temp_full_frame_landmarks_for_features`.
                *   `extract_normalized_hand_kps` is called on these full-frame normalized landmarks to get the flat (x,y) array for the current hand.
            *   Landmarks are drawn on `frame_annotated`.
    5.  **Hand Assignment and Padding:**
        *   The extracted keypoints are assigned to either the left hand (index 0) or right hand (index 1) array in `keypoints_frame_hands`. This list is pre-filled with zeros.
        *   MediaPipe's `multi_handedness` is used if available to determine 'Left' or 'Right'. If not, or if one hand is already filled, it fills the next available slot up to `MAX_NUM_HANDS_MEDIAPIPE`.
        *   This ensures that `keypoints_frame_hands` always contains keypoints for `MAX_NUM_HANDS_MEDIAPIPE` (e.g., 2 hands), with zero-padding if fewer hands are detected.
    6.  **Concatenation:** The keypoint arrays for all hands (e.g., left and right) are concatenated into a single flat NumPy array (`final_keypoints_for_frame`) of size `INPUT_SIZE`.
*   **Output:** `final_keypoints_for_frame` (the flat feature vector for the frame) and `frame_annotated` (the frame with drawings).

### 4.4. Main Preprocessing Loop: `preprocess_videos_to_hdf5`
*   **Purpose:** To process all videos in the dataset, extract keypoint sequences, and save them to an HDF5 file.
*   **Input:** Root directory of videos (`video_dir_root`), output HDF5 file path (`output_h5_path`), target number of frames (`num_frames_target`), and the YOLO model.
*   **Steps:**
    1.  **Discover Videos and Labels:** Traverses `video_dir_root`, identifies class names (subdirectories), and creates a `label_map` (e.g., {"sign_A": 0, "sign_B": 1}).
    2.  **Initialize MediaPipe Hands:** Creates a `mp_hands.Hands` instance with configured parameters. This is done outside the video loop for efficiency.
    3.  **Open HDF5 File:** Opens `output_h5_path` in write mode ('w'). The `label_map` is saved as a JSON string attribute to the HDF5 file.
    4.  **Iterate Through Videos (with `tqdm` progress bar):**
        *   For each `video_path`:
            *   Open video using `cv2.VideoCapture`.
            *   Initialize `video_keypoints_list` to store keypoints for each frame of the current video.
            *   **Frame-by-Frame Processing:**
                *   Read frames until the video ends.
                *   For each `frame`, call `process_frame_for_keypoints()` to get `keypoints_single_frame`.
                *   Append `keypoints_single_frame` to `video_keypoints_list_inf`.
            *   Release video capture.
            *   If no keypoints were extracted, skip the video.
            *   Convert `video_keypoints_list` to a NumPy array `video_keypoints_np`.
    5.  **Sequence Padding/Truncation:**
        *   `current_num_frames = video_keypoints_np.shape[0]`.
        *   If `current_num_frames > num_frames_target`: The sequence is truncated by taking a slice from the middle: `video_keypoints_np[start : start + num_frames_target]`.
        *   If `current_num_frames < num_frames_target`: The sequence is padded with zeros (arrays of shape `(padding_frames, INPUT_SIZE)`) at the end.
        *   The result is `processed_sequence` of shape `(num_frames_target, INPUT_SIZE)`.
    6.  **Save to HDF5:**
        *   Create a unique `dataset_name` based on the video's relative path.
        *   Create a dataset in the "sequences" group of the HDF5 file: `hf.create_dataset(dataset_name, data=processed_sequence)`.
        *   Store `label`, `original_path`, and `class_name` as attributes of this HDF5 dataset.
    7.  Close HDF5 file.
*   **Conditional Execution:** The `run_preprocessing` flag and a check for the HDF5 file's existence determine if this potentially time-consuming step is executed.

### 4.5. HDF5 Data Storage
The processed data is stored in an HDF5 file with the following structure:
*   Root Attributes:
    *   `label_map`: JSON string mapping class names to integer labels.
*   Group "sequences":
    *   Each video becomes a dataset within this group.
    *   Dataset Name: Derived from the video's path (e.g., `class_name_video_name_mp4`).
    *   Dataset Content: A NumPy array of shape `(NUM_FRAMES_PER_VIDEO, INPUT_SIZE)` containing the keypoint sequence.
    *   Dataset Attributes:
        *   `label`: Integer label of the class.
        *   `original_path`: String path to the original video file.
        *   `class_name`: String name of the class.

## 5. Etape 4: Data Loading and Preparation for Training

### 5.1. `KeypointSequenceDataset` (PyTorch Dataset)
*   **Purpose:** A custom PyTorch `Dataset` to load individual keypoint sequences and labels from the HDF5 file.
*   **Inheritance:** `torch.utils.data.Dataset`.
*   **`__init__(self, h5_file_path, dataset_keys)`:**
    *   Stores the HDF5 file path and a list of `dataset_keys` (strings corresponding to the dataset names in the HDF5 file for a particular split like train/val/test).
*   **`__len__(self)`:** Returns the total number of samples (videos) in this dataset split.
*   **`__getitem__(self, idx)`:**
    *   Opens the HDF5 file (in read mode).
    *   Retrieves the `dataset_key` for the given `idx`.
    *   Reads the keypoint sequence data (`hf['sequences'][dataset_key][:]`) and the label attribute (`hf['sequences'][dataset_key].attrs['label']`).
    *   Converts the sequence data to a `torch.FloatTensor` and the label to a `torch.LongTensor`.
    *   Returns the sequence tensor and label tensor.
*   **`__del__`, `__getstate__`, `__setstate__`:** These are included to manage the HDF5 file handle, particularly if `num_workers` in `DataLoader` were greater than 0. With `num_workers=0`, their impact is minimal.

### 5.2. `KeypointDataModule` (PyTorch Lightning DataModule)
*   **Purpose:** A PyTorch Lightning `DataModule` to encapsulate all data-related steps: downloading (not used here), preparing data, splitting, and creating DataLoaders.
*   **Inheritance:** `L.LightningDataModule`.
*   **`__init__(...)`:** Stores HDF5 path, batch size, validation/test split ratios, and random seed. Initializes `label_map`, `num_classes`, `inv_label_map`.
*   **`prepare_data(self)`:** A method that can be used for one-time setup (e.g., downloading). Here, it just checks if the HDF5 file exists.
*   **`setup(self, stage=None)`:**
    *   This method is called on every GPU in distributed training.
    *   Opens the HDF5 file to read `all_dataset_keys` (all video names) and the `label_map`.
    *   Determines `num_classes`.
    *   Creates `inv_label_map` (integer label to class name).
    *   **Data Splitting:** Uses `sklearn.model_selection.train_test_split` to divide `all_dataset_keys` into training, validation, and test sets.
        *   It first splits into `train_keys_val` and `test_keys`.
        *   Then, it splits `train_keys_val` into `train_keys` and `val_keys`.
        *   Stratification (`stratify=...`) is used if possible to maintain class proportions in splits.
    *   Prints dataset split sizes.
*   **`train_dataloader(self)`, `val_dataloader(self)`, `test_dataloader(self)`:**
    *   Each method creates a `KeypointSequenceDataset` instance with the appropriate keys for its split.
    *   Then, it returns a `torch.utils.data.DataLoader` configured with the dataset, `batch_size`, shuffle status (True for train, False for val/test), and `num_workers=0` (data loading happens in the main process). `pin_memory=True` can speed up data transfer to GPU.
*   **Initialization:** An instance of `KeypointDataModule` is created, and its `prepare_data()` and `setup()` methods are called.

### 5.3. Data Splitting
The `KeypointDataModule` handles splitting the dataset keys (video identifiers from HDF5) into three sets:
*   **Training Set:** Used to train the model.
*   **Validation Set:** Used to monitor model performance during training, tune hyperparameters, and for early stopping.
*   **Test Set:** Used for final, unbiased evaluation of the trained model.
The splitting is done using `train_test_split` from `scikit-learn`, with an attempt at stratified splitting to ensure that each class is proportionally represented in each set.

## 6. Etape 5: Sequence Model Definition

### 6.1. `LSTMClassifier` (PyTorch Lightning Module)
*   **Purpose:** Defines the neural network architecture (LSTM-based) and the training/validation/test logic within the PyTorch Lightning framework.
*   **Inheritance:** `L.LightningModule`.
*   **`__init__(...)`:**
    *   `self.save_hyperparameters()`: Saves constructor arguments (like `input_size`, `hidden_size`) to `self.hparams`, making them accessible later and saving them with checkpoints.
    *   Defines the model layers:
        *   `self.lstm = nn.LSTM(...)`: An LSTM layer. `batch_first=True` means input tensors will have shape `(batch, seq_len, features)`. Dropout is applied if `num_layers > 1`.
        *   `self.fc = nn.Linear(hidden_size, num_classes)`: A fully connected linear layer to map the LSTM output to class scores.
    *   `self.criterion = nn.CrossEntropyLoss()`: The loss function for multi-class classification.
    *   Initializes `torchmetrics` (Accuracy and F1Score) for training, validation, and testing. `task="multiclass"` and `num_classes` are specified.
*   **`forward(self, x)`:** Defines the forward pass.
    *   `lstm_out, (hn, cn) = self.lstm(x)`: Passes input `x` through LSTM. `hn` contains the hidden states of the last time step for each layer.
    *   `out = self.fc(hn[-1])`: Takes the hidden state of the last layer at the last time step (`hn[-1]`) and passes it through the fully connected layer to get class logits.
*   **`_common_step(self, batch, batch_idx)`:** A helper method to avoid code duplication in `training_step`, `validation_step`, and `test_step`.
    *   Unpacks the batch into input `x` and true labels `y_true`.
    *   Performs a forward pass: `logits = self(x)`.
    *   Calculates loss: `loss = self.criterion(logits, y_true)`.
    *   Gets predictions: `preds = torch.argmax(logits, dim=1)`.
    *   Returns `loss, preds, y_true`.
*   **`training_step(self, batch, batch_idx)`:**
    *   Calls `_common_step`.
    *   Calculates training accuracy using `self.train_acc`.
    *   Logs `train_loss` and `train_acc` using `self.log()`. `prog_bar=True` shows them on the progress bar.
    *   Returns the loss.
*   **`validation_step(self, batch, batch_idx)`:**
    *   Similar to `training_step`, but uses `self.val_acc` and `self.val_f1`.
    *   Logs `val_loss`, `val_acc`, and `val_f1`.
    *   Returns the loss (often used by callbacks like `ModelCheckpoint`).
*   **`test_step(self, batch, batch_idx)`:**
    *   Similar to `validation_step`, using `self.test_acc` and `self.test_f1`.
    *   Logs test metrics.
    *   Returns a dictionary containing loss, predictions, and targets, which can be aggregated by PyTorch Lightning.
*   **`configure_optimizers(self)`:**
    *   Defines the optimizer: `torch.optim.AdamW` is used, configured with `learning_rate` and `weight_decay` from `self.hparams`.
    *   Defines a learning rate scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau`. This scheduler reduces the learning rate if `val_f1` (the monitored metric) stops improving for a certain number of epochs (`patience`).
    *   Returns a dictionary specifying the optimizer and the LR scheduler configuration.
*   **Model Initialization:** An instance of `LSTMClassifier` is created using parameters from the global configuration and `data_module.num_classes`.

### 6.2. Model Architecture (LSTM)
The core of the sequence model is an LSTM layer followed by a linear (fully connected) layer.
*   **Input to LSTM:** A batch of sequences, each of shape `(NUM_FRAMES_PER_VIDEO, INPUT_SIZE)`.
*   **LSTM Output:** The LSTM processes the sequence and outputs all hidden states for all time steps, as well as the final hidden state (`hn`) and cell state (`cn`).
*   **Classification:** The final hidden state of the *last LSTM layer* (`hn[-1]`) is considered a summary of the entire input sequence. This summary vector is then fed into the linear layer (`self.fc`) to produce raw scores (logits) for each class. A softmax function (implicitly handled by `CrossEntropyLoss` during training, or applied manually during inference) converts these logits into probabilities.

### 6.3. Loss Function, Metrics, and Optimizer
*   **Loss Function:** `nn.CrossEntropyLoss` is used, standard for multi-class classification.
*   **Metrics:**
    *   `Accuracy`: The proportion of correctly classified sequences.
    *   `F1Score (macro)`: The macro-averaged F1 score, which is the unweighted mean of the F1 scores for each class. It's a good metric for imbalanced datasets.
*   **Optimizer:** `AdamW` (Adam with weight decay) is used to update model weights.
*   **Learning Rate Scheduler:** `ReduceLROnPlateau` adaptively reduces the learning rate when the validation F1 score plateaus, helping the model to converge better.

## 7. Etape 6: Model Training

### 7.1. Callbacks
PyTorch Lightning callbacks are used to add custom behavior during training:
*   **`ModelCheckpoint`:**
    *   Saves model checkpoints during training.
    *   `dirpath=MODEL_SAVE_DIR`: Specifies where to save.
    *   `filename='best-action-model-{epoch:02d}-{val_f1:.2f}'`: Naming pattern for saved files.
    *   `save_top_k=1`: Keeps only the best checkpoint.
    *   `monitor='val_f1'`, `mode='max'`: Monitors the validation F1 score and saves the model when it's at its maximum.
*   **`EarlyStopping`:**
    *   Stops training if the monitored metric (`val_f1`) does not improve for a specified number of epochs (`PATIENCE_EARLY_STOPPING`).
    *   `mode='max'`: Indicates that a higher `val_f1` is better.
*   **`LearningRateMonitor`:** Logs the learning rate at each epoch.

### 7.2. Logger
*   **`CSVLogger`:** Saves all logged metrics (loss, accuracy, F1, learning rate) to a CSV file (`metrics.csv`) in a subdirectory of `MODEL_SAVE_DIR`. This is useful for later analysis and plotting.

### 7.3. PyTorch Lightning Trainer Setup
*   An `L.Trainer` object is instantiated. This object handles the training loop, device management, callbacks, and logging.
*   **Key Trainer Arguments:**
    *   `max_epochs=EPOCHS`: Maximum number of training epochs.
    *   `accelerator="gpu" if torch.cuda.is_available() else "cpu"`: Automatically uses GPU if available.
    *   `devices=1`: Uses one GPU or one CPU.
    *   `logger=csv_logger`: Uses the configured CSV logger.
    *   `callbacks=[...]`: List of callbacks to use.
    *   `deterministic=True`: Aims for reproducible training runs (works in conjunction with seed setting).

### 7.4. Training Execution
*   `trainer.fit(model, datamodule=data_module)`: This command starts the training process. The `Trainer` will:
    *   Run the training loop for `max_epochs`.
    *   Call `training_step` and `validation_step` from the `model`.
    *   Invoke callbacks at appropriate times.
    *   Log metrics.
*   After training, the path to the best saved model checkpoint is printed.

## 8. Etape 7: Evaluation

### 8.1. Loading the Best Model
*   The path to the best model checkpoint (saved by `ModelCheckpoint`) is retrieved.
*   `trained_model = LSTMClassifier.load_from_checkpoint(best_model_path)`: The best model is loaded from the checkpoint file.
*   `trained_model.eval()`: Sets the model to evaluation mode (disables dropout, etc.).

### 8.2. Plotting Training Curves
*   **`plot_training_curves(log_dir)` function:**
    *   Reads `metrics.csv` generated by `CSVLogger`.
    *   Uses `matplotlib` to plot:
        *   Training loss and validation loss vs. epoch.
        *   Training accuracy, validation accuracy, and validation F1 score vs. epoch.
    *   This helps visualize the model's learning progress and identify potential issues like overfitting.

### 8.3. Test Set Evaluation
*   `test_results = trainer.test(model=trained_model, datamodule=data_module, verbose=False)`:
    *   Runs the evaluation loop on the test set using the loaded best model.
    *   The `Trainer` calls the `test_step` method of the `model`.
    *   Returns a list of dictionaries containing the test metrics (e.g., `test_loss`, `test_acc`, `test_f1`).

### 8.4. Classification Report and Confusion Matrix
Since `trainer.test()` only returns aggregated metrics, a manual loop is performed to get per-sample predictions for detailed analysis:
1.  Get the `test_dataloader` from `data_module`.
2.  Initialize `all_preds` and `all_targets` lists.
3.  Move the `trained_model` to the appropriate device (CPU/GPU).
4.  Iterate through `test_loader` (with `torch.no_grad()` to disable gradient calculations):
    *   Get a batch of data `x` and true labels `y_true`.
    *   Move `x` to the device.
    *   Get model predictions (logits): `logits = trained_model(x)`.
    *   Get predicted class indices: `preds = torch.argmax(logits, dim=1)`.
    *   Append `preds` and `y_true` to the respective lists.
5.  **Classification Report:**
    *   `classification_report(all_targets, all_preds, target_names=...)` from `sklearn.metrics` is printed. This shows precision, recall, F1-score, and support for each class.
6.  **Confusion Matrix:**
    *   `display_confusion_matrix(y_true, y_pred, class_names)` function:
        *   Calculates the confusion matrix using `confusion_matrix()` from `sklearn.metrics`.
        *   Plots it using `matplotlib.pyplot.imshow()` for a visual representation of true vs. predicted classes.

## 9. Etape 8: Inference on a New Video

*   **Purpose:** To demonstrate how the trained model can be used to predict the action/sign in a single, new video.
*   **Steps:**
    1.  **Video Selection:** Randomly chooses a video from the `RAW_VIDEO_DATA_DIR`.
    2.  **Keypoint Extraction:**
        *   Opens the video with `cv2.VideoCapture`.
        *   Initializes `mp_hands.Hands` for this inference.
        *   Reads the video frame by frame. For each frame:
            *   Calls `process_frame_for_keypoints()` (using `yolo_detector` if available) to get the keypoint vector and an annotated frame.
            *   Stores the keypoint vector and the annotated frame.
    3.  **Sequence Preparation:**
        *   The list of frame keypoint vectors is converted to a NumPy array.
        *   This sequence is padded or truncated to `NUM_FRAMES_PER_VIDEO`, similar to the preprocessing step.
    4.  **Prediction:**
        *   The processed sequence is converted to a PyTorch tensor, a batch dimension is added (`unsqueeze(0)`).
        *   The tensor and the `trained_model` are moved to the correct device.
        *   The model is set to evaluation mode (`trained_model.eval()`).
        *   With `torch.no_grad()`, the model makes a prediction: `logits_inf = trained_model(sequence_tensor_inf)`.
        *   `torch.softmax` is applied to logits to get class probabilities.
        *   `torch.max` finds the class with the highest probability (predicted class) and its confidence.
        *   The predicted class index is converted to a class name using `data_module.inv_label_map`.
    5.  **Output and Visualization:**
        *   The predicted class name and confidence are printed.
        *   A few of the annotated frames (with the prediction overlaid) are displayed as HTML using `IPython.display.HTML` and base64 encoded images.

## 10. Conclusion
The notebook concludes with a simple print statement indicating its execution has finished. It successfully demonstrates an end-to-end pipeline for video action recognition using YOLO for ROI detection, MediaPipe for keypoint extraction, and an LSTM model trained with PyTorch Lightning for sequence classification.
