#!/usr/bin/env python3
"""
Test script to verify prediction works correctly
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import lightning as L
import torchmetrics
import json
import h5py
import numpy as np

print("=== Prediction Test Script ===")

# LSTMClassifier class (same as in app.py)
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

def test_prediction():
    """Test the prediction function"""
    
    print("\n1. Loading model...")
    model_path = Path("./model_yolo/best-action-model-epoch=31-val_f1=0.92.ckpt")
    if not model_path.exists():
        print("❌ Model file not found!")
        return False
    
    try:
        model = LSTMClassifier.load_from_checkpoint(str(model_path))
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    print("\n2. Loading data module...")
    hdf5_file_path = Path("./data-yolo/model.h5")
    if not hdf5_file_path.exists():
        print("❌ HDF5 file not found!")
        return False
    
    try:
        with h5py.File(hdf5_file_path, 'r') as hf:
            data_module = type('DataModule', (), {})()
            data_module.label_map = json.loads(hf.attrs['label_map'])
            data_module.num_classes = len(data_module.label_map)
            data_module.inv_label_map = {v: k for k, v in data_module.label_map.items()}
        print("✅ Data module loaded successfully!")
        print(f"Classes: {data_module.label_map}")
    except Exception as e:
        print(f"❌ Error loading data module: {e}")
        return False
    
    print("\n3. Testing prediction with dummy data...")
    try:
        # Create dummy sequence (30 frames, 84 features each)
        dummy_sequence = np.random.randn(30, 84).astype(np.float32)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(dummy_sequence).unsqueeze(0)  # Add batch dimension
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model.to(device)
        sequence_tensor = sequence_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(sequence_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class name
        class_name = data_module.inv_label_map.get(predicted_class, f"Class_{predicted_class}")
        
        # Get all class probabilities
        all_probabilities = {}
        for i in range(data_module.num_classes):
            class_name_i = data_module.inv_label_map.get(i, f"Class_{i}")
            all_probabilities[class_name_i] = probabilities[0][i].item()
        
        print(f"✅ Prediction successful!")
        print(f"Predicted class: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities: {all_probabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n🎉 Prediction test PASSED!")
        print("The web app should work correctly now!")
    else:
        print("\n❌ Prediction test FAILED!") 