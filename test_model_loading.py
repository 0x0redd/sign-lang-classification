#!/usr/bin/env python3
"""
Test script to debug model loading issues
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

print("=== Model Loading Debug Script ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Lightning version: {L.__version__}")

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

def test_model_loading():
    """Test loading the model step by step"""
    
    print("\n1. Checking model files...")
    model_files = [
        "model_yolo/best-action-model-epoch=31-val_f1=0.92.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88-v3.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88-v2.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88-v1.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88.ckpt"
    ]
    
    model_path = None
    for file_path in model_files:
        if Path(file_path).exists():
            print(f"✅ Found model: {file_path}")
            model_path = Path(file_path)
            break
        else:
            print(f"❌ Not found: {file_path}")
    
    if not model_path:
        print("❌ No model files found!")
        return False
    
    print(f"\n2. Checking model file size...")
    file_size = model_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Model file size: {file_size:.2f} MB")
    
    print(f"\n3. Checking HDF5 file...")
    hdf5_file_path = Path("./data-yolo/model.h5")
    if hdf5_file_path.exists():
        print(f"✅ HDF5 file found: {hdf5_file_path}")
        try:
            with h5py.File(hdf5_file_path, 'r') as hf:
                label_map = json.loads(hf.attrs['label_map'])
                print(f"✅ Label map loaded: {label_map}")
                print(f"Number of classes: {len(label_map)}")
        except Exception as e:
            print(f"❌ Error reading HDF5 file: {e}")
            return False
    else:
        print(f"❌ HDF5 file not found: {hdf5_file_path}")
        return False
    
    print(f"\n4. Attempting to load model...")
    try:
        print(f"Loading model from: {model_path}")
        model = LSTMClassifier.load_from_checkpoint(str(model_path))
        print("✅ Model loaded successfully!")
        
        # Test model
        model.eval()
        print("✅ Model set to evaluation mode")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 30, 84)  # batch_size=1, seq_len=30, features=84
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
            print(f"Output: {output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test PASSED!")
    else:
        print("\n❌ Model loading test FAILED!") 