#!/usr/bin/env python3
"""
Sign Language Recognition Web Application
Startup script that checks for required files and runs the Flask app
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "yolov8n.pt",
        "data-yolo/model.h5"
    ]
    
    # Check for model files
    model_files = [
        "model_yolo/best-action-model-epoch=31-val_f1=0.92.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88-v3.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88-v2.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88-v1.ckpt",
        "model_yolo/best-action-model-epoch=13-val_f1=0.88.ckpt"
    ]
    
    print("Checking required files...")
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    # Check for at least one model file
    model_found = False
    for model_path in model_files:
        if Path(model_path).exists():
            print(f"✅ Found model: {model_path}")
            model_found = True
            break
    
    if not model_found:
        missing_files.append("Any trained model checkpoint")
        print("❌ Missing: Any trained model checkpoint")
    
    if missing_files:
        print("\n❌ Some required files are missing!")
        print("Please ensure you have:")
        print("1. yolov8n.pt (YOLO model)")
        print("2. data-yolo/model.h5 (processed dataset)")
        print("3. At least one trained model checkpoint in model_yolo/")
        print("\nIf you haven't trained the model yet, run the notebook first.")
        return False
    
    print("\n✅ All required files found!")
    return True

def main():
    """Main function to run the application"""
    print("=" * 50)
    print("Sign Language Recognition Web Application")
    print("=" * 50)
    
    # Check if requirements are met
    if not check_requirements():
        print("\nExiting due to missing files.")
        sys.exit(1)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found!")
        print("Please ensure app.py is in the current directory.")
        sys.exit(1)
    
    print("\n🚀 Starting Flask application...")
    print("The app will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 