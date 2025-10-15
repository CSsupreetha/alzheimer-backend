#!/usr/bin/env python3
"""
Test script to validate model loading and basic functionality
Updated for grayscale 3-class hybrid model with threshold-based inference
"""

import sys, os, io, numpy as np
from PIL import Image
import logging

# Add backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_loader import ModelLoader
from utils import preprocess_image, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# 1) Test Model Loading
# ============================
def test_model_loading():
    print("\n=== Testing Model Loading ===")
    try:
        model_loader = ModelLoader()
        model_path = "../attached_assets/gray_hybrid_model.keras"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        model = model_loader.load_model(model_path)
        print("✅ Model loaded successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False


# ============================
# 2) Test Image Processing
# ============================
def test_image_processing():
    print("\n=== Testing Image Preprocessing ===")
    try:
        # Create dummy grayscale image (1-channel)
        dummy_image = Image.new('L', (256, 256), color=128)
        print(f"Dummy image mode: {dummy_image.mode}")

        # Validate image
        if not validate_image(dummy_image):
            print("❌ Image validation failed")
            return False
        print("✅ Image validation passed")

        # Preprocess (returns grayscale single-channel input)
        processed = preprocess_image(dummy_image, target_size=(224, 224))
        print(f"Processed image shape: {processed.shape} (should be (1, 224, 224, 1))")
        return True
    except Exception as e:
        print(f"❌ Image preprocessing failed: {str(e)}")
        return False


# ============================
# 3) Test Model Inference
# ============================
def test_model_inference():
    print("\n=== Testing Model Inference ===")
    try:
        model_loader = ModelLoader()
        model_path = "../attached_assets/gray_hybrid_model.keras"
        model = model_loader.load_model(model_path)

        # 3-class setup from new model
        classes = ['Impaired', 'No Impairment', 'Very Mild Impairment']
        threshold = 0.5

        # Create dummy grayscale image
        dummy_image = Image.new('L', (256, 256), color=128)
        processed = preprocess_image(dummy_image, target_size=(224, 224))

        # Run model prediction
        preds = model.predict(processed, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])

        # Apply threshold logic
        if top_conf < threshold:
            label = "Uncertain"
        else:
            label = classes[top_idx]

        print(f"Predicted class: {label}")
        print(f"Confidence: {top_conf:.3f}")
        print(f"Full prediction vector: {np.round(preds, 3)}")

        return True
    except Exception as e:
        print(f"❌ Model inference failed: {str(e)}")
        return False


# ============================
# 4) Test TFLite Conversion
# ============================
def test_tflite_conversion():
    print("\n=== Testing TFLite Conversion ===")
    try:
        model_loader = ModelLoader()
        model_path = "../attached_assets/gray_hybrid_model.keras"
        model_loader.load_model(model_path)

        tflite_path = model_loader.convert_to_tflite("test_model_mobile.tflite")
        if os.path.exists(tflite_path):
            size_kb = os.path.getsize(tflite_path) / 1024
            print(f"✅ TFLite model saved: {tflite_path}, Size: {size_kb:.2f} KB")
            os.remove(tflite_path)  # cleanup
            return True
        return False
    except Exception as e:
        print(f"❌ TFLite conversion failed: {str(e)}")
        return False


# ============================
# Run All Tests
# ============================
def main():
    tests = [
        ("Model Loading", test_model_loading),
        ("Image Processing", test_image_processing),
        ("Model Inference", test_model_inference),
        ("TFLite Conversion", test_tflite_conversion),
    ]
    results = {}
    for name, func in tests:
        try:
            results[name] = func()
        except Exception as e:
            results[name] = False
            print(f"{name} exception: {str(e)}")

    print("\n=== TEST SUMMARY ===")
    passed = 0
    for name, success in results.items():
        status = "PASS ✅" if success else "FAIL ❌"
        print(f"{name:20} {status}")
        if success:
            passed += 1
    print(f"\nOverall: {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
