# ===========================================
# Medical AI System - Alzheimer's Stage Prediction API
# With MRI Image Validation Pre-Filter
# ===========================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import base64
import hashlib
import logging
from datetime import datetime
import tensorflow as tf

# Import custom modules
from model_loader import ModelLoader
from cloud_storage import CloudStorageManager
from blockchain import BlockchainManager
from utils import preprocess_image, validate_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize components
model_loader = ModelLoader()
cloud_storage = CloudStorageManager()
blockchain_manager = BlockchainManager()

# Global models
model = None
mri_filter = None

# ======================================================
# ✅ Auto-load models when running on Render / Gunicorn
# ======================================================
if os.getenv("RENDER", "false").lower() == "true" or "gunicorn" in os.environ.get("SERVER_SOFTWARE", "").lower():
    try:
        logger.info("🚀 Detected Render or Gunicorn environment — loading models at startup...")
        # Small delay optional (Render may still be initializing environment)
        from time import sleep
        sleep(2)
        load_model_on_startup()
        logger.info("✅ Models auto-loaded successfully on Render")
    except Exception as e:
        logger.error(f"❌ Auto model loading failed: {str(e)}")


# Classes for Alzheimer model (3-class grayscale)
ALZHEIMER_STAGES = ['Impaired', 'No Impairment', 'Very Mild Impairment']


# ======================================================
# Load both main model and MRI filter model
# ======================================================
def load_model_on_startup():
    global model, mri_filter
    try:
        # 1️⃣ Load Alzheimer model
        model_path = "./gray_hybrid_model.keras"
        model = model_loader.load_model(model_path)
        logger.info("✅ Alzheimer model loaded successfully")

        # 2️⃣ Load MRI filter model (optional)
        mri_filter_path = "./mri_filter_model.keras"
        if os.path.exists(mri_filter_path):
            mri_filter = tf.keras.models.load_model(mri_filter_path)
            logger.info("✅ MRI pre-filter model loaded successfully")
        else:
            logger.warning("⚠️ MRI filter model not found — skipping MRI validation")

        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return False


# ======================================================
# Routes
# ======================================================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mri_filter_loaded': mri_filter is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict_alzheimer_stage():
    try:
        # Validate request
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({'error': 'No image provided'}), 400

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Load image (supports multipart and base64)
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        else:
            image_data = request.json['image_data']
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Validate basic image format
        if not validate_image(image):
            return jsonify({'error': 'Invalid image format or size'}), 400

        # Preprocess for both filters
        processed_image = preprocess_image(image)  # shape: (1,224,224,1)

        # ======================================================
        # Step 1️⃣ — Run MRI Filter check
        # ======================================================
        if mri_filter is not None:
            mri_conf = float(mri_filter.predict(processed_image, verbose=0)[0][0])
            logger.info(f"🧩 MRI Filter Confidence: {mri_conf:.3f}")
            if mri_conf < 0.4:  # adjust threshold as needed
                logger.warning("🚫 Non-MRI detected, rejecting prediction")
                return jsonify({'error': 'Please upload a valid brain MRI scan'}), 400

        # ======================================================
        # Step 2️⃣ — Run Alzheimer Prediction
        # ======================================================
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence_scores = prediction[0]

        # Compute image hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        result = {
            'predicted_stage': ALZHEIMER_STAGES[predicted_class],
            'predicted_stage_index': predicted_class,
            'confidence_score': float(confidence_scores[predicted_class]),
            'all_probabilities': {
                stage: float(score) for stage, score in zip(ALZHEIMER_STAGES, confidence_scores)
            },
            'image_hash': image_hash,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'densenet_tinytransformer_gray_v2'
        }

        # Optional cloud + blockchain storage
        try:
            storage_url = cloud_storage.store_image(image_bytes, image_hash)
            result['storage_url'] = storage_url
        except Exception as e:
            logger.warning(f"Cloud storage failed: {str(e)}")
            result['storage_url'] = None

        try:
            tx_hash = blockchain_manager.store_prediction(image_hash, result)
            result['blockchain_tx'] = tx_hash
        except Exception as e:
            logger.warning(f"Blockchain storage failed: {str(e)}")
            result['blockchain_tx'] = None

        logger.info(f"Prediction successful: {result['predicted_stage']} ({result['confidence_score']:.2f})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/model/info', methods=['GET'])
def get_model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        return jsonify({
            'model_type': 'DenseNet + Tiny Transformer',
            'framework': 'Keras 3',
            'input_shape': list(model.input_shape[1:]) if hasattr(model, 'input_shape') else 'Unknown',
            'output_classes': ALZHEIMER_STAGES,
            'total_params': model.count_params() if hasattr(model, 'count_params') else 'Unknown',
            'mri_filter_loaded': mri_filter is not None,
            'model_loaded': True
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


# ======================================================
# Main entry
# ======================================================
if __name__ == '__main__':
    if load_model_on_startup():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Cannot start server - model loading failed")
