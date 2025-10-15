"""
Model loading utilities for Keras 3 models (Grayscale Hybrid CNN + Tiny Transformer)
Fully compatible with your new training script and PositionEmbedding class
"""

import os
import logging
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# -----------------------------
# Custom Position Embedding Layer
# -----------------------------
@keras.saving.register_keras_serializable(package="Custom")
class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen=10000, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = self.pos_emb(positions)
        return x + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config


# -----------------------------
# Transformer Encoder (same as training)
# -----------------------------
def transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.2):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Add()([x, inputs])
    x2 = layers.LayerNormalization(epsilon=1e-6)(x)
    x2 = layers.Dense(ff_dim, activation="relu")(x2)
    x2 = layers.Dropout(dropout)(x2)
    x2 = layers.Dense(inputs.shape[-1])(x2)
    return layers.Add()([x, x2])


# -----------------------------
# Model Loader Class
# -----------------------------
class ModelLoader:
    """Handles safe loading and validation of the grayscale hybrid model"""

    def __init__(self):
        self.model = None
        self.model_path = None

    def load_model(self, model_path):
        """Load the Keras model with custom transformer and position embedding layers"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = keras.models.load_model(
                model_path,
                custom_objects={
                    "PositionEmbedding": PositionEmbedding,
                    "transformer_encoder": transformer_encoder
                }
            )

            # Validate
            self._validate_model(model)
            self.model = model
            self.model_path = model_path
            logger.info("✅ Grayscale hybrid model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            self.model = None
            raise e

    def _validate_model(self, model):
        """Run a dummy prediction to confirm model integrity"""
        try:
            if not hasattr(model, "predict"):
                raise Exception("Model missing predict() method")

            dummy_shape = (1,) + model.input_shape[1:]
            dummy_input = np.random.random(dummy_shape).astype(np.float32)
            dummy_output = model.predict(dummy_input, verbose=0)
            logger.info(f"✅ Model validation successful — output shape: {dummy_output.shape}")

        except Exception as e:
            logger.error(f"❌ Model validation failed: {str(e)}")
            raise Exception(f"Model validation failed: {str(e)}")

    def get_model_summary(self):
        if self.model is None:
            return "No model loaded"
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        return buffer.getvalue()

    def convert_to_tflite(self, output_path=None):
        """Convert the loaded model to TensorFlow Lite"""
        if self.model is None:
            raise Exception("No model loaded")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            if output_path is None:
                output_path = "gray_hybrid_model_mobile.tflite"
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            logger.info(f"✅ TFLite model saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ TFLite conversion failed: {str(e)}")
            raise
