"""
Utility functions for preprocessing and validating grayscale MRI images
"""

import logging
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

logger = logging.getLogger(__name__)

def validate_image(image):
    """
    Validate image format and size.
    Accept grayscale (L) and RGB images, but prefer grayscale (L).
    """
    if image is None:
        logger.error("❌ No image provided for validation")
        return False

    # Accept only grayscale or RGB
    if image.mode not in ["L", "RGB"]:
        logger.warning(f"Unsupported image mode: {image.mode} — converting to grayscale (L)")
        image = image.convert("L")

    # Minimum resolution check
    if image.width < 64 or image.height < 64:
        logger.warning(f"Image too small: {image.size}")
        return False

    return True


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for the grayscale hybrid model.
    - Converts to grayscale (L)
    - Resizes to target_size
    - Normalizes pixel values to [0, 1]
    - Returns array shape: (1, target_size[0], target_size[1], 1)
    """
    try:
        # Ensure grayscale mode
        if image.mode != "L":
            image = image.convert("L")

        # Resize to model input size
        image = image.resize(target_size)

        # Convert to numpy array
        img_array = img_to_array(image)  # shape (224, 224, 1)
        img_array = img_array.astype("float32") / 255.0  # normalize to [0,1]

        # Add batch dimension → (1, 224, 224, 1)
        img_array = np.expand_dims(img_array, axis=0)

        logger.info(f"✅ Grayscale image preprocessed: shape {img_array.shape}")
        return img_array

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return None
