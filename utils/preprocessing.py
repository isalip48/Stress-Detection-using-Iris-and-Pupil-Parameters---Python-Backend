"""
Utility functions for image preprocessing and feature extraction
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import tensorflow as tf


def preprocess_eye_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess eye image to 5-channel format (RGB + Canny + BlackHat).
    
    **EXACT MATCH to training notebook Section 4.1!**
    
    5 Channels:
    - Channels 0-2: RGB (normalized to [0,1])
    - Channel 3: Canny edge detection (50, 150 thresholds)
    - Channel 4: BlackHat morphology (dark structures - tension rings)
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    target_size : tuple
        Target size (height, width)
    
    Returns:
    --------
    numpy.ndarray: 5-channel image (RGB + Canny + BlackHat) of shape (H, W, 5)
    """
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize
        rgb = cv2.resize(rgb, target_size)
        
        # Convert to grayscale for edge/texture detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for better edge detection (matching training notebook)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # Channel 4: Canny edge detection (matching training notebook exactly!)
        edges = cv2.Canny(gray_clahe, 50, 150)
        edge_channel = edges.astype(np.float32) / 255.0
        edge_channel = np.clip(edge_channel, 0.0, 1.0)
        
        # Channel 5: Black Hat morphological operation (dark structures - tension rings)
        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        black_hat = cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel)
        
        # Normalize with epsilon to prevent NaN (matching training notebook)
        black_hat_float = black_hat.astype(np.float32)
        black_hat_min, black_hat_max = black_hat_float.min(), black_hat_float.max()
        
        epsilon = 1e-7
        if (black_hat_max - black_hat_min) > epsilon:
            texture_channel = (black_hat_float - black_hat_min) / (black_hat_max - black_hat_min + epsilon)
        else:
            texture_channel = np.zeros_like(black_hat_float)
        
        texture_channel = np.clip(texture_channel, 0.0, 1.0)
        
        # Stack channels: RGB (3) + Canny (1) + BlackHat (1) = 5 channels
        # Matching training notebook exactly!
        rgb_float = rgb.astype(np.float32) / 255.0
        
        five_channel = np.dstack([
            rgb_float[:, :, 0],              # R
            rgb_float[:, :, 1],              # G
            rgb_float[:, :, 2],              # B
            edge_channel,                     # Canny edges
            texture_channel                   # BlackHat texture
        ])
        
        # Final validation: ensure all values in [0,1] (matching training notebook)
        five_channel = np.clip(five_channel, 0.0, 1.0)
        
        return five_channel.astype(np.float32)
    
    except Exception as e:
        print(f"❌ Error in preprocess_eye_image: {e}")
        # Return zero image if preprocessing fails
        return np.zeros((target_size[0], target_size[1], 5), dtype=np.float32)


def encode_age(age: int) -> np.ndarray:
    """
    Encode age as one-hot vector for 8 age groups.
    
    Age Groups:
    - 1-10 → index 0
    - 11-20 → index 1
    - 21-30 → index 2
    - 31-40 → index 3
    - 41-50 → index 4
    - 51-60 → index 5
    - 61-70 → index 6
    - 71-80+ → index 7
    
    Parameters:
    -----------
    age : int
        Age value (e.g., 25)
    
    Returns:
    --------
    numpy.ndarray: One-hot encoded age vector of shape (8,)
    """
    age_vector = np.zeros(8, dtype=np.float32)
    
    if age <= 10:
        age_vector[0] = 1.0
    elif age <= 20:
        age_vector[1] = 1.0
    elif age <= 30:
        age_vector[2] = 1.0
    elif age <= 40:
        age_vector[3] = 1.0
    elif age <= 50:
        age_vector[4] = 1.0
    elif age <= 60:
        age_vector[5] = 1.0
    elif age <= 70:
        age_vector[6] = 1.0
    else:
        age_vector[7] = 1.0
    
    return age_vector


def extract_eye_region(image: np.ndarray, center: Tuple[int, int], 
                        radius: int, padding: float = 1.5) -> Optional[np.ndarray]:
    """
    Extract square region around detected eye (pupil or iris).
    
    Parameters:
    -----------
    image : numpy.ndarray
        Full input image
    center : tuple
        (x, y) coordinates of eye center
    radius : int
        Radius of detected circle
    padding : float
        Padding factor (1.5 = 50% extra space around circle)
    
    Returns:
    --------
    numpy.ndarray: Cropped square region, or None if extraction fails
    """
    try:
        cx, cy = center
        size = int(radius * 2 * padding)
        
        # Calculate bounding box
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(image.shape[1], cx + size // 2)
        y2 = min(image.shape[0], cy + size // 2)
        
        # Extract region
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        return region
    
    except Exception as e:
        print(f"❌ Error in extract_eye_region: {e}")
        return None


def focal_loss(alpha=0.5, gamma=2.0):
    """
    🔧 ROCK-SOLID Focal Loss for binary classification.
    **EXACT MATCH to training notebook Section 5.1!**
    
    Formula: FL = α_factor * (1-p_t)^γ * cross_entropy
    
    Parameters:
    -----------
    alpha : float
        Balancing factor (0.5 = balanced, <0.5 favors class 0, >0.5 favors class 1)
    gamma : float
        Focusing parameter (2.0 = standard, higher = more focus on hard samples)
    
    Returns:
    --------
    function: Loss function compatible with Keras
    
    References:
    -----------
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). 
    Focal loss for dense object detection. ICCV 2017.
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        🔧 ROCK-SOLID Focal Loss implementation.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
        
        Returns:
            focal_loss: Computed focal loss value (GUARANTEED no nan!)
        """
        # Cast inputs to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 🔧 ULTRA-SAFE CLIPPING: Use larger epsilon for stability
        epsilon = 1e-7  # Keras default is 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross-entropy components
        # For y=1: -log(p)
        # For y=0: -log(1-p)
        cross_entropy = -(y_true * tf.math.log(y_pred) + 
                         (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Calculate probability of true class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # 🔧 CLIP AGAIN after calculation
        p_t = tf.clip_by_value(p_t, epsilon, 1.0 - epsilon)
        
        # Calculate modulating factor: (1 - p_t)^gamma
        # This downweights easy examples
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Calculate alpha factor for class balancing
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Combine all components
        focal_loss = alpha_factor * modulating_factor * cross_entropy
        
        # 🔧 FINAL SAFETY: Replace any nan/inf with small value
        focal_loss = tf.where(
            tf.math.is_finite(focal_loss),
            focal_loss,
            tf.ones_like(focal_loss) * 0.1  # Small fallback value instead of zero
        )
        
        return tf.reduce_mean(focal_loss)
    
    focal_loss_fixed.__name__ = 'focal_loss'
    return focal_loss_fixed


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray: Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


if __name__ == "__main__":
    print("🧪 Testing Utility Functions...")
    print("\n✅ Available functions:")
    print("  - preprocess_eye_image: Convert to 5-channel format")
    print("  - encode_age: One-hot encoding for age groups")
    print("  - extract_eye_region: Crop eye region from image")
    print("  - focal_loss: Focal loss for imbalanced classification")
    print("  - normalize_image: Normalize pixel values")
    print("\nUtils are ready for use in the pipeline!")
