"""
Configuration file for the Stress Detection Testing Pipeline
Author: AI Assistant
Date: 2025-11-11
"""

import os

# ============================================================================
# MODEL PATHS
# ============================================================================
# Production model - the best trained model (99.91% AUC-PR)
MODEL_PATH = os.path.join("Model", "best_dual_stream_age_aware_model.keras")
MODEL_NAME = "Dual-Stream Age-Aware Stress Detection Model"

# Model was trained with optimized 70-20-10 stratified split
# Training config: Focal Loss (α=0.5, γ=2.0), Warmup LR, 2x iris aug, 1.5x pupil aug

# ============================================================================
# IMAGE PROCESSING SETTINGS
# ============================================================================
TARGET_SIZE = (224, 224)  # Model input size
PUPIL_MM_PER_PIXEL = 0.0117  # Conversion factor (from pupil notebook)
AVG_IRIS_DIAMETER_MM = 12.0  # Average iris diameter in mm

# ============================================================================
# AGE ENCODING
# ============================================================================
AGE_GROUPS = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80"]
AGE_GROUP_LABELS = {
    "1-10": 0,
    "11-20": 1,
    "21-30": 2,
    "31-40": 3,
    "41-50": 4,
    "51-60": 5,
    "61-70": 6,
    "71-80": 7
}

def get_age_group(age):
    """Convert age to age group string"""
    if age <= 10:
        return "1-10"
    elif age <= 20:
        return "11-20"
    elif age <= 30:
        return "21-30"
    elif age <= 40:
        return "31-40"
    elif age <= 50:
        return "41-50"
    elif age <= 60:
        return "51-60"
    elif age <= 70:
        return "61-70"
    else:
        return "71-80"

# ============================================================================
# DETECTION CONFIGURATIONS (Tiered Strategy from Pupil Notebook)
# ============================================================================

# TIER 1: JACKPOT - High-quality images (most strict)
JACKPOT_CONFIG = {
    # Pupil detection parameters
    'INPAINT_THRESHOLD': 220,      # High threshold = only brightest reflections
    'CLAHE_CLIP_LIMIT': 2.0,       # Moderate contrast enhancement
    'BINARY_THRESHOLD': 40,        # Conservative threshold
    'MIN_PUPIL_AREA': 500,         # Larger minimum area
    'MAX_PUPIL_AREA': 50000,       # Maximum area
    'MIN_CIRCULARITY': 0.75,       # High circularity requirement
    
    # Iris detection parameters (needed by notebook grayscale functions!)
    'MIN_IRIS_RADIUS': 80,
    'MAX_IRIS_RADIUS': 250,
    'CANNY_LOW': 50,               # CRITICAL FIX - Added for grayscale notebook compatibility
    'CANNY_HIGH': 150,             # CRITICAL FIX - Added for grayscale notebook compatibility
    'IRIS_HOUGH_PARAM1': 50,       # CRITICAL FIX - Added for grayscale notebook compatibility
    'IRIS_HOUGH_PARAM2': 30,       # CRITICAL FIX - Added for grayscale notebook compatibility
    
    # Legacy Hough params (keep for backwards compatibility)
    'HOUGH_DP': 1.2,
    'HOUGH_MIN_DIST': 100,
    'HOUGH_PARAM1': 50,
    'HOUGH_PARAM2': 30
}

# TIER 2: HIGH PASS - Good quality images (relaxed)
HIGH_PASS_CONFIG = {
    # Pupil detection parameters
    'INPAINT_THRESHOLD': 200,      # Lower threshold = more reflections removed
    'CLAHE_CLIP_LIMIT': 3.0,       # More contrast enhancement
    'BINARY_THRESHOLD': 35,        # More permissive
    'MIN_PUPIL_AREA': 300,         # Smaller minimum area
    'MAX_PUPIL_AREA': 50000,
    'MIN_CIRCULARITY': 0.65,       # Lower circularity requirement
    
    # Iris detection parameters
    'MIN_IRIS_RADIUS': 60,
    'MAX_IRIS_RADIUS': 300,
    'CANNY_LOW': 40,               # CRITICAL FIX - Added for grayscale notebook compatibility
    'CANNY_HIGH': 120,             # CRITICAL FIX - Added for grayscale notebook compatibility
    'IRIS_HOUGH_PARAM1': 50,       # CRITICAL FIX - Added for grayscale notebook compatibility
    'IRIS_HOUGH_PARAM2': 25,       # CRITICAL FIX - Added for grayscale notebook compatibility
    
    # Legacy Hough params
    'HOUGH_DP': 1.2,
    'HOUGH_MIN_DIST': 80,
    'HOUGH_PARAM1': 50,
    'HOUGH_PARAM2': 25
}

# TIER 3: RESCUE - Challenging images (most lenient)
RESCUE_CONFIG = {
    # Pupil detection parameters
    'INPAINT_THRESHOLD': 180,      # Aggressive reflection removal
    'CLAHE_CLIP_LIMIT': 4.0,       # Aggressive contrast
    'BINARY_THRESHOLD': 30,        # Very permissive
    'MIN_PUPIL_AREA': 200,         # Very small minimum
    'MAX_PUPIL_AREA': 50000,
    'MIN_CIRCULARITY': 0.50,       # Low circularity (desperate)
    
    # Iris detection parameters
    'MIN_IRIS_RADIUS': 40,
    'MAX_IRIS_RADIUS': 350,
    'CANNY_LOW': 30,               # CRITICAL FIX - Added for grayscale notebook compatibility
    'CANNY_HIGH': 100,             # CRITICAL FIX - Added for grayscale notebook compatibility
    'IRIS_HOUGH_PARAM1': 30,       # CRITICAL FIX - Added for grayscale notebook compatibility
    'IRIS_HOUGH_PARAM2': 20,       # CRITICAL FIX - Added for grayscale notebook compatibility
    
    # Legacy Hough params
    'HOUGH_DP': 1.5,
    'HOUGH_MIN_DIST': 60,
    'HOUGH_PARAM1': 30,
    'HOUGH_PARAM2': 20
}

# Default config to use
DEFAULT_CONFIG = JACKPOT_CONFIG

# ============================================================================
# IRIS DETECTION SETTINGS (From Iris Notebooks)
# ============================================================================

# Color image detection settings (from Iris_Tension_Detection notebooks)
COLOR_DETECTION_SETTINGS = {
    'USE_ADAPTIVE_THRESHOLD': True,
    'MIN_CONTOUR_AREA': 1000,
    'MAX_CONTOUR_AREA': 100000,
    'GAUSSIAN_BLUR_SIZE': 5,
    'CANNY_THRESHOLD1': 50,
    'CANNY_THRESHOLD2': 150
}

# Ring counting settings (from Iris_Pattern_Unwrap notebook)
RING_COUNTING_SETTINGS = {
    'UNWRAP_HEIGHT': 64,           # Height of unwrapped iris
    'UNWRAP_WIDTH': 512,           # Width of unwrapped iris
    'RADIAL_RESOLUTION': 64,       # Radial sampling points
    'ANGULAR_RESOLUTION': 512,     # Angular sampling points
    'MIN_RING_SPACING': 5,         # Minimum pixels between rings
    'GRADIENT_THRESHOLD': 10       # Threshold for edge detection
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
VISUALIZATION_DPI = 100
SAVE_INTERMEDIATE_RESULTS = True
VERBOSE = True

# Prediction thresholds
STRESS_THRESHOLD = 0.5  # Above this = stressed, below = normal
HIGH_CONFIDENCE_THRESHOLD = 0.7  # High confidence if > 0.7 or < 0.3
LOW_CONFIDENCE_THRESHOLD = 0.3

# ============================================================================
# PATHS
# ============================================================================
INPUT_COLOR_DIR = "input/color"
INPUT_GRAYSCALE_DIR = "input/grayscale"
OUTPUT_VIZ_DIR = "output/visualizations"
OUTPUT_PRED_DIR = "output/predictions"
OUTPUT_COMP_DIR = "output/comparison"

# ============================================================================
# GUI SETTINGS
# ============================================================================
GUI_WINDOW_TITLE = "Iris Stress Detection Testing System"
GUI_WINDOW_SIZE = "1400x900"
GUI_THEME = "default"  # Options: 'default', 'clam', 'alt', 'classic'
