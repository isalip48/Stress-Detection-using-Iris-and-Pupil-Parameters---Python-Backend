"""
Detection modules for eye region analysis.

Supports:
- Grayscale detection: For Pupil dataset images (grayscale_eye.py)
- Color detection: For Iris Normal/Stressed dataset images (color_eye.py)
- Ring counting: For iris tension pattern analysis (ring_counter.py)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# Grayscale detection (for grayscale pupil images)
from .grayscale_eye import detect_pupil_robust, detect_iris_robust

# Color detection (for color iris images) - HYBRID approach
from .color_eye import detect_pupil_hybrid, detect_iris_hybrid

# Ring counting and iris unwrapping
from .ring_counter import (
    unwrap_iris_region,
    detect_tension_rings_radial_profile,
    count_tension_rings as _count_tension_rings_notebook
)


def count_tension_rings(image: np.ndarray,
                       pupil_center: Tuple[int, int],
                       pupil_radius: int,
                       iris_center: Tuple[int, int],  # For backwards compatibility (unused)
                       iris_radius: int) -> int:
    """
    Wrapper for count_tension_rings to match pipeline signature.
    Note: iris_center parameter is kept for backwards compatibility but not used.
    The notebook version uses only pupil_center as origin.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR or grayscale)
    pupil_center : tuple
        (x, y) coordinates of pupil center
    pupil_radius : int
        Pupil radius in pixels
    iris_center : tuple
        (x, y) coordinates of iris center (unused, for backwards compatibility)
    iris_radius : int
        Iris radius in pixels
    
    Returns:
    --------
    int: Number of detected tension rings
    """
    # Call the notebook version (which returns tuple)
    ring_count, _, _ = _count_tension_rings_notebook(image, pupil_center, pupil_radius, iris_radius)
    return ring_count


def detect_eye_color(image_path: str, brown_iris_mode: bool = False) -> Dict:
    """
    High-level wrapper for color eye detection.
    Uses hybrid detection approach combining Normal + Stressed notebook methods.
    
    Parameters:
    -----------
    image_path : str
        Path to color eye image
    brown_iris_mode : bool
        If True, uses brown iris detection mode (area filtering)
    
    Returns:
    --------
    dict: {
        'success': bool,
        'pupil': ((x, y), radius),
        'iris': ((x, y), radius),
        'image': numpy array,
        'error': str (if failed)
    }
    """
    try:
        # Load image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f"Failed to load image: {image_path}"
            }
        
        # Detect pupil (returns tuple: (center, radius))
        pupil_center, pupil_radius = detect_pupil_hybrid(image_path, brown_iris_mode=brown_iris_mode)
        
        if pupil_center is None or pupil_radius is None:
            return {
                'success': False,
                'error': "Pupil detection failed"
            }
        
        # Detect iris (returns tuple: (center, radius))
        iris_center, iris_radius = detect_iris_hybrid(image, pupil_center, pupil_radius)
        
        if iris_center is None or iris_radius is None:
            return {
                'success': False,
                'error': "Iris detection failed"
            }
        
        return {
            'success': True,
            'pupil': (pupil_center, pupil_radius),
            'iris': (iris_center, iris_radius),
            'image': image
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def detect_eye_grayscale(image_path: str, config: dict) -> Dict:
    """
    High-level wrapper for grayscale eye detection.
    Uses TIERED FALLBACK strategy from Pupil dataset notebook.
    
    Tries configs in order:
    1. JACKPOT (strict) - passed as 'config' parameter
    2. HIGH_PASS (relaxed) - if pupil/iris fails
    3. RESCUE (lenient) - last resort
    
    Parameters:
    -----------
    image_path : str
        Path to grayscale eye image
    config : dict
        Primary configuration (usually JACKPOT_CONFIG)
    
    Returns:
    --------
    dict: {
        'success': bool,
        'pupil': (x, y, radius),
        'iris': (x, y, radius),
        'image': numpy array,
        'error': str (if failed),
        'config_used': str (which tier succeeded)
    }
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f"Failed to load image: {image_path}"
            }
        
        # Import tier configs (avoid circular import)
        from config import HIGH_PASS_CONFIG, RESCUE_CONFIG
        
        # TIER 1: Try primary config (JACKPOT)
        print(f"🔍 Trying TIER 1 (JACKPOT)...")
        pupil_x, pupil_y, pupil_radius = detect_pupil_robust(image, config)
        iris_x, iris_y, iris_radius = detect_iris_robust(image, config)
        
        config_used = "JACKPOT"
        
        # TIER 2: If pupil failed, try HIGH_PASS
        if pupil_x is None:
            print(f"   ⚠️  Pupil failed, trying TIER 2 (HIGH_PASS)...")
            pupil_x, pupil_y, pupil_radius = detect_pupil_robust(image, HIGH_PASS_CONFIG)
            if pupil_x is not None:
                config_used = "HIGH_PASS"
                print(f"   ✅ Pupil rescued with HIGH_PASS!")
        
        # TIER 2: If iris failed, try HIGH_PASS
        if iris_x is None:
            print(f"   ⚠️  Iris failed, trying TIER 2 (HIGH_PASS)...")
            iris_x, iris_y, iris_radius = detect_iris_robust(image, HIGH_PASS_CONFIG)
            if iris_x is not None:
                config_used = "HIGH_PASS"
                print(f"   ✅ Iris rescued with HIGH_PASS!")
        
        # TIER 3: If still failing, try RESCUE (last resort)
        if pupil_x is None:
            print(f"   ⚠️  Pupil still failing, trying TIER 3 (RESCUE)...")
            pupil_x, pupil_y, pupil_radius = detect_pupil_robust(image, RESCUE_CONFIG)
            if pupil_x is not None:
                config_used = "RESCUE"
                print(f"   ✅ Pupil rescued with RESCUE!")
        
        if iris_x is None:
            print(f"   ⚠️  Iris still failing, trying TIER 3 (RESCUE)...")
            iris_x, iris_y, iris_radius = detect_iris_robust(image, RESCUE_CONFIG)
            if iris_x is not None:
                config_used = "RESCUE"
                print(f"   ✅ Iris rescued with RESCUE!")

        # Final validation
        if pupil_x is None:
            return {
                'success': False,
                'error': "Pupil detection failed (all tiers exhausted)"
            }

        if iris_x is None:
            return {
                'success': False,
                'error': "Iris detection failed (all tiers exhausted)"
            }
        
        print(f"   🎯 Detection successful using {config_used} config!")
        
        return {
            'success': True,
            'pupil': (pupil_x, pupil_y, pupil_radius),
            'iris': (iris_x, iris_y, iris_radius),
            'image': image,
            'config_used': config_used
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


__all__ = [
    # Grayscale detection (Pupil dataset)
    'detect_pupil_robust',
    'detect_iris_robust',
    
    # Color detection (Iris dataset) - HYBRID
    'detect_pupil_hybrid',
    'detect_iris_hybrid',
    
    # Ring counting and unwrapping
    'unwrap_iris_region',
    'detect_tension_rings_radial_profile',
    'count_tension_rings',
    
    # High-level wrappers for pipeline
    'detect_eye_color',
    'detect_eye_grayscale'
]

