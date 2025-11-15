"""
Hybrid Color Eye Detection Module
Combining best approaches from Iris_Tension_Detection_Normal.ipynb and Iris_Tension_Detection_Stressed.ipynb

HYBRID STRATEGY:
- Brightness normalization from Stressed notebook (handles any lighting)
- Enhanced glint removal from Normal notebook (handles reflections)
- Adaptive HSV thresholds from Stressed notebook (bright/normal/dark)
- Area constraints from Normal notebook (2463-4300 px² for brown iris)
- Gradient-based iris from Normal notebook (works for brown iris)
- Multiple fallback strategies for maximum compatibility

Authors: Extracted from both notebooks
Date: 2025-11-12
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """
    Normalize image brightness for consistent detection across lighting conditions.
    FROM: Stressed notebook
    
    Handles:
    - Very bright/washed out images (overexposed)
    - Very dark images (underexposed)
    - Inconsistent lighting
    
    Args:
        image: BGR color image
    
    Returns:
        Brightness-normalized BGR image
    """
    # Convert to LAB color space (separates brightness from color)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return normalized


def remove_glints_enhanced(gray_image: np.ndarray) -> np.ndarray:
    """
    ENHANCED glint removal from Normal notebook.
    
    Improvements from Normal notebook:
    - Lower threshold (220 instead of 240) to catch more subtle reflections
    - Larger inpaint radius (7px instead of 3px) for better filling
    - Local contrast detection for reflections in dark regions
    - Morphological dilation of glint mask for complete coverage
    
    Args:
        gray_image: Grayscale image
    
    Returns:
        Image with glints removed
    """
    # Primary glint detection - lowered threshold
    glint_mask = (gray_image > 220).astype(np.uint8)
    
    # Additional detection: bright spots in dark regions (pupil area)
    dark_regions = (gray_image < 80).astype(np.uint8)
    local_bright = cv2.dilate(dark_regions, np.ones((15, 15), np.uint8))
    local_bright_spots = ((gray_image > 180) & (local_bright > 0)).astype(np.uint8)
    
    # Combine both glint detection methods
    glint_mask = cv2.bitwise_or(glint_mask, local_bright_spots)
    
    # Dilate glint mask to ensure complete coverage
    if np.any(glint_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glint_mask = cv2.dilate(glint_mask, kernel, iterations=1)
        
        # Inpaint with larger radius for better filling
        inpainted = cv2.inpaint(gray_image, glint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return inpainted
    
    return gray_image


# ============================================================================
# PUPIL DETECTION - HYBRID APPROACH
# ============================================================================

def segment_pupil_hybrid(image: np.ndarray, image_normalized: np.ndarray) -> np.ndarray:
    """
    HYBRID pupil segmentation combining Normal and Stressed approaches.
    
    Combines:
    - Adaptive HSV thresholds from Stressed (handles any brightness)
    - Enhanced glint removal from Normal (handles reflections)
    - Both notebook's fusion strategies
    
    Args:
        image: Original BGR image
        image_normalized: Brightness-normalized BGR image
    
    Returns:
        Binary mask of pupil region
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = remove_glints_enhanced(gray)
    
    # Enhanced preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
    
    # ========================================
    # ADAPTIVE HSV THRESHOLDING (from Stressed)
    # ========================================
    hsv = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # Analyze image brightness
    mean_v = np.mean(v_channel)
    v_percentile_10 = np.percentile(v_channel, 10)
    
    # Adaptive threshold strategy
    if mean_v > 150:  # Very bright image
        v_threshold = min(90, max(60, int(v_percentile_10 * 1.3)))
        strategy = "bright"
    elif mean_v < 80:  # Dark image
        v_threshold = min(70, max(40, int(v_percentile_10 * 1.1)))
        strategy = "dark"
    else:  # Normal brightness
        v_threshold = min(75, max(50, v_percentile_10))
        strategy = "normal"
    
    # Primary HSV mask
    lower_bound = np.array([0, 0, 0], dtype=np.uint8)
    upper_bound = np.array([180, 255, int(v_threshold)], dtype=np.uint8)
    pupil_mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # ========================================
    # ADAPTIVE GRAYSCALE THRESHOLDING
    # ========================================
    mean_intensity = np.mean(blurred)
    std_intensity = np.std(blurred)
    
    if strategy == "bright":
        threshold = np.clip(mean_intensity - 2.0 * std_intensity, 25, 65)
    elif strategy == "dark":
        threshold = np.clip(mean_intensity - 1.2 * std_intensity, 35, 75)
    else:
        threshold = np.clip(mean_intensity - 1.5 * std_intensity, 30, 70)
    
    pupil_mask_gray = (blurred < threshold).astype(np.uint8) * 255
    
    # ========================================
    # FUSION LOGIC
    # ========================================
    pupil_mask = cv2.bitwise_or(pupil_mask_hsv, pupil_mask_gray)
    
    # Check mask quality
    mask_area = np.sum(pupil_mask > 0)
    image_area = image.shape[0] * image.shape[1]
    mask_ratio = mask_area / image_area
    
    # If mask is too large, use intersection
    if mask_ratio > 0.30:
        pupil_mask = cv2.bitwise_and(pupil_mask_hsv, pupil_mask_gray)
    
    # Fallback strategies if no detection
    if np.sum(pupil_mask > 0) < 50:
        upper_bound_relaxed = np.array([180, 255, int(v_threshold * 1.5)], dtype=np.uint8)
        pupil_mask = cv2.inRange(hsv, lower_bound, upper_bound_relaxed)
    
    if np.sum(pupil_mask > 0) < 50 and strategy == "bright":
        upper_bound_relaxed2 = np.array([180, 255, int(v_threshold * 2.0)], dtype=np.uint8)
        pupil_mask = cv2.inRange(hsv, lower_bound, upper_bound_relaxed2)
    
    return pupil_mask


def refine_pupil_mask(pupil_mask: np.ndarray) -> np.ndarray:
    """
    Morphological refinement to clean binary mask.
    FROM: Both notebooks (identical implementation)
    
    Operations:
    1. Opening - Remove small noise
    2. Closing - Fill small holes
    3. Hole filling - Fill all interior holes
    4. Smoothing - Make boundaries circular
    
    Args:
        pupil_mask: Binary mask
    
    Returns:
        Refined binary mask
    """
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Flood fill large holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cv2.drawContours(mask, contours, -1, 255, -1)
    
    # Smooth boundaries
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth)
    
    return mask


def select_best_pupil_candidate_hybrid(pupil_mask: np.ndarray, 
                                        image_height: int, 
                                        image_width: int,
                                        use_brown_iris_constraints: bool = False) -> Tuple:
    """
    HYBRID candidate selection combining both approaches.
    
    Args:
        pupil_mask: Binary mask
        image_height: Image height
        image_width: Image width
        use_brown_iris_constraints: If True, use Normal notebook's area constraints (2463-4300 px²)
                                     If False, use Stressed notebook's adaptive constraints
    
    Returns:
        (contour, centroid, area, bbox) or (None, None, None, None)
    """
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None, None, None
    
    # Choose area constraints
    if use_brown_iris_constraints:
        # FROM NORMAL NOTEBOOK: Specific for brown iris (based on successful detections)
        min_area = 2463  # Subject670_L1.JPG (radius=28)
        max_area = 4300  # Subject610_R3.JPG (radius=37)
        min_circularity = 0.40
        location_weight = 1.0
    else:
        # FROM STRESSED NOTEBOOK: Adaptive for any image
        min_dim = min(image_height, image_width)
        
        if min_dim <= 250:  # Small images
            min_area, max_area = 80, 18000
        elif min_dim <= 550:  # Medium images
            min_area, max_area = 600, 70000
        else:  # Large images
            min_area = int(min_dim * min_dim * 0.0025)
            max_area = int(min_dim * min_dim * 0.20)
        
        min_circularity = 0.35
        location_weight = 0.8
    
    valid_candidates = []
    
    for contour in contours:
        # Filter 1: Size
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        # Filter 2: Circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue
        
        # Filter 3: Location
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Prefer central region
        x_min, x_max = int(image_width * 0.05), int(image_width * 0.95)
        y_min, y_max = int(image_height * 0.05), int(image_height * 0.95)
        
        in_central = x_min <= cx <= x_max and y_min <= cy <= y_max
        location_score = 1.0 if in_central else location_weight
        
        # KEY SCORING: area × circularity × location
        score = area * circularity * location_score
        
        valid_candidates.append({
            'contour': contour,
            'centroid': (cx, cy),
            'area': area,
            'circularity': circularity,
            'score': score
        })
    
    if len(valid_candidates) == 0:
        return None, None, None, None
    
    # Select best candidate (highest score)
    best = max(valid_candidates, key=lambda x: x['score'])
    x, y, w, h = cv2.boundingRect(best['contour'])
    
    return best['contour'], best['centroid'], best['area'], (x, y, w, h)


def fit_circle_to_contour(contour) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    Fit circle to contour using minimum enclosing circle.
    FROM: Both notebooks (identical)
    
    Args:
        contour: OpenCV contour
    
    Returns:
        ((center_x, center_y), radius) or (None, None)
    """
    if contour is None or len(contour) < 5:
        return None, None
    
    # Minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    
    return (int(cx), int(cy)), int(radius)


def detect_pupil_hybrid(image_path: Union[str, Path, np.ndarray],
                        brown_iris_mode: bool = False) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    HYBRID pupil detection combining best of both notebooks.
    
    Pipeline:
    1. Brightness normalization (Stressed)
    2. Enhanced glint removal (Normal)
    3. Adaptive HSV + grayscale fusion (Both)
    4. Morphological refinement (Both)
    5. Smart candidate selection (Both with mode switch)
    6. Circle fitting (Both)
    
    Args:
        image_path: Path to image OR numpy array
        brown_iris_mode: If True, use area constraints for brown iris (2463-4300 px²)
                         If False, use adaptive constraints for any image
    
    Returns:
        (center, radius) or (None, None)
        center: (x, y) tuple of pupil center
        radius: int pupil radius in pixels
    """
    # Load image
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
    else:
        img = image_path
    
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    
    # Stage 1: Brightness normalization (from Stressed)
    img_normalized = normalize_brightness(img)
    
    # Stage 2: Hybrid segmentation (combines both approaches)
    pupil_mask = segment_pupil_hybrid(img, img_normalized)
    
    # Stage 3: Morphological refinement (both notebooks)
    pupil_mask = refine_pupil_mask(pupil_mask)
    
    # Stage 4: Select best candidate (with mode switch)
    contour, centroid, area, bbox = select_best_pupil_candidate_hybrid(
        pupil_mask, h, w, use_brown_iris_constraints=brown_iris_mode
    )
    
    if contour is None:
        return None, None
    
    # Stage 5: Fit circle
    center, radius = fit_circle_to_contour(contour)
    
    return center, radius


# ============================================================================
# IRIS DETECTION - HYBRID APPROACH
# ============================================================================

def detect_iris_gradient_based(image: np.ndarray, 
                                pupil_center: Tuple[int, int], 
                                pupil_radius: int) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    Gradient-based iris detection from Normal notebook.
    PROVEN: Works well for brown iris images.
    
    Strategy:
    - Search radially outward from pupil
    - Look for strong gradient (edge) indicating iris boundary
    - Brown iris has clear boundary with sclera (white part)
    
    Args:
        image: Input image (BGR)
        pupil_center: (cx, cy) - center of detected pupil
        pupil_radius: radius of detected pupil
    
    Returns:
        (iris_center, iris_radius) or (None, None)
    """
    if pupil_center is None or pupil_radius is None:
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Calculate gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    px, py = pupil_center
    
    # Sample points in multiple directions (angles)
    num_angles = 36  # Sample every 10 degrees
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Search range for iris boundary
    min_search_radius = int(pupil_radius * 1.8)  # Start searching after pupil
    max_search_radius = int(pupil_radius * 4.5)  # Maximum expected iris size
    
    # Ensure within image bounds
    max_search_radius = min(max_search_radius, 
                           min(px, w-px, py, h-py) - 5)
    
    if min_search_radius >= max_search_radius:
        return None, None
    
    detected_radii = []
    
    # For each angle, search radially for the iris boundary
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Sample gradient along this radial line
        radii = np.arange(min_search_radius, max_search_radius, 1)
        max_grad = 0
        best_radius = None
        
        for r in radii:
            x = int(px + r * dx)
            y = int(py + r * dy)
            
            # Check bounds
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            
            # Get gradient at this point
            grad = gradient_magnitude[y, x]
            
            if grad > max_grad:
                max_grad = grad
                best_radius = r
        
        # Only accept if gradient is strong enough (clear edge)
        if best_radius is not None and max_grad > 10:
            detected_radii.append(best_radius)
    
    # Need at least 50% of radial samples to succeed
    if len(detected_radii) < num_angles * 0.5:
        return None, None
    
    # Use median radius (robust to outliers)
    iris_radius = int(np.median(detected_radii))
    
    # Validate: iris should be reasonable size relative to pupil
    if iris_radius < pupil_radius * 1.8 or iris_radius > pupil_radius * 4.5:
        return None, None
    
    # Use pupil center as iris center (they should be concentric)
    iris_center = pupil_center
    
    return iris_center, iris_radius


def detect_iris_hybrid(image: np.ndarray,
                        pupil_center: Tuple[int, int],
                        pupil_radius: int) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    HYBRID iris detection.
    Currently uses gradient-based method from Normal notebook (proven for brown iris).
    
    Args:
        image: BGR image
        pupil_center: (x, y) pupil center
        pupil_radius: pupil radius in pixels
    
    Returns:
        (iris_center, iris_radius) or (None, None)
    """
    return detect_iris_gradient_based(image, pupil_center, pupil_radius)
