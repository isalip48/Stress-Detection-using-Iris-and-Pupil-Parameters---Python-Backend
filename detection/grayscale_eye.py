"""
Grayscale Eye Detection Module
Extracted from Pupil_Stress_Analysis_Notebook.ipynb

This module contains detection functions for pupil and iris in grayscale images.
These are exact 1:1 copies from the notebook for consistent detection behavior.
"""

import cv2
import numpy as np


def detect_pupil_robust(image_cv, config):
    """
    Detect pupil in an eye image using robust computer vision techniques.
    NOW CONFIGURABLE: Accepts a config dictionary for tiered detection strategy.
    
    Parameters:
    -----------
    image_cv : numpy.ndarray
        Input image in BGR format (loaded with cv2.imread)
    config : dict
        Configuration dictionary containing detection parameters:
        - INPAINT_THRESHOLD, CLAHE_CLIP_LIMIT, BINARY_THRESHOLD
        - MIN_PUPIL_AREA, MAX_PUPIL_AREA, MIN_CIRCULARITY
    
    Returns:
    --------
    tuple: (center_x, center_y, radius)
        - center_x, center_y: Pupil center coordinates
        - radius: Pupil radius in pixels
        Returns (None, None, None) if detection fails
    """
    try:
        # 1. PREPROCESSING PIPELINE
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply Median Blur to reduce noise
        blurred = cv2.medianBlur(gray, 5)
        
        # 2. REFLECTION REMOVAL (using config)
        # Detect bright spots (reflections) using threshold from config
        _, reflection_mask = cv2.threshold(
            blurred, 
            config['INPAINT_THRESHOLD'], 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Inpaint the reflections to fill them with surrounding pixels
        if np.sum(reflection_mask) > 0:
            inpainted = cv2.inpaint(blurred, reflection_mask, 3, cv2.INPAINT_TELEA)
        else:
            inpainted = blurred.copy()
        
        # 3. CONTRAST ENHANCEMENT using CLAHE (using config)
        clahe = cv2.createCLAHE(
            clipLimit=config['CLAHE_CLIP_LIMIT'], 
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(inpainted)
        
        # 4. THRESHOLDING (using config)
        _, binary = cv2.threshold(
            enhanced, 
            config['BINARY_THRESHOLD'], 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Optional: Use morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 5. CONTOUR FILTERING
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return (None, None, None)
        
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (using config)
            if area < config['MIN_PUPIL_AREA'] or area > config['MAX_PUPIL_AREA']:
                continue
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # Calculate circularity: 4π * area / perimeter²
            # Perfect circle = 1.0, less circular shapes < 1.0
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Check if circularity meets threshold (using config)
            if circularity > config['MIN_CIRCULARITY'] and circularity > best_circularity:
                best_circularity = circularity
                best_contour = contour
        
        # 6. GET FINAL PUPIL MEASUREMENTS
        if best_contour is not None:
            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            
            # Verify radius is reasonable
            if radius < 5 or radius > 200:
                return (None, None, None)
            
            return (int(x), int(y), int(radius))
        else:
            return (None, None, None)
    
    except Exception as e:
        # Silently fail - tiered detection will try fallback configs
        return (None, None, None)


def detect_iris_robust(image_cv, config):
    """
    Detect the iris boundary using Hough Circle Transform.
    NOW CONFIGURABLE: Accepts a config dictionary for tiered detection strategy.
    
    The iris is the colored ring around the pupil. We detect it to:
    1. Establish a reference size for pixel-to-millimeter conversion
    2. The average human iris diameter is ~12mm (scientific standard)
    
    Parameters:
    -----------
    image_cv : numpy.ndarray
        Input image in BGR format (loaded with cv2.imread)
    config : dict
        Configuration dictionary containing detection parameters:
        - CANNY_LOW, CANNY_HIGH, IRIS_HOUGH_PARAM1, IRIS_HOUGH_PARAM2
        - MIN_IRIS_RADIUS, MAX_IRIS_RADIUS (FIXED: matches config keys)
    
    Returns:
    --------
    tuple: (center_x, center_y, radius) or (None, None, None) if detection fails
    """
    try:
        # 1. Preprocessing
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur to reduce noise (iris needs more smoothing)
        blurred = cv2.medianBlur(gray, 7)
        
        # 2. Edge Detection using Canny (using config)
        edges = cv2.Canny(
            blurred, 
            config['CANNY_LOW'], 
            config['CANNY_HIGH']
        )
        
        # 3. Hough Circle Transform (using config)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=config['IRIS_HOUGH_PARAM1'],
            param2=config['IRIS_HOUGH_PARAM2'],
            minRadius=config['MIN_IRIS_RADIUS'],  # FIXED: was IRIS_MIN_RADIUS
            maxRadius=config['MAX_IRIS_RADIUS']   # FIXED: was IRIS_MAX_RADIUS
        )
        
        # 4. Process detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Get the largest circle (most likely to be the iris)
            largest_circle = None
            max_radius = 0
            
            for circle in circles[0, :]:
                x, y, radius = circle
                if radius > max_radius:
                    max_radius = radius
                    largest_circle = circle
            
            if largest_circle is not None:
                x, y, r = largest_circle
                return (int(x), int(y), int(r))
        
        return (None, None, None)
    
    except Exception as e:
        # Silently fail - tiered detection will try fallback configs
        return (None, None, None)
