"""
Pupil Diameter Measurement - Pixel to Millimeter Conversion

Extracted 1:1 from: Pupil_Stress_Analysis_Notebook.ipynb

Author: AI Assistant
Date: 2025-11-12

This module contains the EXACT measurement functions used in the notebook.
DO NOT MODIFY - These must match the notebook exactly for production deployment.
"""

from typing import Tuple


# Constants from notebook (EXACT VALUES)
AVG_IRIS_DIAMETER_MM = 12.0  # Average human iris diameter in millimeters
MIN_PUPIL_DIAMETER_MM = 1.8  # Minimum pupil constriction (bright light)
MAX_PUPIL_DIAMETER_MM = 10.0  # Maximum pupil dilation (darkness)


def measure_pupil_diameter_from_notebook(pupil_radius_px: int, 
                                          iris_radius_px: int) -> Tuple[float, float, float]:
    """
    Convert pupil measurements from pixels to millimeters.
    
    EXACT COPY from Pupil_Stress_Analysis_Notebook.ipynb
    This is the measurement logic used in tiered detection pipeline.
    
    Uses iris diameter as reference (average human iris = 12mm).
    
    Parameters:
    -----------
    pupil_radius_px : int
        Pupil radius in pixels
    iris_radius_px : int
        Iris radius in pixels
    
    Returns:
    --------
    tuple: (pupil_diameter_px, pupil_diameter_mm, pixels_per_mm)
        - pupil_diameter_px: Pupil diameter in pixels
        - pupil_diameter_mm: Pupil diameter in millimeters
        - pixels_per_mm: Conversion factor (pixels per millimeter)
    
    Example:
    --------
    >>> # From notebook: p_rad_px = 30, i_rad_px = 120
    >>> pupil_px, pupil_mm, px_per_mm = measure_pupil_diameter_from_notebook(30, 120)
    >>> print(f"Pupil: {pupil_px}px = {pupil_mm:.2f}mm")
    Pupil: 60px = 3.00mm
    """
    # Calculate diameters from radii
    iris_diameter_px = iris_radius_px * 2
    pupil_diameter_px = pupil_radius_px * 2
    
    # Calculate pixel-to-mm conversion factor
    pixels_per_mm = iris_diameter_px / AVG_IRIS_DIAMETER_MM
    
    # Convert pupil measurements
    pupil_diameter_mm = pupil_diameter_px / pixels_per_mm
    
    return pupil_diameter_px, pupil_diameter_mm, pixels_per_mm


def validate_pupil_measurement(pupil_diameter_mm: float) -> Tuple[bool, str]:
    """
    Validate if pupil diameter is within physiological range.
    
    Based on Pupil_Stress_Analysis_Notebook.ipynb validation logic.
    
    Parameters:
    -----------
    pupil_diameter_mm : float
        Pupil diameter in millimeters
    
    Returns:
    --------
    tuple: (is_valid, message)
        - is_valid: True if measurement is in valid range
        - message: Validation message
    """
    if pupil_diameter_mm < MIN_PUPIL_DIAMETER_MM:
        return False, f"Too small ({pupil_diameter_mm:.2f}mm < {MIN_PUPIL_DIAMETER_MM}mm)"
    elif pupil_diameter_mm > MAX_PUPIL_DIAMETER_MM:
        return False, f"Too large ({pupil_diameter_mm:.2f}mm > {MAX_PUPIL_DIAMETER_MM}mm)"
    else:
        return True, f"Valid ({MIN_PUPIL_DIAMETER_MM}mm - {MAX_PUPIL_DIAMETER_MM}mm)"


if __name__ == "__main__":
    # Test the measurement functions
    print("🧪 Testing Pupil Diameter Measurement from Notebook...")
    print("\n" + "="*60)
    
    # Example 1: Normal pupil
    pupil_r, iris_r = 30, 120
    pupil_px, pupil_mm, px_per_mm = measure_pupil_diameter_from_notebook(pupil_r, iris_r)
    is_valid, msg = validate_pupil_measurement(pupil_mm)
    
    print(f"Example 1: Normal Pupil")
    print(f"  Pupil radius: {pupil_r}px, Iris radius: {iris_r}px")
    print(f"  Pupil diameter: {pupil_px}px = {pupil_mm:.2f}mm")
    print(f"  Conversion: {px_per_mm:.2f} pixels/mm")
    print(f"  Validation: {'✅' if is_valid else '❌'} {msg}")
    
    # Example 2: Dilated pupil (stressed or low light)
    pupil_r, iris_r = 50, 120
    pupil_px, pupil_mm, px_per_mm = measure_pupil_diameter_from_notebook(pupil_r, iris_r)
    is_valid, msg = validate_pupil_measurement(pupil_mm)
    
    print(f"\nExample 2: Dilated Pupil")
    print(f"  Pupil radius: {pupil_r}px, Iris radius: {iris_r}px")
    print(f"  Pupil diameter: {pupil_px}px = {pupil_mm:.2f}mm")
    print(f"  Conversion: {px_per_mm:.2f} pixels/mm")
    print(f"  Validation: {'✅' if is_valid else '❌'} {msg}")
    
    print("\n" + "="*60)
    print("✅ Module ready for use in the pipeline!")

