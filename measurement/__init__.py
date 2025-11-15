"""
Measurement modules for pupil/iris analysis
"""

from .pupil_diameter import (
    measure_pupil_diameter_from_notebook,
    validate_pupil_measurement,
    AVG_IRIS_DIAMETER_MM,
    MIN_PUPIL_DIAMETER_MM,
    MAX_PUPIL_DIAMETER_MM
)

# Alias for backwards compatibility with pipeline
measure_pupil_diameter = measure_pupil_diameter_from_notebook

__all__ = [
    'measure_pupil_diameter_from_notebook',
    'measure_pupil_diameter',  # Alias for pipeline
    'validate_pupil_measurement',
    'AVG_IRIS_DIAMETER_MM',
    'MIN_PUPIL_DIAMETER_MM',
    'MAX_PUPIL_DIAMETER_MM'
]
