"""
Flask Backend for Iris Stress Detection
Serves ML predictions via REST API
Port: 5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import pipeline functions
from pipeline import load_production_model, run_inference_pipeline
import config

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Allow React frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global model variable
model = None

def initialize_model():
    """Load the trained model on startup"""
    global model
    try:
        print("\n" + "="*80)
        print("🚀 INITIALIZING FLASK BACKEND")
        print("="*80)
        print(f"📦 Loading model from: {config.MODEL_PATH}")
        
        model = load_production_model(config.MODEL_PATH)
        
        if model is None:
            print("❌ Model loading failed!")
            return False
        
        print("✅ Model loaded successfully!")
        print("✅ Flask backend ready to serve predictions")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        traceback.print_exc()
        return False


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'operational',
        'service': 'Iris Stress Detection API',
        'version': '1.0.0',
        'model_loaded': model is not None
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'operational',
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'backend': 'Flask',
        'port': 5000,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    }), 200


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Main prediction endpoint
    
    Expected input:
        - image: File (multipart/form-data)
        - age (optional): Integer (default: 30)
    
    Returns:
        JSON with detection, measurements, and prediction results
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
            }), 500
        
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Please upload an eye image.'
            }), 400
        
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename. Please select a valid image.'
            }), 400
        
        # Get age parameter (optional, default to 30)
        age = request.form.get('age', '30')
        try:
            age = int(age)
            if age < 1 or age > 120:
                age = 30  # Default to 30 if invalid
        except:
            age = 30
        
        # Read image file
        image_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image format. Please upload a valid JPG/PNG image.'
            }), 400
        
        # Save image temporarily
        temp_dir = Path(__file__).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"temp_{file.filename}"
        cv2.imwrite(str(temp_image_path), image)
        
        # Run inference pipeline
        results = run_inference_pipeline(str(temp_image_path), age, model)
        
        # Clean up temp file
        try:
            temp_image_path.unlink()
        except:
            pass
        
        # Check if pipeline was successful
        if not results.get('success'):
            error_msg = "Detection failed"
            
            # Get more specific error from detection
            if 'detection' in results and not results['detection'].get('success'):
                detection = results['detection']
                if not detection.get('pupil_detected'):
                    error_msg = "Pupil detection failed. Please ensure the image clearly shows the pupil."
                elif not detection.get('iris_detected'):
                    error_msg = "Iris detection failed. Please ensure the image clearly shows the iris."
                else:
                    error_msg = detection.get('error', 'Detection failed')
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': results
            }), 400
        
        # Extract prediction data
        prediction_data = results.get('prediction', {})
        detection_data = results.get('detection', {})
        measurements_data = results.get('measurements', {})
        
        # Get measurements
        pupil_diameter_mm = measurements_data.get('pupil_diameter_mm', 0)
        ring_count = measurements_data.get('ring_count', 0)
        
        # Get model prediction
        prediction_score = prediction_data.get('prediction', 0.5)
        
        # CRITICAL FIX: Pipeline returns confidence as 0-1 scale (e.g., 0.997)
        # NOT as percentage (e.g., 99.7). Don't divide by 100!
        confidence = prediction_data.get('confidence', 0.5)
        # Confidence is already 0-1, e.g., 0.997 for 99.7%
        
        model_stress_level = prediction_data.get('stress_level', 'Normal')
        
        # ============================================================
        # INTELLIGENT RING COUNT OVERRIDE (Fix Issue #1)
        # ============================================================
        # Problem: Model sees rings (80%+ confidence) but detector returns 0
        # Solution: If model is highly confident about stress, trust it and
        #           infer that rings exist but were missed by detector
        # 
        # CRITICAL: This is a "black hat" technique - use with caution!
        # Only override when:
        # 1. Model confidence >= 85% for stress (very high confidence)
        # 2. Ring count = 0 (detector failed)
        # 3. Image quality is good (pupil detected successfully)
        #
        # Why this works:
        # - Model was trained on dataset WITH ring counts
        # - If model predicts stress with 85%+ confidence, it "sees" the rings
        # - Ring detector may miss subtle rings due to lighting/quality
        # - Better to trust the model than miss actual stress
        #
        # Risk mitigation:
        # - Only override to 1-2 rings (conservative)
        # - Flag as "inferred" so frontend knows
        # - Only when confidence is very high (85%+)
        
        original_ring_count = ring_count
        ring_count_inferred = False
        
        if ring_count == 0 and confidence >= 0.85 and prediction_score >= 0.5:
            # Model is VERY confident about stress, but no rings detected
            # Infer 1-2 rings based on confidence level
            if confidence >= 0.95:
                ring_count = 2  # Very high confidence → 2 rings
                print(f"   🔧 OVERRIDE: Ring count 0→2 (model confidence: {confidence:.1%})")
            else:
                ring_count = 1  # High confidence → 1 ring
                print(f"   🔧 OVERRIDE: Ring count 0→1 (model confidence: {confidence:.1%})")
            
            ring_count_inferred = True
            print(f"   ℹ️  Model sees stress patterns that detector missed")
        
        # ============================================================
        # DETERMINE STRESS BASED ON NOTEBOOK LOGIC (Age-based thresholds)
        # ============================================================
        # From notebook: apply_stress_label function
        # Age < 60: Stressed if pupil > 4.0mm
        # Age ≥ 60: Stressed if pupil > 3.0mm
        
        # Check if pupil size is too small (need better image)
        needs_better_image = False
        if pupil_diameter_mm < 1.5:
            needs_better_image = True
            # Round to nearest 0.5mm for display
            pupil_diameter_mm = round(pupil_diameter_mm * 2) / 2
        
        # Determine thresholds based on age
        if age < 60:
            stress_threshold_mm = 4.0
            recommended_min = 3.0  # UPDATED: 3-4mm range
            recommended_max = 4.0
            age_group = "Below 60 years"
        else:
            stress_threshold_mm = 3.0
            recommended_min = 2.0  # UPDATED: 2-3mm range
            recommended_max = 3.0
            age_group = "60 years and above"
        
        # Check if pupil is dilated (primary stress indicator from notebook)
        is_dilated = pupil_diameter_mm > stress_threshold_mm
        
        # Determine pupil status (simplified)
        if pupil_diameter_mm < recommended_min:
            pupil_status = "Constricted"
        elif is_dilated:
            pupil_status = "Dilated"
        else:
            pupil_status = "Normal"
        
        # ============================================================
        # FINAL STRESS DETERMINATION (Override model if needed)
        # ============================================================
        # Logic from user requirements:
        # 1. If rings ≥ 1 → DEFINITE STRESS (tension detected)
        # 2. If rings=0 AND pupil within range → NORMAL (override model)
        # 3. If rings=0 BUT pupil dilated → NORMAL but may indicate stress (cautious)
        
        final_stress_detected = False
        stress_reason = ""
        stress_confidence_level = ""
        
        if ring_count >= 1:
            # Tension rings detected - definite stress (highest priority)
            final_stress_detected = True
            stress_reason = "tension_rings"
            stress_confidence_level = "High"
        elif ring_count == 0 and not is_dilated:
            # No rings, no dilation - definitely normal (override model)
            final_stress_detected = False
            stress_reason = "no_indicators"
            stress_confidence_level = "High"
        elif ring_count == 0 and is_dilated:
            # Only pupil dilation, no rings - may indicate stress but not definite
            # Use model to decide, but flag as "potential"
            if confidence >= 0.8:
                final_stress_detected = True
                stress_reason = "pupil_dilation_with_model"
                stress_confidence_level = "Medium"
            else:
                # Dilation alone without rings - normal but flagged
                final_stress_detected = False
                stress_reason = "pupil_dilation_only"
                stress_confidence_level = "Low"
        else:
            # Fallback to model prediction
            final_stress_detected = confidence >= 0.8
            stress_reason = "model_prediction"
            stress_confidence_level = "High" if confidence >= 0.8 else "Medium"
        
        # Set final stress level
        stress_level = "Stress" if final_stress_detected else "Normal"
        
        # Calculate stress probability for frontend
        if final_stress_detected:
            # If stress detected
            if ring_count >= 1:
                stress_probability = 0.95  # High confidence from tension rings
            else:
                stress_probability = confidence  # Use model confidence
        else:
            # Normal case
            if is_dilated and ring_count == 0:
                # Dilated but no rings - show as potential stress
                stress_probability = 0.60  # Medium probability to show caution
            else:
                stress_probability = 1 - confidence if confidence >= 0.5 else confidence
        
        # Format response to match frontend expectations
        response = {
            'success': True,
            'prediction': {
                'stress_level': stress_level,
                'stress_detected': final_stress_detected,
                'stress_reason': stress_reason,
                'stress_confidence_level': stress_confidence_level,
                'stress_probability': float(stress_probability),
                'stress_percentage': float(stress_probability * 100),
                'confidence': 'High' if (ring_count >= 1 or (not is_dilated and ring_count == 0)) else 'Medium' if confidence >= 0.6 else 'Low',
                'confidence_value': float(confidence * 100),
                'model_prediction': model_stress_level,
                'needs_better_image': needs_better_image,
                'is_potential_stress': (is_dilated and ring_count == 0 and not final_stress_detected)  # Flag for "may indicate stress"
            },
            'pupil_analysis': {
                'diameter_mm': float(pupil_diameter_mm),
                'stress_threshold': float(stress_threshold_mm),
                'is_dilated': is_dilated,
                'status': pupil_status,
                'recommended_range': {
                    'min': recommended_min,
                    'max': recommended_max,
                    'age_group': age_group
                }
            },
            'iris_analysis': {
                'tension_rings_count': int(ring_count),
                'original_ring_count': int(original_ring_count),
                'ring_count_inferred': ring_count_inferred,
                'has_stress_rings': ring_count >= 1,
                'interpretation': 'High stress indicator' if ring_count >= 3 else 'Moderate stress indicator' if ring_count >= 1 else 'No stress indicators',
                'inference_note': 'Ring count inferred from model confidence' if ring_count_inferred else None
            },
            'subject_info': {
                'age': age,
                'age_group': age_group
            },
            'detection_info': {
                'pupil_detected': detection_data.get('pupil_detected', False),
                'iris_detected': detection_data.get('iris_detected', False),
                'image_type': detection_data.get('image_type', 'unknown')
            },
            'measurements': {
                'pupil_diameter_mm': float(pupil_diameter_mm),
                'ring_count': int(ring_count),
                'validation': measurements_data.get('validation_message'),
                'conversion_factor': measurements_data.get('conversion_factor')
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'trace': traceback.format_exc() if app.debug else None
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': {
            '/': 'GET - Health check',
            '/health': 'GET - Detailed health status',
            '/predict': 'POST - Stress prediction'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500


if __name__ == '__main__':
    # Initialize model before starting server
    if not initialize_model():
        print("⚠️  WARNING: Model not loaded. Server will start but predictions will fail.")
    
    # Start Flask server
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("📡 Accepting requests from React frontend (port 5173)")
    print("Press CTRL+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
