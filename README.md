# 🐍 Python Backend - Iris Stress Detection API

Flask-based REST API for ML-powered stress detection from eye images.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```

Server will start on `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```
GET /
GET /health
```

### Predict Stress Level
```
POST /predict

Body (multipart/form-data):
  - image: File (required) - Eye image (JPG/PNG)
  - age: Integer (optional) - User age (default: 30)

Response:
{
  "success": true,
  "prediction": {
    "stress_level": "Normal" | "Stress",
    "stress_probability": 0.76,
    "stress_percentage": 76.0,
    "confidence": "High" | "Medium" | "Low"
  },
  "pupil_detection": {...},
  "iris_detection": {...},
  "measurements": {...}
}
```

## 🏗️ Architecture

```
Frontend (React)
    ↓
Flask API (Port 5000)
    ↓
├── Detection Pipeline
├── Measurement Pipeline
└── ML Model (TensorFlow)
```

## 📁 Project Structure

```
Python_Backend/
├── app.py                 # Flask application
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── detection/             # Eye detection modules
├── measurement/           # Pupil & iris measurements
├── pipeline/              # ML inference pipeline
├── layers/                # Custom TensorFlow layers
├── utils/                 # Preprocessing utilities
└── Model/                 # Trained model files
```

## 🔧 Configuration

Edit `config.py` to customize:
- Model path
- Detection thresholds
- Stress classification threshold
- Image preprocessing parameters

## 🐛 Troubleshooting

### Model Not Loading
- Check `Model/best_dual_stream_age_aware_model.keras` exists
- Verify TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`

### Detection Failing
- Ensure image clearly shows pupil and iris
- Try different lighting conditions
- Image should be at least 200x200 pixels

### CORS Errors
- Check Flask CORS configuration in `app.py`
- Frontend must run on port 5173 or 5174

## 📊 Model Details

- **Type**: Dual-stream CNN (Pupil + Iris)
- **Architecture**: EfficientNetB0 + Weighted Feature Fusion
- **Training Accuracy**: 99.91% AUC-PR
- **Input**: Eye images (224x224x5 channels)
- **Output**: Stress probability (0-1)

## ⚡ Performance

- Average inference time: ~2-3 seconds
- Supports concurrent requests (Flask threaded mode)
- Model loaded once at startup

## 🔒 Security

- File validation (image types only)
- Size limits enforced
- Temporary files auto-deleted
- No sensitive data stored

## 📝 Notes

- This backend is independent from the Node.js server
- Both servers run simultaneously
- Node.js handles: Auth, MongoDB, file storage
- Flask handles: ML predictions only
