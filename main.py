import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from config import *
import warnings
warnings.filterwarnings('ignore')


# ==================== LOAD MODEL & PREPROCESSING ====================
print("Loading model and preprocessing data...")


# Load trained model with custom_objects
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")
    raise



# Load preprocessing data
try:
    with open(PREPROCESSING_PATH, 'rb') as f:
        preprocessing_data = pickle.load(f)
    
    scalers = preprocessing_data['scalers']
    scaler_target = preprocessing_data['scaler_target']
    all_features = preprocessing_data['all_features']
    le_weather = preprocessing_data['le_weather']
    le_crop = preprocessing_data['le_crop']
    le_disease = preprocessing_data['le_disease']
    
    print(f"✓ Preprocessing data loaded from {PREPROCESSING_PATH}")
except Exception as e:
    print(f"✗ Error loading preprocessing data: {str(e)}")
    raise


# ==================== PYDANTIC MODELS ====================


class YieldPredictionInput(BaseModel):
    """Input model for single prediction"""
    historical_yield: float
    ndvi: float
    healthy_area: float
    weed_area: float
    soil_area: float
    weather: str  # 'Cold', 'Moderate', 'Warm'
    crop_type: str
    disease_class: str


class BatchPredictionInput(BaseModel):
    """Input model for batch predictions"""
    samples: List[YieldPredictionInput]


class YieldPredictionOutput(BaseModel):
    """Output model for predictions"""
    predicted_yield: float
    uncertainty_range: Dict[str, float]
    confidence_percent: float
    status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_name: str
    api_version: str
    features: int


# ==================== FASTAPI APP ====================


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)


# ==================== HELPER FUNCTIONS ====================


def encode_features(weather: str, crop_type: str, disease_class: str) -> tuple:
    """
    Encode categorical features
    """
    try:
        weather_encoded = le_weather.transform([weather])[0]
    except:
        raise HTTPException(status_code=400, detail=f"Invalid weather: {weather}")
    
    try:
        crop_encoded = le_crop.transform([crop_type])[0]
    except:
        raise HTTPException(status_code=400, detail=f"Invalid crop type: {crop_type}")
    
    try:
        disease_encoded = le_disease.transform([disease_class])[0]
    except:
        raise HTTPException(status_code=400, detail=f"Invalid disease class: {disease_class}")
    
    return weather_encoded, crop_encoded, disease_encoded


def normalize_features(historical_yield: float, ndvi: float, healthy_area: float,
                       weed_area: float, soil_area: float, 
                       weather_encoded: float, crop_encoded: float, 
                       disease_encoded: float) -> np.ndarray:
    """
    Normalize input features using saved scalers
    """
    features_dict = {
        'Historical_Yield': historical_yield,
        'NDVI': ndvi,
        'Healthy_Area': healthy_area,
        'Weed_Area': weed_area,
        'Soil_Area': soil_area,
        'Weather_Encoded': weather_encoded,
        'Crop_Type_Encoded': crop_encoded,
        'Disease_Encoded': disease_encoded
    }
    
    normalized = []
    for feature_name in all_features:
        value = features_dict[feature_name]
        scaler = scalers[feature_name]
        normalized_value = scaler.transform([[value]])[0][0]
        normalized.append(normalized_value)
    
    return np.array(normalized)


def make_prediction(input_data: YieldPredictionInput) -> Dict:
    """
    Make a single prediction using the model
    """
    # Validate inputs
    if not (0 <= input_data.ndvi <= 1):
        raise HTTPException(status_code=400, detail="NDVI must be between 0 and 1")
    
    if not (0 <= input_data.healthy_area <= 1):
        raise HTTPException(status_code=400, detail="Healthy area must be between 0 and 1")
    
    if not (0 <= input_data.weed_area <= 1):
        raise HTTPException(status_code=400, detail="Weed area must be between 0 and 1")
    
    if not (0 <= input_data.soil_area <= 1):
        raise HTTPException(status_code=400, detail="Soil area must be between 0 and 1")
    
    # Check area sum
    total_area = input_data.healthy_area + input_data.weed_area + input_data.soil_area
    if not (0.95 <= total_area <= 1.05):
        raise HTTPException(status_code=400, 
                          detail=f"Field areas must sum to ~1.0, got {total_area:.3f}")
    
    # Encode categorical features
    weather_enc, crop_enc, disease_enc = encode_features(
        input_data.weather, 
        input_data.crop_type, 
        input_data.disease_class
    )
    
    # Normalize features
    normalized_features = normalize_features(
        input_data.historical_yield,
        input_data.ndvi,
        input_data.healthy_area,
        input_data.weed_area,
        input_data.soil_area,
        weather_enc,
        crop_enc,
        disease_enc
    )
    
    # Create sequence (replicate for lookback length)
    # Using the current state repeated LOOKBACK times
    sequence = np.tile(normalized_features, (LOOKBACK, 1))
    sequence = sequence.reshape(1, LOOKBACK, len(all_features))
    
    # Prepare decoder input (historical yield scaled)
    historical_yield_scaled = scalers['Historical_Yield'].transform(
        [[input_data.historical_yield]]
    )[0][0]
    decoder_input = np.array([[historical_yield_scaled]])
    
    # Make prediction
    try:
        prediction_scaled = model.predict([sequence, decoder_input], verbose=0)[0][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")
    
    # Inverse transform prediction
    prediction = scaler_target.inverse_transform([[prediction_scaled]])[0][0]
    
    # Calculate uncertainty band
    uncertainty_value = prediction * (UNCERTAINTY_PERCENT / 100)
    lower_bound = prediction - uncertainty_value
    upper_bound = prediction + uncertainty_value
    
    # Estimate confidence (0-100%)
    # Closer to historical yield = higher confidence
    error_percent = abs(prediction - input_data.historical_yield) / input_data.historical_yield * 100
    confidence = max(0, 100 - error_percent)
    
    return {
        'predicted_yield': round(float(prediction), 2),
        'uncertainty_range': {
            'lower_bound': round(float(lower_bound), 2),
            'upper_bound': round(float(upper_bound), 2),
            'margin': round(float(uncertainty_value), 2)
        },
        'confidence_percent': round(float(confidence), 2),
        'status': 'success'
    }


# ==================== ROUTES ====================


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Crop Yield Prediction API"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_name": "Seq2Seq LSTM",
        "api_version": API_VERSION,
        "features": len(all_features)
    }


@app.post("/predict", response_model=YieldPredictionOutput, tags=["Predictions"])
async def predict(input_data: YieldPredictionInput):
    """
    Make a single yield prediction
    
    Example request body:
    {
        "historical_yield": 95.5,
        "ndvi": 0.75,
        "healthy_area": 0.78,
        "weed_area": 0.12,
        "soil_area": 0.10,
        "weather": "Warm",
        "crop_type": "Cabbage",
        "disease_class": "Healthy"
    }
    """
    try:
        result = make_prediction(input_data)
        return YieldPredictionOutput(
            predicted_yield=result['predicted_yield'],
            uncertainty_range=result['uncertainty_range'],
            confidence_percent=result['confidence_percent'],
            status=result['status']
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Make batch predictions for multiple samples
    
    Returns array of predictions
    """
    if len(batch_input.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")
    
    if len(batch_input.samples) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 samples per batch")
    
    predictions = []
    
    for idx, sample in enumerate(batch_input.samples):
        try:
            result = make_prediction(sample)
            predictions.append({
                'sample_index': idx,
                **result
            })
        except HTTPException as e:
            predictions.append({
                'sample_index': idx,
                'status': 'error',
                'error': e.detail
            })
        except Exception as e:
            predictions.append({
                'sample_index': idx,
                'status': 'error',
                'error': str(e)
            })
    
    return {
        'total_samples': len(batch_input.samples),
        'successful': len([p for p in predictions if p['status'] == 'success']),
        'predictions': predictions
    }


@app.get("/valid-categories", tags=["Information"])
async def valid_categories():
    """
    Get valid categories for categorical features
    """
    return {
        'weather_categories': list(le_weather.classes_),
        'crop_types': list(le_crop.classes_),
        'disease_classes': list(le_disease.classes_)
    }


@app.get("/model-info", tags=["Information"])
async def model_info():
    """
    Get model information and configuration
    """
    return {
        'model_name': 'Seq2Seq LSTM',
        'model_type': 'Bidirectional LSTM Encoder-Decoder',
        'api_version': API_VERSION,
        'uncertainty_percent': UNCERTAINTY_PERCENT,
        'target_accuracy': TARGET_ACCURACY,
        'input_features': all_features,
        'lookback_window': LOOKBACK,
        'total_parameters': int(model.count_params())
    }


# ==================== ERROR HANDLING ====================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Value error: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Display API information on startup"""
    print("\n" + "="*80)
    print("🚀 CROP YIELD PREDICTION API - STARTED SUCCESSFULLY!")
    print("="*80)
    print("\n📚 API DOCUMENTATION:")
    print("   Interactive Docs: http://localhost:8000/docs")
    print("   Alternative Docs: http://localhost:8000/redoc")
    print("\n🔗 USEFUL ENDPOINTS:")
    print("   Health Check: http://localhost:8000/health")
    print("   Model Info: http://localhost:8000/model-info")
    print("   Valid Categories: http://localhost:8000/valid-categories")
    print("   Make Prediction: POST http://localhost:8000/predict")
    print("\n✅ Server is running at: http://localhost:8000")
    print("="*80 + "\n")


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print("\n⏳ Starting API server...\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

