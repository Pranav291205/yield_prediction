import os

# Model paths
MODEL_PATH = "models/seq2seq_lstm_model_7percent_final.h5"
PREPROCESSING_PATH = "models/seq2seq_preprocessing_7percent_final.pkl"

# API Configuration
API_TITLE = "Crop Yield Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Seq2Seq LSTM model for crop yield prediction with ±7% uncertainty"

# Model parameters
LOOKBACK = 5
UNCERTAINTY_PERCENT = 7.0
TARGET_ACCURACY = 75.0

# Input validation
MIN_LOOKBACK_SAMPLES = 5
MAX_LOOKBACK_SAMPLES = 10
