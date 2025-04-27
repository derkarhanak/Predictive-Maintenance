import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger

def load_production_model():
    """
    Load the production model for inference
    
    Returns:
        Loaded Keras model
    """
    model_path = os.path.join(CONFIG["models_dir"], "production_model.h5")
    
    if not os.path.exists(model_path):
        logger.error(f"Production model not found at {model_path}")
        return None
    
    logger.info(f"Loading production model from {model_path}")
    model = load_model(model_path)
    return model

def predict_rul(model, input_data):
    """
    Predict RUL using the trained model
    
    Args:
        model: Trained Keras model
        input_data: Input data for prediction (vibration sequences)
        
    Returns:
        Predicted RUL values
    """
    if model is None:
        logger.error("Model is None, cannot make predictions")
        return None
    
    # Ensure input data has the right shape
    if len(input_data.shape) == 2:
        # Add batch dimension if missing
        input_data = np.expand_dims(input_data, axis=0)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions

def create_sample_input(sequence_length=100, num_features=3):
    """
    Create a sample input for testing the model
    
    Args:
        sequence_length: Length of the sequence
        num_features: Number of features
        
    Returns:
        Sample input data
    """
    # Create random vibration data
    sample_input = np.random.randn(1, sequence_length, num_features)
    return sample_input

def test_model_inference():
    """
    Test model inference with sample data
    
    This function:
    1. Loads the production model
    2. Creates sample input data
    3. Makes predictions
    4. Prints the results
    
    Returns:
        True if successful, False otherwise
    """
    # Load the model
    model = load_production_model()
    
    if model is None:
        return False
    
    # Create sample input
    sample_input = create_sample_input(
        sequence_length=CONFIG["sequence_length"],
        num_features=len(CONFIG["feature_columns"])
    )
    
    # Make predictions
    predictions = predict_rul(model, sample_input)
    
    if predictions is None:
        return False
    
    logger.info(f"Sample prediction: {predictions[0][0]:.2f}")
    return True

if __name__ == "__main__":
    test_model_inference()
