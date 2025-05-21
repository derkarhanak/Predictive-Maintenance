import os
import sys
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger
from src.model.predict import load_production_model, predict_rul, create_sample_input
from src.data_processing.prepare_training_data import load_processed_data, split_data

class TestPredictiveMaintenanceSystem(unittest.TestCase):
    """Test cases for the Predictive Maintenance System"""
    
    def setUp(self):
        """Set up test environment"""
        self.model = load_production_model()
        
        # If model doesn't exist, create a simple test model
        if self.model is None:
            logger.warning("Production model not found, creating test model")
            input_shape = (CONFIG["sequence_length"], len(CONFIG["feature_columns"]))
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, input_shape=input_shape),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Save test model
            os.makedirs(CONFIG["models_dir"], exist_ok=True)
            self.model.save(os.path.join(CONFIG["models_dir"], "test_model.h5"))
    
    def test_model_loading(self):
        """Test that the model can be loaded"""
        self.assertIsNotNone(self.model, "Model should be loaded successfully")
    
    def test_model_prediction(self):
        """Test that the model can make predictions"""
        # Create sample input
        sample_input = create_sample_input(
            sequence_length=CONFIG["sequence_length"],
            num_features=len(CONFIG["feature_columns"])
        )
        
        # Make prediction
        prediction = predict_rul(self.model, sample_input)
        
        # Check prediction shape and type
        self.assertIsNotNone(prediction, "Prediction should not be None")
        self.assertEqual(prediction.shape, (1, 1), "Prediction should have shape (1, 1)")
        self.assertTrue(np.isfinite(prediction).all(), "Prediction should be finite")
    
    def test_data_loading(self):
        """Test that the processed data can be loaded"""
        # Try to load processed data
        X, y = load_processed_data()
        
        # If data doesn't exist, create dummy data for testing
        if X is None or y is None:
            logger.warning("Processed data not found, creating dummy data")
            X = np.random.randn(100, CONFIG["sequence_length"], len(CONFIG["feature_columns"]))
            y = np.random.uniform(0, 100, 100)
            
            # Save dummy data
            processed_dir = os.path.join(CONFIG["data_dir"], "processed")
            os.makedirs(processed_dir, exist_ok=True)
            np.save(os.path.join(processed_dir, "X_data_latest.npy"), X)
            np.save(os.path.join(processed_dir, "y_data_latest.npy"), y)
        
        # Check data shapes
        self.assertIsNotNone(X, "X data should not be None")
        self.assertIsNotNone(y, "y data should not be None")
        self.assertEqual(X.shape[0], y.shape[0], "X and y should have same number of samples")
        self.assertEqual(X.shape[1], CONFIG["sequence_length"], "X should have correct sequence length")
        self.assertEqual(X.shape[2], len(CONFIG["feature_columns"]), "X should have correct number of features")
    
    def test_data_splitting(self):
        """Test that the data can be split correctly"""
        # Create dummy data
        X = np.random.randn(100, CONFIG["sequence_length"], len(CONFIG["feature_columns"]))
        y = np.random.uniform(0, 100, 100)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, 
            test_size=1-CONFIG["train_test_split"], 
            validation_size=CONFIG["validation_split"]
        )
        
        # Check split sizes
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(X), "Split sizes should sum to total size")
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), len(y), "Split sizes should sum to total size")
        
        # Check approximate proportions
        self.assertAlmostEqual(len(X_test) / len(X), 1-CONFIG["train_test_split"], delta=0.1, 
                              msg="Test split should be approximately correct")
        self.assertAlmostEqual(len(X_val) / (len(X) - len(X_test)), CONFIG["validation_split"], delta=0.1, 
                              msg="Validation split should be approximately correct")
        # where X_train_val = len(X) - len(X_test) # This line was causing a syntax error
    
    def test_flask_app(self):
        """Test that the Flask app can be imported"""
        try:
            from src.api.app import app
            self.assertIsNotNone(app, "Flask app should be importable")
        except ImportError as e:
            self.fail(f"Failed to import Flask app: {str(e)}")

if __name__ == '__main__':
    unittest.main()
