import os
import sys
import unittest
import numpy as np
import json
from flask import Flask
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger
from src.api.app import app

class TestFlaskAPI(unittest.TestCase):
    """Test cases for the Flask API"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_index_route(self):
        """Test that the index route returns 200"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200, "Index route should return 200")
    
    def test_visualizations_route(self):
        """Test that the visualizations route returns valid data"""
        response = self.app.get('/api/visualizations')
        self.assertEqual(response.status_code, 200, "Visualizations route should return 200")
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success', "API should return success status")
        self.assertIn('visualizations', data, "API should return visualizations data")
        
        # Check that all required visualizations are present
        required_visualizations = [
            'rul_distribution', 'training_history', 'actual_vs_predicted',
            'model_comparison', 'feature_importance', 'time_series', 'health_index'
        ]
        for vis in required_visualizations:
            self.assertIn(vis, data['visualizations'], f"API should return {vis} visualization")
    
    def test_sensor_data_route(self):
        """Test that the sensor data route returns valid data"""
        response = self.app.get('/api/sensor_data')
        self.assertEqual(response.status_code, 200, "Sensor data route should return 200")
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success', "API should return success status")
        self.assertIn('data', data, "API should return sensor data")
        
        # Check that all required data fields are present
        required_fields = ['timestamps', 'vibration_x', 'vibration_y', 'vibration_z', 'health_index']
        for field in required_fields:
            self.assertIn(field, data['data'], f"API should return {field} data")
    
    @patch('src.api.app.predict_rul')
    @patch('src.api.app.model', new_callable=MagicMock)  # Patch the model object in app.py
    def test_predict_route(self, mock_app_model_object, mock_predict_rul_call):
        """Test that the predict route returns valid predictions"""
        # Configure the mock for the predict_rul function
        # The mock_app_model_object is already a MagicMock, so app.model is not None
        mock_predict_rul_call.return_value = np.array([[75.0]])
        
        # Make request to predict endpoint
        response = self.app.post('/api/predict', 
                                json={},
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200, "Predict route should return 200")
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success', "API should return success status")
        self.assertIn('prediction', data, "API should return prediction data")
        self.assertIn('rul', data['prediction'], "Prediction should include RUL value")
        self.assertIn('health_index', data['prediction'], "Prediction should include health index")
        self.assertIn('timestamp', data['prediction'], "Prediction should include timestamp")

if __name__ == '__main__':
    unittest.main()
