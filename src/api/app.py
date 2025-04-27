from flask import Flask, render_template, request, jsonify
import os
import sys
import numpy as np
import json
import tensorflow as tf
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger
from src.model.predict import load_production_model, predict_rul, create_sample_input
from src.visualization.visualization import (
    create_rul_distribution_plot,
    create_training_history_plot,
    create_actual_vs_predicted_plot,
    create_model_comparison_plot,
    create_feature_importance_plot,
    create_time_series_plot,
    create_health_index_plot,
    create_sample_visualizations
)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static'))

# Load the model at startup
model = None
try:
    model = load_production_model()
    logger.info("Production model loaded successfully")
except Exception as e:
    logger.error(f"Error loading production model: {str(e)}")

# Sample data for demonstration
sample_data = {
    'timestamps': list(range(100)),
    'vibration_x': [np.sin(i/5) + np.random.normal(0, 0.1) for i in range(100)],
    'vibration_y': [np.cos(i/5) + np.random.normal(0, 0.1) for i in range(100)],
    'vibration_z': [np.sin(i/10) + np.random.normal(0, 0.1) for i in range(100)],
    'health_index': [100 - i/100*100 for i in range(100)]
}

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/visualizations')
def get_visualizations():
    """Get all visualizations for the dashboard"""
    try:
        visualizations = create_sample_visualizations()
        return jsonify({
            'status': 'success',
            'visualizations': visualizations
        })
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction using the loaded model"""
    try:
        # Get input data from request
        data = request.json
        
        # If real data is provided, use it
        if data and 'vibration_data' in data:
            input_data = np.array(data['vibration_data'])
        else:
            # Otherwise use sample data
            input_data = create_sample_input(
                sequence_length=CONFIG["sequence_length"],
                num_features=len(CONFIG["feature_columns"])
            )
        
        # Make prediction
        if model is not None:
            prediction = predict_rul(model, input_data)
            rul_value = float(prediction[0][0])
            
            # Calculate health index (100% = full health, 0% = failure)
            # Assuming max RUL is 100
            health_index = min(100, max(0, rul_value))
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return jsonify({
                'status': 'success',
                'prediction': {
                    'rul': rul_value,
                    'health_index': health_index,
                    'timestamp': timestamp
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get sensor data for time series visualization"""
    try:
        # In a real application, this would fetch data from a database
        # For demonstration, we'll use sample data
        return jsonify({
            'status': 'success',
            'data': sample_data
        })
    except Exception as e:
        logger.error(f"Error getting sensor data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def run_app(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask application"""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_app(debug=True)
