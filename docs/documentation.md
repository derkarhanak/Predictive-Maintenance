# Predictive Maintenance System Documentation

## Overview

This documentation provides comprehensive information about the Predictive Maintenance System for bearing fault detection and Remaining Useful Life (RUL) prediction. The system uses deep learning approaches (CNN-LSTM) to analyze vibration data from the Case Western Bearing Dataset and predict when maintenance will be required.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation](#installation)
3. [Data Processing](#data-processing)
4. [Model Development](#model-development)
5. [Web Application](#web-application)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Usage Guide](#usage-guide)

## System Architecture

The Predictive Maintenance System consists of the following components:

```
predictive_maintenance/
├── data/                  # Dataset storage
├── models/                # Trained model files
├── src/                   # Source code
│   ├── data_processing/   # Data preprocessing modules
│   ├── model/             # Model definition and training
│   ├── visualization/     # Visualization components
│   └── api/               # Flask API endpoints
├── static/                # Static assets for web app
├── templates/             # HTML templates
└── docs/                  # Documentation
```

The system follows a modular architecture with clear separation of concerns:
- Data processing pipeline for handling the Case Western Bearing Dataset
- Deep learning models (LSTM, CNN, and CNN-LSTM hybrid) for RUL prediction
- Visualization components for data analysis and model performance evaluation
- Flask web application for interactive dashboard and API endpoints

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd predictive_maintenance
   ```

2. Run the setup script to install required dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. Verify the installation:
   ```
   chmod +x run_tests.sh
   ./run_tests.sh
   ```

## Data Processing

### Dataset

The system uses the Case Western Reserve University Bearing Dataset, which contains vibration measurements from bearings with different fault conditions. The dataset includes:
- Normal baseline data
- Drive end bearing fault data (12k and 48k sampling rates)
- Fan end bearing fault data

### Data Pipeline

The data processing pipeline consists of the following steps:

1. **Download Dataset**: The `download_dataset.py` script downloads the CWRU dataset files.

2. **Preprocess Data**: The `preprocess_data.py` script:
   - Loads the MATLAB (.mat) files
   - Extracts vibration signals and metadata
   - Simulates RUL values based on fault conditions
   - Creates sequences for time series prediction

3. **Prepare Training Data**: The `prepare_training_data.py` script:
   - Splits the data into training, validation, and test sets
   - Visualizes the data distribution
   - Saves the split data for model training

To run the data pipeline:
```
python src/data_processing/data_pipeline.py
```

## Model Development

### Model Architecture

The system implements three deep learning models:

1. **LSTM Model**: Long Short-Term Memory network for sequence modeling
   - Input: Vibration sequences (shape: sequence_length × features)
   - Output: RUL prediction (scalar value)

2. **CNN Model**: Convolutional Neural Network for feature extraction
   - Input: Vibration sequences (shape: sequence_length × features)
   - Output: RUL prediction (scalar value)

3. **CNN-LSTM Hybrid Model**: Combines CNN for feature extraction and LSTM for sequence modeling
   - Input: Vibration sequences (shape: sequence_length × features)
   - Output: RUL prediction (scalar value)

### Training Process

The `train_models.py` script handles the model training process:
- Loads the prepared training data
- Creates and trains the three model architectures
- Evaluates models on validation and test data
- Compares model performance and selects the best model
- Saves the best model as the production model

To train the models:
```
python src/model/train_models.py
```

### Model Evaluation

Models are evaluated using:
- Mean Squared Error (MSE) for training loss
- Mean Absolute Error (MAE) for interpretability
- Actual vs. Predicted plots for visual assessment

## Web Application

### Dashboard

The web application provides an interactive dashboard with the following features:
- Real-time monitoring of bearing health status
- Prediction of Remaining Useful Life (RUL)
- Visualization of vibration patterns and trends
- Model performance comparison
- Feature importance analysis

### Running the Application

To start the web application:
```
python src/api/app.py
```

The application will be accessible at `http://localhost:5000`.

## API Reference

The system provides the following API endpoints:

### GET /api/visualizations

Returns base64-encoded images of all visualizations.

**Response:**
```json
{
  "status": "success",
  "visualizations": {
    "rul_distribution": "base64-encoded-image",
    "training_history": "base64-encoded-image",
    "actual_vs_predicted": "base64-encoded-image",
    "model_comparison": "base64-encoded-image",
    "feature_importance": "base64-encoded-image",
    "time_series": "base64-encoded-image",
    "health_index": "base64-encoded-image"
  }
}
```

### GET /api/sensor_data

Returns sensor data for time series visualization.

**Response:**
```json
{
  "status": "success",
  "data": {
    "timestamps": [0, 1, 2, ...],
    "vibration_x": [0.1, 0.2, ...],
    "vibration_y": [0.3, 0.4, ...],
    "vibration_z": [0.5, 0.6, ...],
    "health_index": [100, 99, ...]
  }
}
```

### POST /api/predict

Makes a prediction using the loaded model.

**Request Body (optional):**
```json
{
  "vibration_data": [[[0.1, 0.2, 0.3], ...]]
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "rul": 75.5,
    "health_index": 75.5,
    "timestamp": "2025-04-26 20:21:34"
  }
}
```

## Testing

The system includes comprehensive tests for all components:

### System Tests

The `test_system.py` script tests:
- Model loading and prediction
- Data loading and splitting
- Flask application importability

### API Tests

The `test_api.py` script tests:
- Index route
- Visualizations API endpoint
- Sensor data API endpoint
- Prediction API endpoint

To run all tests:
```
chmod +x run_tests.sh
./run_tests.sh
```

## Usage Guide

### Monitoring Bearing Health

1. Access the dashboard at `http://localhost:5000`
2. View the current health status and RUL prediction
3. Click "Run Prediction" to get updated predictions
4. Click "Refresh Data" to update sensor data

### Analyzing Model Performance

1. Scroll down to the "Model Performance" section
2. Review training history, actual vs. predicted plots, and model comparison
3. Analyze feature importance to understand which sensors contribute most to predictions

### Interpreting Results

- **Health Index**: Percentage value indicating bearing health (100% = perfect health)
- **RUL Prediction**: Estimated remaining useful life before maintenance is required
- **Status Alert**: Color-coded alert indicating maintenance urgency:
  - Green: Healthy, no maintenance required
  - Yellow: Warning, schedule maintenance soon
  - Red: Critical, immediate maintenance required
