# Predictive Maintenance System - README

## Project Overview

This project implements a predictive maintenance system for bearing fault detection and Remaining Useful Life (RUL) prediction using the Case Western Bearing Dataset. The system uses deep learning approaches (CNN-LSTM) and is deployed as a Flask web application with an interactive dashboard.

## Key Features

- Data processing pipeline for the Case Western Bearing Dataset
- Deep learning models (LSTM, CNN, and CNN-LSTM hybrid) for RUL prediction
- Interactive dashboard for real-time monitoring of bearing health
- Visualization components for data analysis and model performance
- RESTful API for predictions and data retrieval
- Comprehensive documentation and tests

## Quick Start

1. **Setup the environment**:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Run the application**:
   ```
   chmod +x run_app.sh
   ./run_app.sh
   ```

3. **Access the dashboard**:
   Open your browser and navigate to `http://localhost:5000`

## Project Structure

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

## Documentation

For detailed information about the system, please refer to the [documentation](docs/documentation.md).

## Testing

To run the tests:
```
chmod +x run_tests.sh
./run_tests.sh
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
