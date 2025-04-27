#!/bin/bash

# Install required Python libraries for the predictive maintenance project

echo "Installing required Python libraries..."

# Core data science and machine learning libraries
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Deep learning libraries
pip install tensorflow keras

# Time series feature extraction
pip install tsfresh

# Web application framework
pip install flask flask-cors gunicorn

# Utilities
pip install requests tqdm pyyaml

echo "Installation complete!"
