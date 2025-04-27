#!/bin/bash

# Start the Flask application for the Predictive Maintenance System

echo "Starting Predictive Maintenance System..."

# Set environment variables
export FLASK_APP=src/api/app.py
export FLASK_ENV=production

# Run the Flask application
python -m flask run --host=0.0.0.0 --port=5000
