#!/bin/bash

# Run tests for the predictive maintenance system

echo "Running system tests..."
python3 -m unittest src/tests/test_system.py

echo "Running API tests..."
python3 -m unittest src/tests/test_api.py

echo "Testing complete!"
