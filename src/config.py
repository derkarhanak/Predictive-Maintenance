import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictive_maintenance.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# System configuration
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
    "models_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    "dataset_url": "https://engineering.case.edu/bearingdatacenter/download-data-file",
    "random_seed": 42,
    "train_test_split": 0.8,
    "validation_split": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 10,
    "sequence_length": 100,
    "feature_columns": ["vibration_x", "vibration_y", "vibration_z"],
    "target_column": "RUL"
}

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    logger.info("Directory structure verified.")

def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    """Main entry point for the application."""
    logger.info("Initializing Predictive Maintenance System")
    create_directories()
    logger.info("System initialized successfully")

if __name__ == "__main__":
    main()
