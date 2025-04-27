import os
import sys
import argparse

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import logger

def main():
    """
    Main entry point for data processing pipeline
    
    This script orchestrates the data processing pipeline by:
    1. Downloading the CWRU Bearing Dataset
    2. Preprocessing the data for RUL prediction
    """
    parser = argparse.ArgumentParser(description='Process CWRU Bearing Dataset for RUL prediction')
    parser.add_argument('--download-only', action='store_true', help='Only download the dataset without preprocessing')
    parser.add_argument('--preprocess-only', action='store_true', help='Only preprocess existing data without downloading')
    args = parser.parse_args()
    
    logger.info("Starting data processing pipeline")
    
    # Import modules here to avoid circular imports
    from src.data_processing.download_dataset import download_cwru_bearing_dataset
    from src.data_processing.preprocess_data import process_cwru_dataset
    
    # Download the dataset if needed
    if not args.preprocess_only:
        logger.info("Downloading CWRU Bearing Dataset")
        download_cwru_bearing_dataset()
    
    # Preprocess the data if needed
    if not args.download_only:
        logger.info("Preprocessing CWRU Bearing Dataset")
        process_cwru_dataset()
    
    logger.info("Data processing pipeline completed")

if __name__ == "__main__":
    main()
