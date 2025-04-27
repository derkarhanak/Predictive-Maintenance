import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger

def load_processed_data():
    """
    Load the processed data from disk
    
    Returns:
        X: Features (sequences of vibration data)
        y: Target (RUL values)
    """
    processed_dir = os.path.join(CONFIG["data_dir"], "processed")
    
    # Try to load the latest data
    X_path = os.path.join(processed_dir, "X_data_latest.npy")
    y_path = os.path.join(processed_dir, "y_data_latest.npy")
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        logger.info(f"Loading processed data from {X_path} and {y_path}")
        X = np.load(X_path)
        y = np.load(y_path)
        return X, y
    else:
        logger.error("Processed data not found. Please run the data processing pipeline first.")
        return None, None

def split_data(X, y, test_size=0.2, validation_size=0.2, random_state=CONFIG["random_seed"]):
    """
    Split the data into training, validation, and test sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Then split the training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, random_state=random_state
    )
    
    logger.info(f"Data split: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def visualize_data_distribution(y_train, y_val, y_test, save_dir=None):
    """
    Visualize the distribution of RUL values in the dataset
    
    Args:
        y_train: Training target values
        y_val: Validation target values
        y_test: Test target values
        save_dir: Directory to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histograms of RUL values
    plt.hist(y_train, bins=20, alpha=0.5, label='Training')
    plt.hist(y_val, bins=20, alpha=0.5, label='Validation')
    plt.hist(y_test, bins=20, alpha=0.5, label='Test')
    
    plt.xlabel('Remaining Useful Life (RUL)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RUL Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'rul_distribution.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Saved RUL distribution plot to {save_dir}")
    
    plt.close()

def prepare_data_for_training():
    """
    Prepare the data for model training
    
    This function:
    1. Loads the processed data
    2. Splits it into training, validation, and test sets
    3. Visualizes the data distribution
    4. Saves the split data to disk
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load the processed data
    X, y = load_processed_data()
    
    if X is None or y is None:
        return None, None, None, None, None, None
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, 
        test_size=1-CONFIG["train_test_split"], 
        validation_size=CONFIG["validation_split"]
    )
    
    # Visualize the data distribution
    visualization_dir = os.path.join(CONFIG["data_dir"], "visualizations")
    visualize_data_distribution(y_train, y_val, y_test, save_dir=visualization_dir)
    
    # Save the split data
    split_data_dir = os.path.join(CONFIG["data_dir"], "split")
    os.makedirs(split_data_dir, exist_ok=True)
    
    np.save(os.path.join(split_data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(split_data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(split_data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(split_data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(split_data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(split_data_dir, "y_test.npy"), y_test)
    
    logger.info(f"Saved split data to {split_data_dir}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    prepare_data_for_training()
