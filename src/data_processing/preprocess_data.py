import os
import numpy as np
import pandas as pd
import scipy.io
import logging
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, get_timestamp, logger

def load_mat_file(file_path):
    """
    Load a MATLAB .mat file and convert it to a pandas DataFrame
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        DataFrame containing the data
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        
        # CWRU data is typically stored with variable names like 'X100_DE_time'
        # Find the variable that contains the actual data
        data_var = None
        for var_name in mat_data.keys():
            if not var_name.startswith('__'):  # Skip metadata variables
                data_var = var_name
                break
        
        if data_var is None:
            logger.error(f"No data variable found in {file_path}")
            return None
        
        # Extract the data and convert to DataFrame
        data = mat_data[data_var]
        
        # CWRU data typically has multiple columns for different sensors
        # Column 0: Drive End accelerometer
        # Column 1: Fan End accelerometer
        # Column 2: Base accelerometer
        # Additional columns may include RPM, load, etc.
        
        column_names = ['DE_accelerometer', 'FE_accelerometer', 'BA_accelerometer']
        if data.shape[1] > 3:
            for i in range(3, data.shape[1]):
                column_names.append(f'channel_{i}')
        
        df = pd.DataFrame(data, columns=column_names)
        
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        
        # CWRU filenames contain information about the experiment
        # Example: '100_0.mat' - normal baseline at 100 Hz
        # Example: '105_1.mat' - 1 hp load at 105 Hz
        # Example: '118_7.mat' - 7 mils fault at 118 Hz
        
        # Extract RPM/frequency from filename
        rpm_info = filename.split('_')[0]
        if rpm_info.isdigit():
            df['rpm'] = int(rpm_info)
        
        # Extract fault info from directory name
        parent_dir = os.path.basename(os.path.dirname(file_path))
        if 'normal' in parent_dir.lower():
            df['fault_type'] = 'normal'
            df['fault_size'] = 0
        else:
            # For fault data, try to extract fault size
            fault_size = 0
            if 'fault' in parent_dir.lower():
                # Try to extract fault size from filename
                parts = filename.split('_')
                if len(parts) > 1 and parts[1].replace('.', '', 1).isdigit():
                    fault_size = float(parts[1])
            
            df['fault_type'] = 'fault'
            df['fault_size'] = fault_size
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def preprocess_data(df, sequence_length=100, step=20):
    """
    Preprocess the data for RUL prediction
    
    Args:
        df: DataFrame containing the raw data
        sequence_length: Length of sequences to create
        step: Step size for sliding window
        
    Returns:
        X: Features (sequences of vibration data)
        y: Target (RUL values)
    """
    # For RUL prediction, we need to simulate degradation
    # In CWRU dataset, we don't have actual RUL values since tests were not run-to-failure
    # We'll simulate RUL by assuming:
    # - Normal data has high RUL (100% remaining life)
    # - Fault data has lower RUL depending on fault size
    
    # Create RUL column
    if 'fault_type' in df.columns:
        if df['fault_type'].iloc[0] == 'normal':
            # Normal bearings have 100% remaining life
            df['RUL'] = 100
        else:
            # Faulty bearings have lower RUL based on fault size
            # Larger fault = lower RUL
            max_fault_size = 0.021  # 21 mils is the largest fault in CWRU dataset
            fault_size = df['fault_size'].iloc[0]
            
            # Scale RUL inversely with fault size (larger fault = lower RUL)
            if fault_size > 0:
                rul_percentage = max(0, 100 - (fault_size / max_fault_size) * 100)
            else:
                rul_percentage = 50  # Default for unknown fault size
                
            df['RUL'] = rul_percentage
    else:
        # If fault information is not available, assume middle RUL
        df['RUL'] = 50
    
    # Create sequences for time series prediction
    X = []
    y = []
    
    # Use only accelerometer data for features
    feature_cols = ['DE_accelerometer', 'FE_accelerometer', 'BA_accelerometer']
    
    # Create sliding windows
    for i in range(0, len(df) - sequence_length, step):
        X.append(df[feature_cols].iloc[i:i+sequence_length].values)
        y.append(df['RUL'].iloc[i+sequence_length-1])
    
    return np.array(X), np.array(y)

def process_cwru_dataset():
    """
    Process the CWRU Bearing Dataset for RUL prediction
    
    This function:
    1. Loads the .mat files from the dataset
    2. Converts them to pandas DataFrames
    3. Preprocesses the data for RUL prediction
    4. Saves the processed data to disk
    """
    data_dir = CONFIG["data_dir"]
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all .mat files in the dataset directories
    mat_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root, file))
    
    if not mat_files:
        logger.warning("No .mat files found in the dataset directories")
        return
    
    logger.info(f"Found {len(mat_files)} .mat files")
    
    # Process each .mat file
    all_X = []
    all_y = []
    
    for file_path in tqdm(mat_files, desc="Processing files"):
        df = load_mat_file(file_path)
        
        if df is not None and not df.empty:
            X, y = preprocess_data(df, 
                                  sequence_length=CONFIG["sequence_length"],
                                  step=CONFIG["sequence_length"]//5)
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                logger.info(f"Processed {file_path}: {X.shape[0]} sequences")
    
    if not all_X:
        logger.warning("No data was successfully processed")
        return
    
    # Combine all processed data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    logger.info(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")
    
    # Save the processed data
    timestamp = get_timestamp()
    np.save(os.path.join(processed_dir, f"X_data_{timestamp}.npy"), X_combined)
    np.save(os.path.join(processed_dir, f"y_data_{timestamp}.npy"), y_combined)
    
    # Also save the latest version with a fixed name for easier loading
    np.save(os.path.join(processed_dir, "X_data_latest.npy"), X_combined)
    np.save(os.path.join(processed_dir, "y_data_latest.npy"), y_combined)
    
    logger.info(f"Saved processed data to {processed_dir}")
    
    return X_combined, y_combined

if __name__ == "__main__":
    process_cwru_dataset()
