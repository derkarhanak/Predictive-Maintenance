import os
import requests
import zipfile
import logging
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, get_timestamp, logger

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    logger.info(f"Downloading file from {url} to {destination}")
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    
    logger.info(f"Download complete: {destination}")
    return destination

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a destination directory
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extraction complete: {extract_to}")
    return extract_to

def download_cwru_bearing_dataset():
    """
    Download the Case Western Reserve University Bearing Dataset
    
    This function downloads the dataset from the CWRU Bearing Data Center.
    The dataset contains vibration measurements from bearings with different fault conditions.
    """
    # Create data directory if it doesn't exist
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    
    # Define dataset URLs - these are direct download links to the CWRU dataset
    # Note: The actual CWRU website requires manual downloads, so we're using a mirror
    dataset_urls = {
        "normal_baseline": "https://engineering.case.edu/sites/default/files/Normal_Baseline_Data.zip",
        "12k_drive_end_bearing_fault": "https://engineering.case.edu/sites/default/files/12k_Drive_End_Bearing_Fault_Data.zip",
        "48k_drive_end_bearing_fault": "https://engineering.case.edu/sites/default/files/48k_Drive_End_Bearing_Fault_Data.zip",
        "fan_end_bearing_fault": "https://engineering.case.edu/sites/default/files/Fan_End_Bearing_Fault_Data.zip"
    }
    
    # Download and extract each dataset
    for dataset_name, url in dataset_urls.items():
        zip_path = os.path.join(CONFIG["data_dir"], f"{dataset_name}.zip")
        extract_dir = os.path.join(CONFIG["data_dir"], dataset_name)
        
        # Skip if already downloaded and extracted
        if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
            logger.info(f"Dataset {dataset_name} already exists, skipping download")
            continue
        
        try:
            # Download the dataset
            download_file(url, zip_path)
            
            # Extract the dataset
            os.makedirs(extract_dir, exist_ok=True)
            extract_zip(zip_path, extract_dir)
            
            # Remove the zip file to save space
            os.remove(zip_path)
            
            logger.info(f"Successfully downloaded and extracted {dataset_name}")
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {str(e)}")
            
            # Create a placeholder file with instructions for manual download
            with open(os.path.join(CONFIG["data_dir"], f"{dataset_name}_download_instructions.txt"), 'w') as f:
                f.write(f"The {dataset_name} dataset could not be automatically downloaded.\n")
                f.write("Please manually download the dataset from the CWRU Bearing Data Center:\n")
                f.write("https://engineering.case.edu/bearingdatacenter/download-data-file\n")
                f.write(f"After downloading, extract the files to: {extract_dir}\n")
    
    logger.info("Dataset download process completed")

if __name__ == "__main__":
    download_cwru_bearing_dataset()
