import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger

def generate_base64_image(fig):
    """
    Convert a matplotlib figure to a base64 encoded string
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string of the image
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_rul_distribution_plot(y_train, y_val, y_test):
    """
    Create a plot showing the distribution of RUL values
    
    Args:
        y_train: Training target values
        y_val: Validation target values
        y_test: Test target values
        
    Returns:
        Base64 encoded string of the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms of RUL values
    plt.hist(y_train, bins=20, alpha=0.5, label='Training')
    plt.hist(y_val, bins=20, alpha=0.5, label='Validation')
    plt.hist(y_test, bins=20, alpha=0.5, label='Test')
    
    plt.xlabel('Remaining Useful Life (RUL)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RUL Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_training_history_plot(history, model_name):
    """
    Create a plot showing the training history
    
    Args:
        history: Training history from model.fit()
        model_name: Name of the model
        
    Returns:
        Base64 encoded string of the plot
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'])
    plt.plot(history['val_mae'])
    plt.title(f'{model_name} Model MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_actual_vs_predicted_plot(y_true, y_pred, model_name):
    """
    Create a plot showing actual vs predicted RUL values
    
    Args:
        y_true: Actual RUL values
        y_pred: Predicted RUL values
        model_name: Name of the model
        
    Returns:
        Base64 encoded string of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'{model_name} Model: Actual vs Predicted RUL')
    plt.grid(True, alpha=0.3)
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_model_comparison_plot(model_results):
    """
    Create a plot comparing the performance of different models
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Base64 encoded string of the plot
    """
    models = list(model_results.keys())
    loss_values = [model_results[model]['loss'] for model in models]
    mae_values = [model_results[model]['mae'] for model in models]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(models, loss_values)
    plt.title('Model Comparison - Loss (MSE)')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(models, mae_values)
    plt.title('Model Comparison - MAE')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_feature_importance_plot(feature_names, importance_values):
    """
    Create a plot showing feature importance
    
    Args:
        feature_names: Names of features
        importance_values: Importance values for each feature
        
    Returns:
        Base64 encoded string of the plot
    """
    # Sort features by importance
    indices = np.argsort(importance_values)
    sorted_names = [feature_names[i] for i in indices]
    sorted_values = [importance_values[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_names, sorted_values)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_time_series_plot(time_data, sensor_data, sensor_name, prediction=None):
    """
    Create a plot showing time series data with optional prediction
    
    Args:
        time_data: Time points
        sensor_data: Sensor readings
        sensor_name: Name of the sensor
        prediction: Optional prediction data
        
    Returns:
        Base64 encoded string of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_data, sensor_data, label='Sensor Data')
    
    if prediction is not None:
        # Assuming prediction is a single RUL value
        failure_time = time_data[-1] + prediction
        plt.axvline(x=failure_time, color='r', linestyle='--', label=f'Predicted Failure: RUL={prediction:.2f}')
    
    plt.xlabel('Time')
    plt.ylabel(f'{sensor_name} Reading')
    plt.title(f'{sensor_name} Time Series Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_health_index_plot(time_data, health_index):
    """
    Create a plot showing the health index over time
    
    Args:
        time_data: Time points
        health_index: Health index values (0-100)
        
    Returns:
        Base64 encoded string of the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create a colormap for the health index
    cmap = plt.cm.RdYlGn
    colors = cmap(health_index / 100)
    
    # Plot the health index
    for i in range(len(time_data)-1):
        plt.plot(time_data[i:i+2], health_index[i:i+2], color=colors[i])
    
    plt.xlabel('Time')
    plt.ylabel('Health Index (%)')
    plt.title('Bearing Health Index Over Time')
    plt.ylim(0, 100)
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Health Status')
    
    plt.grid(True, alpha=0.3)
    
    img_str = generate_base64_image(plt)
    plt.close()
    
    return img_str

def create_sample_visualizations():
    """
    Create sample visualizations for testing
    
    Returns:
        Dictionary of base64 encoded images
    """
    # Create sample data
    np.random.seed(CONFIG["random_seed"])
    
    # Sample RUL distribution
    y_train = np.random.normal(70, 15, 1000)
    y_val = np.random.normal(65, 15, 200)
    y_test = np.random.normal(60, 15, 300)
    
    # Sample training history
    history = {
        'loss': np.exp(-np.linspace(0, 3, 50)) * 10 + 1,
        'val_loss': np.exp(-np.linspace(0, 2.5, 50)) * 10 + 1.5,
        'mae': np.exp(-np.linspace(0, 3, 50)) * 5 + 0.5,
        'val_mae': np.exp(-np.linspace(0, 2.5, 50)) * 5 + 0.8
    }
    
    # Sample actual vs predicted
    y_true = np.linspace(0, 100, 100)
    y_pred = y_true + np.random.normal(0, 10, 100)
    
    # Sample model comparison
    model_results = {
        'LSTM': {'loss': 5.2, 'mae': 2.1},
        'CNN': {'loss': 4.8, 'mae': 1.9},
        'CNN-LSTM': {'loss': 4.5, 'mae': 1.7}
    }
    
    # Sample feature importance
    feature_names = ['Vibration X', 'Vibration Y', 'Vibration Z', 'Temperature', 'RPM']
    importance_values = [0.35, 0.25, 0.20, 0.15, 0.05]
    
    # Sample time series
    time_data = np.linspace(0, 100, 1000)
    sensor_data = np.sin(time_data / 5) + np.random.normal(0, 0.1, 1000)
    
    # Sample health index
    health_index = 100 - time_data / 100 * 100
    
    # Create visualizations
    visualizations = {
        'rul_distribution': create_rul_distribution_plot(y_train, y_val, y_test),
        'training_history': create_training_history_plot(history, 'CNN-LSTM'),
        'actual_vs_predicted': create_actual_vs_predicted_plot(y_true, y_pred, 'CNN-LSTM'),
        'model_comparison': create_model_comparison_plot(model_results),
        'feature_importance': create_feature_importance_plot(feature_names, importance_values),
        'time_series': create_time_series_plot(time_data, sensor_data, 'Vibration', prediction=20),
        'health_index': create_health_index_plot(time_data, health_index)
    }
    
    return visualizations

if __name__ == "__main__":
    # Create sample visualizations
    visualizations = create_sample_visualizations()
    
    # Save visualizations to disk for testing
    vis_dir = os.path.join(CONFIG["data_dir"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save visualization names and their base64 strings to a file
    with open(os.path.join(vis_dir, "visualization_samples.txt"), 'w') as f:
        for name, img_str in visualizations.items():
            f.write(f"{name}: data:image/png;base64,{img_str[:30]}...\n")
    
    logger.info(f"Created sample visualizations and saved references to {vis_dir}")
