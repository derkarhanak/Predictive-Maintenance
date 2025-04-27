import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger, get_timestamp

def create_lstm_model(input_shape, output_units=1):
    """
    Create an LSTM model for RUL prediction
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        output_units: Number of output units (1 for regression)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_units)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created LSTM model with input shape {input_shape}")
    return model

def create_cnn_model(input_shape, output_units=1):
    """
    Create a CNN model for RUL prediction
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        output_units: Number of output units (1 for regression)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(output_units)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created CNN model with input shape {input_shape}")
    return model

def create_cnn_lstm_model(input_shape, output_units=1):
    """
    Create a hybrid CNN-LSTM model for RUL prediction
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        output_units: Number of output units (1 for regression)
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN branch
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Dropout(0.2)(cnn)
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    
    # LSTM branch
    lstm = LSTM(64, return_sequences=True)(inputs)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(32)(lstm)
    
    # Combine branches
    combined = concatenate([Flatten()(cnn), lstm])
    
    # Output layers
    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.2)(x)
    outputs = Dense(output_units)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"Created CNN-LSTM hybrid model with input shape {input_shape}")
    return model

def load_training_data():
    """
    Load the training, validation, and test data
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    split_data_dir = os.path.join(CONFIG["data_dir"], "split")
    
    # Check if split data exists
    if not os.path.exists(split_data_dir):
        logger.error("Split data not found. Please run the data preparation script first.")
        return None, None, None, None, None, None
    
    # Load the split data
    X_train = np.load(os.path.join(split_data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(split_data_dir, "X_val.npy"))
    X_test = np.load(os.path.join(split_data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(split_data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(split_data_dir, "y_val.npy"))
    y_test = np.load(os.path.join(split_data_dir, "y_test.npy"))
    
    logger.info(f"Loaded training data: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"Loaded validation data: X_val={X_val.shape}, y_val={y_val.shape}")
    logger.info(f"Loaded test data: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=CONFIG["epochs"]):
    """
    Train a model and save the best version
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_name: Name of the model for saving
        epochs: Number of training epochs
        
    Returns:
        Training history
    """
    # Create callbacks
    model_dir = CONFIG["models_dir"]
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = get_timestamp()
    model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.h5")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG["early_stopping_patience"],
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    # Train the model
    logger.info(f"Training {model_name} model for {epochs} epochs")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, f"{model_name}_final.h5"))
    
    # Also save as latest model for easy loading
    model.save(os.path.join(model_dir, "latest_model.h5"))
    
    logger.info(f"Model training completed. Best model saved to {model_path}")
    
    return history

def plot_training_history(history, model_name):
    """
    Plot the training history
    
    Args:
        history: Training history from model.fit()
        model_name: Name of the model
    """
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(CONFIG["data_dir"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title(f'{model_name} Model MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(vis_dir, f"{model_name}_training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training history plot to {vis_dir}")

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        
    Returns:
        Evaluation metrics
    """
    # Evaluate the model
    logger.info(f"Evaluating {model_name} model on test data")
    metrics = model.evaluate(X_test, y_test, verbose=1)
    
    # Log metrics
    metrics_dict = dict(zip(model.metrics_names, metrics))
    logger.info(f"Test metrics: {metrics_dict}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(CONFIG["data_dir"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'{model_name} Model: Actual vs Predicted RUL')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(vis_dir, f"{model_name}_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved actual vs predicted plot to {vis_dir}")
    
    return metrics_dict, y_pred

def train_and_evaluate_models():
    """
    Train and evaluate multiple models for RUL prediction
    
    This function:
    1. Loads the training data
    2. Creates and trains LSTM, CNN, and CNN-LSTM models
    3. Evaluates the models on test data
    4. Compares the models and selects the best one
    
    Returns:
        Best model and its metrics
    """
    # Load the training data
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data()
    
    if X_train is None:
        logger.error("Failed to load training data")
        return None, None
    
    # Get input shape from training data
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Dictionary to store model results
    model_results = {}
    
    # Train and evaluate LSTM model
    lstm_model = create_lstm_model(input_shape)
    lstm_history = train_model(lstm_model, X_train, y_train, X_val, y_val, "lstm")
    plot_training_history(lstm_history, "LSTM")
    lstm_metrics, lstm_preds = evaluate_model(lstm_model, X_test, y_test, "LSTM")
    model_results["LSTM"] = lstm_metrics
    
    # Train and evaluate CNN model
    cnn_model = create_cnn_model(input_shape)
    cnn_history = train_model(cnn_model, X_train, y_train, X_val, y_val, "cnn")
    plot_training_history(cnn_history, "CNN")
    cnn_metrics, cnn_preds = evaluate_model(cnn_model, X_test, y_test, "CNN")
    model_results["CNN"] = cnn_metrics
    
    # Train and evaluate CNN-LSTM model
    cnn_lstm_model = create_cnn_lstm_model(input_shape)
    cnn_lstm_history = train_model(cnn_lstm_model, X_train, y_train, X_val, y_val, "cnn_lstm")
    plot_training_history(cnn_lstm_history, "CNN-LSTM")
    cnn_lstm_metrics, cnn_lstm_preds = evaluate_model(cnn_lstm_model, X_test, y_test, "CNN-LSTM")
    model_results["CNN-LSTM"] = cnn_lstm_metrics
    
    # Compare models and select the best one
    best_model_name = min(model_results, key=lambda k: model_results[k]['loss'])
    best_metrics = model_results[best_model_name]
    
    logger.info(f"Model comparison:")
    for model_name, metrics in model_results.items():
        logger.info(f"  {model_name}: Loss={metrics['loss']:.4f}, MAE={metrics['mae']:.4f}")
    
    logger.info(f"Best model: {best_model_name} with Loss={best_metrics['loss']:.4f}, MAE={best_metrics['mae']:.4f}")
    
    # Save the best model as the production model
    if best_model_name == "LSTM":
        best_model = lstm_model
    elif best_model_name == "CNN":
        best_model = cnn_model
    else:  # CNN-LSTM
        best_model = cnn_lstm_model
    
    best_model.save(os.path.join(CONFIG["models_dir"], "production_model.h5"))
    logger.info(f"Saved best model ({best_model_name}) as production model")
    
    return best_model, best_metrics

if __name__ == "__main__":
    train_and_evaluate_models()
