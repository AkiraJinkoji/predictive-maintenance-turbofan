import numpy as np
import tensorflow as tf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Conv1D, MaxPooling1D, BatchNormalization, GaussianNoise
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from target_metrics_baseline import rul_score


def build_lstm_model(input_shape, n_units=100, dropout_rate=0.2, architecture = 'PIML paper'):
    """
    Build and compile LSTM model
    Args:
        input_shape: tuple of (sequence_length, n_features)
        n_units: number of LSTM units
        dropout_rate: dropout rate
        architecture: 'PIML paper' or 'default', different designs of the NN
    Returns:
        compiled Keras model
    """
    if architecture == 'PIML paper':
        model = Sequential([
        LSTM(12, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        # Two-layer MLP (Dense layers)
        Dense(8, activation='relu'),  # First hidden layer
        Dense(4, activation='relu'),  # Second hidden layer

        Dense(1)  # Linear activation for regression
        ])

    if architecture == 'PIML paper light':
        model = Sequential([
        # Single LSTM layer with 12 units
        LSTM(12, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        # Two-layer MLP (Dense layers)
        Dense(8, activation='elu'),  # First hidden layer
        Dense(4, activation='relu'),  # Second hidden layer
        Dropout(0.2),
        # Dense(4, activation='elu'),  # Third hidden layer
        # Output layer for RUL prediction
        Dense(1)  # Linear activation for regression
        ])
    if architecture == 'default':
        model = Sequential([
            # Masking(mask_value=0., input_shape=input_shape),  # Mask padded values
            LSTM(n_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(n_units//2),  # Half the units in second layer
            Dropout(dropout_rate),
            Dense(1)  # Single output for RUL prediction
        ])

    if architecture == '1':
        model = Sequential([
        # Single LSTM layer with 12 units
        LSTM(10, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        # Two-layer MLP (Dense layers)
        Dense(8, activation='relu'),  # First hidden layer
        # Dense(4, activation='relu'),  # Second hidden layer
        ])
    if architecture == '2':
        model = Sequential([
        # Single LSTM layer with 12 units
        LSTM(12, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        # Two-layer MLP (Dense layers)
        Dense(8, activation='relu'),  # First hidden layer
        # Dense(4, activation='relu'),  # Second hidden layer
        
        # Output layer for RUL prediction
        Dense(1)  # Linear activation for regression
        ])
    if architecture == '3':    
        model = Sequential([
        LSTM(64//2, return_sequences=True, input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=0.02, l2=0.02), recurrent_regularizer=l2(0.02), recurrent_dropout=0.1),
        Dropout(0.3),
        LSTM(32//2, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(16//2, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
        ])
        
        
    if architecture == '4':
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(32, return_sequences=True),
            Dropout(0.3),
            LSTM(16),
            Dense(1)
        ])

            
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

def train_lstm_model(X, y, groups, architecture = 'default', test_size=0.2, random_state=42, epochs=10, batch_size=32):
    """
    Train LSTM model with group-aware train/test split
    Args:
        X: input data (sequence data)
        y: target data (RUL)
        groups: group labels
        architecture: the architecture of the LSTM model ('default' or 'PIML paper')
    Returns:
        trained model, history of training
    """
    # Build model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), architecture=architecture)
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train
    history = model.fit(
        X, y,
        # validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # After training
    y_pred_train = model.predict(X).flatten()
    rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
    mae_train = mean_absolute_error(y, y_pred_train)
    mape_train = mean_absolute_percentage_error(y, y_pred_train)
    custom_score_train = rul_score(y, y_pred_train)
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Train MAE: {mae_train:.4f}")
    print(f"Train MAPE: {mape_train:.4f}")
    print(f"Train Custom RUL Score: {custom_score_train:.4f}")

    return model, history