import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model
from sklearn.compose import make_column_selector
import matplotlib.pyplot as plt

from cusum import cusum
from autoencoder import Autoencoder


def detect_anomalies(df, healthy_cycles=15, latent_dim=16):
    """
    Detect anomalies in engine data using an LSTM autoencoder and CUSUM.
    
    Args:
        df (pd.df): Input datframe with engine data (the df may have to be scaled beforehand).
        healthy_cycles (int): Number of initial cycles assumed healthy.
        latent_dim (int): Dimension of the autoencoder bottleneck.
        
    Returns:
        anomalies (dict): Dictionary with MSE and anomaly positions per engine.
    """
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(df)
    engine_units = df['unit'].unique()
    engine_tot = len(engine_units)
    N, M, T = engine_tot, len(features), healthy_cycles

    # Prepare healthy data and full padded data
    X_train_healthy = np.zeros((N, T, M))
    engine_windows = []
    max_cycles = 0
    for i, engine_unit in enumerate(engine_units):
        engine_data = df[df['unit'] == engine_unit][features].values
        X_train_healthy[i] = engine_data[:T]
        engine_windows.append(engine_data)
        max_cycles = max(max_cycles, len(engine_data))
    padded_engines = [np.pad(engine, ((0, max_cycles - len(engine)), (0, 0))) for engine in engine_windows]
    X_train_full = np.array(padded_engines)

    # Train autoencoder
    autoencoder = Autoencoder(latent_dim, shape=(T, M))
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train_healthy, X_train_healthy, shuffle=True, epochs=20, batch_size=32, verbose=0)

    # Compute reconstruction errors
    window_size = T
    mse_per_engine = []
    for engine_data in X_train_full:
        windows = np.array([engine_data[t: t + window_size]
                            for t in range(max_cycles - window_size + 1)])
        reconstructions = autoencoder.predict(windows, batch_size=128)
        mse_engine = np.mean((reconstructions - windows) ** 2, axis=(1, 2))
        mse_per_engine.append(mse_engine)
    mse = np.array(mse_per_engine)

    # CUSUM anomaly detection
    healthy_mean = np.mean(mse[:, :T])
    healthy_std = np.std(mse[:, :T])
    # threshold = 3 * healthy_std    # 3 standard deviations
    threshold = 3

    print(f"Healthy Mean: {healthy_mean}, Healthy Std: {healthy_std}, Threshold: {threshold}")

    anomalies = {}
    for i, engine_unit in enumerate(engine_units):
        num_actual_windows = len(engine_windows[i]) - window_size + 1
        engine_mse = mse[i, :num_actual_windows]
        print(f"Engine {engine_unit}: MSE values: {engine_mse}")  # Print MSE values for each engine
        change_inds_pos = cusum(engine_mse, healthy_mean, threshold=threshold, k=0.7)
        print(f"Engine {engine_unit}: Detected Anomalies: {change_inds_pos}")  # Print anomaly positions
        anomalies[engine_unit] = {
            'mse': engine_mse,
            'pos': change_inds_pos,#anomaly location (at cycle indicies)
        }
    health_parameters = {}
    health_parameters['healthy_mean'] = healthy_mean
    health_parameters['healthy_std'] = healthy_std
    health_parameters['threshold'] = threshold
    return anomalies, health_parameters, history

def get_clipped_sequences(df, anomalies):
    """
    Clip each engine's data at the first detected anomaly and pad sequences.
    Args:
        df (df): Original dataframe.
        anomalies (dict): Output from anomaly detection, with 'pos' for each engine.
        
    Returns:
        clipped_df (pd.DataFrame): DataFrame of post-anomaly data.
        X_train_clipped (np.ndarray): Padded array of post-anomaly sequences.
        pre_anomaly_df (pd.DataFrame): DataFrame of pre-anomaly data.
        X_train_pre_anomaly (np.ndarray): Padded array of pre-anomaly sequences.
    """
    clipped_rows = []
    pre_anomaly_rows = []
    engine_units = df['unit'].unique()
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(df)

    for engine_unit in engine_units:
        engine_data = df[df['unit'] == engine_unit].copy()  # keep all columns
        anomaly_inds = anomalies[engine_unit]['pos']
        if len(anomaly_inds) > 0:
            first_anomaly_cycle = anomaly_inds[0]
            clipped_data = engine_data.iloc[first_anomaly_cycle:].copy()
            pre_anomaly_data = engine_data.iloc[:first_anomaly_cycle].copy()
            print(f"Engine {engine_unit}: Clipped at cycle {first_anomaly_cycle}")  # Debug clipping point
        else:
            clipped_data = engine_data.copy()
            pre_anomaly_data = pd.DataFrame(columns=engine_data.columns)  # Empty DataFrame
            print(f"Engine {engine_unit}: No anomaly detected, using full data")
        clipped_rows.append(clipped_data)
        pre_anomaly_rows.append(pre_anomaly_data)

    # Concatenate and preserve columns/order
    clipped_df = pd.concat(clipped_rows, axis=0).reset_index(drop=True)
    clipped_df = clipped_df[df.columns]
    pre_anomaly_df = pd.concat(pre_anomaly_rows, axis=0).reset_index(drop=True)
    pre_anomaly_df = pre_anomaly_df[df.columns]

    # pad sensor features only (post-anomaly)
    clipped_sequences = []
    for engine_unit in engine_units:
        engine_data = clipped_df[clipped_df['unit'] == engine_unit][features].values
        clipped_sequences.append(engine_data)
    max_clipped_cycles = max(seq.shape[0] for seq in clipped_sequences)
    X_train_clipped = np.array([
        np.pad(seq, ((0, max_clipped_cycles - seq.shape[0]), (0, 0)))
        for seq in clipped_sequences
    ])

    # pad sensor features only (pre-anomaly)
    pre_anomaly_sequences = []
    for engine_unit in engine_units:
        engine_data = pre_anomaly_df[pre_anomaly_df['unit'] == engine_unit][features].values
        pre_anomaly_sequences.append(engine_data)
    max_pre_anomaly_cycles = max(seq.shape[0] for seq in pre_anomaly_sequences) if pre_anomaly_sequences else 0
    X_train_pre_anomaly = np.array([
        np.pad(seq, ((0, max_pre_anomaly_cycles - seq.shape[0]), (0, 0)))
        for seq in pre_anomaly_sequences
    ]) if max_pre_anomaly_cycles > 0 else np.empty((len(engine_units), 0, len(features)))

    print(f"Max Clipped Cycles: {max_clipped_cycles}")
    print(f"Max Pre-Anomaly Cycles: {max_pre_anomaly_cycles}")
    print(f"Clipped Data Shape: {X_train_clipped.shape}") 
    print(f"Pre-Anomaly Data Shape: {X_train_pre_anomaly.shape}")
    return clipped_df, X_train_clipped, pre_anomaly_df, X_train_pre_anomaly



def get_clipped_sequences_forTest(df, anomalies):
    """
    Clip each engine's data at the first detected anomaly and pad sequences.
    Args:
        df (df): Original dataframe.
        anomalies (dict): Output from anomaly detection, with 'pos' for each engine.
        
    Returns:
        clipped_sequences (dict): Key: engine_unit, Value: clipped 2D np array.
        X_train_clipped (np.ndarray): Padded array of clipped sequences <-- easily usable for LSTM etc....
    """
    clipped_rows = []
    engine_units = df['unit'].unique()
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(df)

    for engine_unit in engine_units:
        engine_data = df[df['unit'] == engine_unit].copy()  # keep all columns
        anomaly_inds = anomalies[engine_unit]['pos']
        if len(anomaly_inds) > 0:
            first_anomaly_cycle = anomaly_inds[0]
            clipped_data = engine_data.iloc[first_anomaly_cycle:].copy()
            print(f"Engine {engine_unit}: Clipped at cycle {first_anomaly_cycle}")  # Debug clipping point
        else:
            clipped_data = engine_data.copy()
            print(f"Engine {engine_unit}: No anomaly detected, using full data")
        clipped_rows.append(clipped_data)

    # Concatenate and preserve columns/order
    clipped_df = pd.concat(clipped_rows, axis=0).reset_index(drop=True)
    clipped_df = clipped_df[df.columns]

    # pad sensor features only
    clipped_sequences = []
    for engine_unit in engine_units:
        engine_data = clipped_df[clipped_df['unit'] == engine_unit][features].values
        clipped_sequences.append(engine_data)
    max_clipped_cycles = max(seq.shape[0] for seq in clipped_sequences)
    # print(f"Max Clipped Cycles: {max_clipped_cycles}")
    X_train_clipped = np.array([
        np.pad(seq, ((0, max_clipped_cycles - seq.shape[0]), (0, 0)))
        for seq in clipped_sequences
    ])

    # print(f"Clipped Data Shape: {X_train_clipped.shape}") 
    return clipped_df, X_train_clipped



def plot_anomaly_results(anomalies, engine_unit, healthy_mean, threshold, history):
    """
    Plot reconstruction error and training loss for a given engine.
    
    Args:
        anomalies (dict): Output from anomaly detection, with 'mse' and 'pos' for each engine.
        engine_unit (int): Engine unit to plot.
        healthy_mean (float): Mean MSE for healthy cycles.
        threshold (float): Threshold for anomaly detection.
        history: Keras training history object.
    """

    # Reconstruction Error Plot
    plt.plot(anomalies[engine_unit]['mse'], label='Reconstruction Error')
    plt.axhline(y=healthy_mean + threshold, color='r', linestyle='--', label='Threshold')
    plt.scatter(
        anomalies[engine_unit]['pos'],
        [anomalies[engine_unit]['mse'][j] for j in anomalies[engine_unit]['pos']],
        color='red', label='Anomalies'
    )
    plt.xlabel('Cycle Number')
    plt.ylabel('MSE')
    plt.title(f'Engine {engine_unit}: Anomaly Detection via CUSUM')
    plt.legend()
    plt.show()

    # Training Loss Plot
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()


# def plot_engine_degradation(engine_index, sensor_index, 
#                             X_train_healthy, X_train_full, X_train_clipped, 
#                             anomalies, engine_units, window_size=30):
    
#     engine_unit = engine_units[engine_index]
#     healthy_signal = X_train_healthy[engine_index, :, sensor_index]
#     full_signal = X_train_full[engine_index, :, sensor_index]
#     clipped_signal = X_train_clipped[engine_index, :, sensor_index]

#     anomaly_windows = anomalies[engine_unit]['pos']

#     if len(anomaly_windows) > 0:
#         anomaly_cycle = anomaly_windows[0] #+ window_size - 1
        
#     else:
#         anomaly_cycle = None

#     plt.figure(figsize=(12, 6))
#     plt.plot(full_signal, label='Full Signal (padded)', color='gray', linewidth=1)
#     non_zero_mask = clipped_signal != 0
#     non_zero_clipped_signal = clipped_signal[non_zero_mask]

#     plt.plot(range(window_size), healthy_signal, label='Healthy Start (first 30 cycles)', color='green')
#     if anomaly_cycle is not None:
#         plt.plot(range(anomaly_cycle, anomaly_cycle + len(non_zero_clipped_signal)), 
#                  non_zero_clipped_signal, label='Clipped Post-Anomaly Signal', color='red')
#         plt.axvline(anomaly_cycle, color='black', linestyle='--', label='Anomaly Detected')
#     else:
#         plt.plot(non_zero_clipped_signal, label='Clipped Signal (no anomaly)', color='blue')
    
#     plt.title(f"Engine {engine_unit} – Sensor {sensor_index + 1}")
#     plt.xlabel("Cycle")
#     plt.ylabel("Sensor Reading")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_engine_degradation_from_df(df, anomalies, engine_index, sensor_index, window_size=30):
    """
    Plot engine degradation for a given engine and sensor using only the original DataFrame and anomalies.
    Args:
        df (pd.df): Original training data.
        anomalies (dict): Output from anomaly detection.
        engine_index (int): Index of engine in df['unit'].unique().
        sensor_index (int): Index of sensor column (0 based).
        window_size (int): Number of healthy cycles (default 30).
    """

    engine_units = df['unit'].unique()
    engine_unit = engine_units[engine_index]
    print(engine_unit)
    sensor_cols = [col for col in df.columns if col.startswith('sensor')]
    sensor_col = sensor_cols[sensor_index]

    # Get all cycles for this engine
    engine_data = df[df['unit'] == engine_unit].reset_index(drop=True)
    healthy_signal = engine_data[sensor_col].values[:window_size]
    full_signal = engine_data[sensor_col].values
    anomaly_windows = anomalies[engine_unit]['pos']

    # Clip at first anomaly
    if len(anomaly_windows) > 0:
        anomaly_cycle = anomaly_windows[0]
        clipped_signal = engine_data[sensor_col].values[anomaly_cycle:]
    else:
        anomaly_cycle = None
        clipped_signal = engine_data[sensor_col].values

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(full_signal, label='Full Signal', color='gray', linewidth=1)
    plt.plot(range(window_size), healthy_signal, label='Healthy Start (first 30 cycles)', color='green')
    if anomaly_cycle is not None:
        plt.plot(range(anomaly_cycle, anomaly_cycle + len(clipped_signal)),
                 clipped_signal, label='Clipped Post-Anomaly Signal', color='red')
        plt.axvline(anomaly_cycle, color='black', linestyle='--', label='Anomaly Detected')
    else:
        plt.plot(clipped_signal, label='Clipped Signal (no anomaly)', color='blue')
    plt.title(f"Engine {engine_unit} – Sensor {sensor_index+1}")
    plt.xlabel("Cycle")
    plt.ylabel("Sensor Reading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def set_pre_anomaly_rul(df, anomalies, upper_threshold=135):
    """
    For each engine, set the RUL for all cycles before the anomaly to the RUL at the anomaly cycle.
    If no anomaly, use the normal RUL calculation.
    Returns a numpy array of RULs for the input DataFrame.
    """
    rul_labels = np.zeros(len(df))
    for unit in df['unit'].unique():
        unit_mask = df['unit'] == unit
        unit_data = df[unit_mask]
        anomaly_pos = anomalies.get(unit, {}).get('pos', [])
        if anomaly_pos:
            anomaly_cycle = anomaly_pos[0]
            # RUL at anomaly = min(upper_threshold, max_cycle - anomaly_cycle)
            max_cycle = unit_data['time_cycles'].max()
            anomaly_rul = min(upper_threshold, max_cycle - anomaly_cycle)
            # Set this RUL for all cycles before anomaly
            pre_anomaly_mask = unit_data['time_cycles'] < anomaly_cycle
            rul_labels[unit_mask & pre_anomaly_mask] = anomaly_rul
            # For cycles after anomaly, set to zero (or use normal RUL if needed)
            rul_labels[unit_mask & ~pre_anomaly_mask] = 0
        else:
            # No anomaly, use normal RUL
            cycles = unit_data['time_cycles']
            max_cycle = cycles.max()
            rul = np.clip(max_cycle - cycles, 0, upper_threshold)
            rul_labels[unit_mask] = rul
    return rul_labels

# Example usage:
# y_pre_anomaly = set_pre_anomaly_rul(train1a, anomalies, upper_threshold