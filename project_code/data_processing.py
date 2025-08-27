
import pandas as pd
import numpy as np

from utils_MyProject import addRUL
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_selector

from run_cusum_autoencoder import detect_anomalies

def drop_low_correlation_features(df, threshold=0.2, rul_column='RUL',  keep_columns=['unit', 'time_cycles']):
    """
    Drops features with absolute correlation to RUL below a threshold,
    while preserving specified operational columns.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing sensor data and RUL
        rul_column (str): Name of the RUL column (default: 'RUL')
        threshold (float): Correlation threshold (absolute value) for feature retention
        keep_columns (list): Columns to always preserve (e.g., operational settings)
        
    Returns:
        pd.DataFrame: DataFrame with low-correlation features removed
    """
    # add rul column
    df = addRUL(df)
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Get features with sufficient correlation to RUL
    relevant_features = corr_matrix[abs(corr_matrix[rul_column]) >= threshold].index
    droped_features = corr_matrix[abs(corr_matrix[rul_column]) < threshold].index
    
    op_settings = [col for col in df.columns if col.startswith('op_setting')]
    
    # Combine features to keep
    features_to_keep = keep_columns + op_settings + list(relevant_features) 
    features_to_keep = list(set(features_to_keep))  # Remove duplicates
    
    # Print droped features
    dropped_to_display = [feat for feat in droped_features if feat != 'unit']
    print(f"Features to drop: {dropped_to_display}")

    # Return filtered dataframe
    return df[features_to_keep]


def scale_by_engine(data, n_first_cycles=10, sensor_columns=None):
    """
    Scale sensor readings by subtracting initial average values per engine.
    
    Args:
        data : DataFrame
        n_first_cycles : int, default=10
            Number of initial cycles to use for baseline calculation
        sensor_columns : list, optional
            List of sensor columns to scale. If None, uses columns starting with 'sensor'
            
    Returns:
    DataFrame
        Scaled data with initial cycles removed
    """
    # # Identify columns
    # unit_col = 'unit' 
    # time_col = 'time_cycles' 
    
    # # sensor columns if not specified
    # if sensor_columns is None:
    #     sensor_columns = [col for col in data.columns if col.startswith('sensor')]
    
    # # Calculate initial averages per engine
    # init_avg = (data[data[time_col] <= n_first_cycles]
    #             .groupby(unit_col)[sensor_columns]
    #             .mean()
    #             .reset_index())
    
    # # Merge initial means to all rows
    # scaled_data = data.merge(init_avg, on=unit_col, suffixes=('', '_init'))
    # for sensor in sensor_columns:
    #     scaled_data[sensor] = scaled_data[sensor] - scaled_data[f'{sensor}_init']
    # scaled_data = scaled_data.drop(columns=scaled_data.filter(regex='_init$').columns)

    # # Sort to preserve order
    # scaled_data = scaled_data.sort_values([unit_col, time_col]).reset_index(drop=True)
    # return scaled_data

    if data.empty:
        return data
    
    unit_col = 'unit'
    time_col = 'time_cycles'
    
    if sensor_columns is None:
        sensor_columns = [col for col in data.columns if col.startswith('sensor')]
    
    # Get first n_first_cycles AVAILABLE IN THE DATA per engine
    init_avg = (
        data.groupby(unit_col)
        .apply(lambda x: x.nsmallest(n_first_cycles, time_col)[sensor_columns].mean())
        .reset_index()
    )
    
    # Merge and subtract baseline
    scaled_data = data.merge(init_avg, on=unit_col, suffixes=('', '_init'))
    for sensor in sensor_columns:
        scaled_data[sensor] = scaled_data[sensor] - scaled_data[f'{sensor}_init']
    
    return scaled_data.drop(columns=scaled_data.filter(regex='_init$').columns)
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def remove_low_variance_features(X, threshold=0, verbose=True):
    """
    Remove low variance features from a pandas DataFrame. 
    Parameters:
        X : pandas DataFrame
            Input data with features as columns
        threshold : float
            Features with variance below this threshold will be removed
        verbose : bool, default=True
            Whether to print which features were removed
        
    Returns:
    Df: Data with low variance features removed
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    
    # Get features to keep
    features_kept = selector.get_feature_names_out(X.columns)
    
    if verbose:
        # Get dropped features
        dropped_features = set(X.columns) - set(features_kept)
        if dropped_features:
            print(f'Removed low variance features: {sorted(dropped_features)}')
        else:
            print('No low variance features found')
    
    # Transform and return as DataFrame
    return pd.DataFrame(selector.transform(X), columns=features_kept)



def rolling_stat(df, window_size = 5):
    """
    Apply rolling statistics to reduce noise
    Parameters:
    df (pd.DataFrame): input dataframe
    window_size (int): size of the rolling window (default = 5)
    
    Returns:
    pd.DataFrame: dataframe with rolling statistics
    """
    get_ftr_names = make_column_selector(pattern='sensor')
    feature_columns = get_ftr_names(df)

    for sensor in feature_columns:
        df[f"rolling {sensor}"] = (df.groupby("unit")[sensor]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .reset_index(drop=True))
    return df


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_selector

def minmaxscaler(df):
    """
    Scale the time series using MinMaxScaler
    Parameters:
    df (pd.DataFrame): input dataframe
    Returns:
    pd.DataFrame: scaled dataframe
    """
    scaler = MinMaxScaler()
    get_ftr_names = make_column_selector(pattern='sensor')
    df_scaled = scaler.fit_transform(df[get_ftr_names(df)])
    print(np.shape(df_scaled))

    # Scale only the sensor columns
    sensor_scaled = scaler.fit_transform(df[get_ftr_names(df)])

    # Replace the sensor columns in the original DataFrame with the scaled values
    df_scaled = df.copy()
    df_scaled[get_ftr_names(df)] = sensor_scaled
    return df_scaled



def drop_features(df, sensors_to_drop):
    """
    Drops specified sensor columns from a DataFrame
    
    Args:
        df: Pandas DataFrame containing the data
        sensors_to_drop: List of sensor indexes to remove
        
    Returns:
        DataFrame with specified sensors removed
    """
    # Generate the column names to drop (format: 'sensor_1', 'sensor_2', etc.)
    cols_to_drop = [f'sensor_{num}' for num in sensors_to_drop 
                   if f'sensor_{num}' in df.columns]
    
    # Drop the columns and return the modified DataFrame
    return df.drop(columns=cols_to_drop)

def create_degradation_features(df, anomalies):
    """
    Create advanced degradation features using anomaly information
    """
    df_enhanced = df.copy()
    
    # Add anomaly-based features
    for unit in df_enhanced['unit'].unique():
        unit_mask = df_enhanced['unit'] == unit
        unit_data = df_enhanced[unit_mask].copy()
        
        if unit in anomalies:
            anomaly_cycle = anomalies[unit]  # Ensure this is a float, not a dictionary
            
            
            if isinstance(anomaly_cycle, (float, int)): # Check if anomaly_cycle is indeed a number (float or int)
                cycles_since = np.maximum(0, unit_data['time_cycles'].values - anomaly_cycle) # Distance from anomaly detection point
                cycles_to = np.maximum(0, anomaly_cycle - unit_data['time_cycles'].values)
                
                df_enhanced.loc[unit_mask, 'cycles_since_anomaly'] = cycles_since
                # df_enhanced.loc[unit_mask, 'cycles_to_anomaly'] = cycles_to
                df_enhanced.loc[unit_mask, 'post_anomaly'] = (unit_data['time_cycles'].values >= anomaly_cycle).astype(int) # Anomaly flag
                
                
                max_cycles_since = cycles_since.max() if cycles_since.max() > 0 else 1 # Degradation progression (non-linear)
                df_enhanced.loc[unit_mask, 'degradation_progression'] = np.power(cycles_since / (max_cycles_since + 1), 2)
    
            
             # For units without detected anomalies, use alternative features
        else:
           
            max_cycle = unit_data['time_cycles'].max()
            df_enhanced.loc[unit_mask, 'cycles_since_anomaly'] = 0
            # df_enhanced.loc[unit_mask, 'cycles_to_anomaly'] = max_cycle - unit_data['time_cycles'].values 
            df_enhanced.loc[unit_mask, 'post_anomaly'] = 0
            df_enhanced.loc[unit_mask, 'degradation_progression'] = 0
    
    return df_enhanced

def create_health_index_features(df):
    """
    Create health index features using autoencoder reconstruction error patterns
    """
    df_enhanced = df.copy()
    
    
    sensor_cols = [col for col in df.columns if 'sensor' in col] # Health index based on reconstruction error trend
    
    for unit in df_enhanced['unit'].unique():
        unit_mask = df_enhanced['unit'] == unit
        unit_data = df_enhanced[unit_mask].copy()
        
        
        # for window in [3, 5, 10]: # Calculate local health index (moving average of sensor deviations)
        for window in [5]:
            for sensor in sensor_cols:  # Use first 5 sensors for its faster 
                if sensor in unit_data.columns:
                    rolling_mean = unit_data[sensor].rolling(window=window, min_periods=1).mean()
                    rolling_std = unit_data[sensor].rolling(window=window, min_periods=1).std().fillna(0)
                    
                    df_enhanced.loc[unit_mask, f'{sensor}_trend_{window}'] = rolling_mean
                    df_enhanced.loc[unit_mask, f'{sensor}_volatility_{window}'] = rolling_std
    
    return df_enhanced

def create_operational_features(df):
    """
    Create features based on operational settings and their interactions
    """
    df_enhanced = df.copy()
    
    
    op_cols = [col for col in df.columns if 'op_setting' in col] #Operational setting interactions
    if len(op_cols) >= 2:
        df_enhanced['op_interaction_1_2'] = df_enhanced[op_cols[0]] * df_enhanced[op_cols[1]]
        if len(op_cols) >= 3:
            df_enhanced['op_interaction_1_3'] = df_enhanced[op_cols[0]] * df_enhanced[op_cols[2]]
            df_enhanced['op_interaction_2_3'] = df_enhanced[op_cols[1]] * df_enhanced[op_cols[2]]
    
    
    if len(op_cols) >= 1: #Operational regime classification (based on operational settings)
        try:
            df_enhanced['op_regime'] = pd.cut(df_enhanced[op_cols[0]], bins=3, labels=[0, 1, 2]).astype(float)
        except:
            df_enhanced['op_regime'] = pd.qcut(df_enhanced[op_cols[0]], q=3, labels=[0, 1, 2], duplicates='drop').astype(float) #If cut fails, create simple regime based on quantiles
    
    return df_enhanced


def estimate_test_anomalies(test_df, train_anomalies, health_parameters, train_df=None):
    """
    Estimate anomaly points for test engines based on training patterns
    """
    test_anomalies = {}
    
    # Calculate average anomaly ratio from training data
    if train_anomalies and train_df is not None:
        # Calculate actual ratios from training data
        anomaly_ratios = []
        for train_unit, anomaly_cycle in train_anomalies.items():
            train_unit_data = train_df[train_df['unit'] == train_unit]
            if len(train_unit_data) > 0:
                max_cycle = train_unit_data['time_cycles'].max()
                ratio = anomaly_cycle / max_cycle if max_cycle > 0 else 0.7
                anomaly_ratios.append(ratio)
        
        avg_ratio = np.mean(anomaly_ratios) if anomaly_ratios else 0.7
    else:
        avg_ratio = 0.7 #default
    
    
    for unit in test_df['unit'].unique(): # Estimate anomaly points for test units
        unit_data = test_df[test_df['unit'] == unit]
        max_cycle = unit_data['time_cycles'].max()
        estimated_anomaly = max(1, int(max_cycle * avg_ratio)) # Estimate anomaly point
        test_anomalies[unit] = estimated_anomaly
    
    return test_anomalies



## Enhanced Data Processing Pipeline

def enhanced_preprocessing_pipeline(train_df, test_df, dataset_name):
    """
    Enhanced preprocessing pipeline that leverages CUSUM-Autoencoder effectively
    """
    print(f"Processing {dataset_name}...")
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    #Remove low variance features
    train_processed = remove_low_variance_features(train_processed)

    # Apply same feature removal to test
    low_var_features = [col for col in test_processed.columns if col not in train_processed.columns and 'sensor' in col]
    test_processed = test_processed.drop(low_var_features, axis=1, errors='ignore')
    
    #Scale by engine 
    train_processed = scale_by_engine(train_processed)
    test_processed = scale_by_engine(test_processed)
    
    #CUSUM-Autoencoder anomaly detection
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(train_processed)
    
    #Normalize for anomaly detection
    scaler_anomaly = StandardScaler()
    train_normalized = train_processed.copy()
    train_normalized[features] = scaler_anomaly.fit_transform(train_processed[features])
    
    #Detect anomalies
    anomalies, health_parameters, history = detect_anomalies(train_normalized, healthy_cycles=20)  # Can be changed for 15 for more robust detection
    
    #Create enhanced features using anomaly information
    train_enhanced = create_degradation_features(train_processed, anomalies)
    train_enhanced = create_health_index_features(train_enhanced)
    train_enhanced = create_operational_features(train_enhanced)
    
    # For test data, we need to estimate anomaly points
    test_anomalies = estimate_test_anomalies(test_processed, anomalies, health_parameters)
    test_enhanced = create_degradation_features(test_processed, test_anomalies)
    test_enhanced = create_health_index_features(test_enhanced)
    test_enhanced = create_operational_features(test_enhanced)
    
    #Rolling statistics on enhanced features
    # train_enhanced = rolling_stat(train_enhanced)
    # test_enhanced = rolling_stat(test_enhanced)
    
    #Final normalization
    feature_cols = [col for col in train_enhanced.columns if col not in ['unit', 'time_cycles'] and train_enhanced[col].dtype in [np.float64, np.int64]]
    
    scaler_final = StandardScaler()
    train_enhanced[feature_cols] = scaler_final.fit_transform(train_enhanced[feature_cols])
    test_enhanced[feature_cols] = scaler_final.transform(test_enhanced[feature_cols])
    
    return train_enhanced, test_enhanced, anomalies, scaler_final

def enhanced_preprocessing_pipeline_V2(train_df, test_df, dataset_name):
    """
    Enhanced preprocessing pipeline that leverages CUSUM-Autoencoder effectively
    """
    print(f"Processing {dataset_name}...")
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    #Remove low variance features
    train_processed = remove_low_variance_features(train_processed)

    # Apply same feature removal to test
    low_var_features = [col for col in test_processed.columns if col not in train_processed.columns and 'sensor' in col]
    test_processed = test_processed.drop(low_var_features, axis=1, errors='ignore')
    
    #Scale by engine 
    train_processed = scale_by_engine(train_processed)
    test_processed = scale_by_engine(test_processed)
    
    #CUSUM-Autoencoder anomaly detection
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(train_processed)
    
    #Normalize for anomaly detection
    scaler_anomaly = StandardScaler()
    train_normalized = train_processed.copy()
    train_normalized[features] = scaler_anomaly.fit_transform(train_processed[features])
    
    test_normalized = test_processed.copy()
    test_normalized[features] = scaler_anomaly.transform(test_processed[features])
    
    #Detect anomalies
    anomalies, health_parameters, history = detect_anomalies(train_normalized, healthy_cycles=20)  # Can be changed for 15 for more robust detection
    test_anomalies, health_parameters_test, history_test = detect_anomalies(test_normalized, healthy_cycles=20)
    
    #Create enhanced features using anomaly information
    train_enhanced = create_degradation_features(train_processed, anomalies)
    train_enhanced = create_health_index_features(train_enhanced)
    # train_enhanced = create_operational_features(train_enhanced)
    
    # For test data, we need to estimate anomaly points
    test_anomalies_preAnomaly = estimate_test_anomalies(test_processed, anomalies, health_parameters)
    
    # in the test data, for each engine, we distinguish the case where the last cycle is post-anomaly or not
    # for each engine in test_anomalies, we check if the last cycle is post-anomaly 
    # whcih is equivalent to checking if the pos key in test_anomalies is not empty

    is_post_anomaly = {}
    for unit in test_processed['unit'].unique():
        unit_mask = test_processed['unit'] == unit
        unit_data = test_processed[unit_mask].copy()
        last_cycle = unit_data['time_cycles'].max()
        anomaly_info = test_anomalies.get(unit, {})
        anomaly_positions = anomaly_info['pos'] if isinstance(anomaly_info, dict) and 'pos' in anomaly_info else []
        
        if anomaly_positions and last_cycle >= anomaly_positions[0]:
            is_post_anomaly[unit] = True # if the last cycle is post-anomaly
            test_enhanced = create_degradation_features(test_processed, test_anomalies)
            test_enhanced = create_health_index_features(test_enhanced)
            # test_enhanced = create_operational_features(test_enhanced)
        else:
            is_post_anomaly[unit] = False
            test_enhanced = create_degradation_features(test_processed, test_anomalies_preAnomaly)
            test_enhanced = create_health_index_features(test_enhanced)
            # test_enhanced = create_operational_features(test_enhanced)
    
    
    # test_enhanced = create_degradation_features(test_processed, test_anomalies)
    # test_enhanced = create_health_index_features(test_enhanced, test_anomalies, health_parameters)
    # test_enhanced = create_operational_features(test_enhanced)
    
    #Rolling statistics on enhanced features
    train_enhanced = rolling_stat(train_enhanced)
    test_enhanced = rolling_stat(test_enhanced)
    
    #Final normalization
    feature_cols = [col for col in train_enhanced.columns if col not in ['unit', 'time_cycles'] and train_enhanced[col].dtype in [np.float64, np.int64]]
    
    scaler_final = StandardScaler()
    train_enhanced[feature_cols] = scaler_final.fit_transform(train_enhanced[feature_cols])
    test_enhanced[feature_cols] = scaler_final.transform(test_enhanced[feature_cols])
    
    return train_enhanced, test_enhanced, anomalies, scaler_final

def enhanced_preprocessing_pipeline_V2(train_df, test_df, dataset_name):
    """
    Enhanced preprocessing pipeline that leverages CUSUM-Autoencoder effectively
    """
    print(f"Processing {dataset_name}...")
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    #Remove low variance features
    train_processed = remove_low_variance_features(train_processed)

    # Apply same feature removal to test
    low_var_features = [col for col in test_processed.columns if col not in train_processed.columns and 'sensor' in col]
    test_processed = test_processed.drop(low_var_features, axis=1, errors='ignore')
    
    #Scale by engine 
    train_processed = scale_by_engine(train_processed)
    test_processed = scale_by_engine(test_processed)
    
    #CUSUM-Autoencoder anomaly detection
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(train_processed)
    
    #Normalize for anomaly detection
    scaler_anomaly = StandardScaler()
    train_normalized = train_processed.copy()
    train_normalized[features] = scaler_anomaly.fit_transform(train_processed[features])
    
    test_normalized = test_processed.copy()
    test_normalized[features] = scaler_anomaly.transform(test_processed[features])
    
    #Detect anomalies
    anomalies, health_parameters, history = detect_anomalies(train_normalized, healthy_cycles=20)  # Can be changed for 15 for more robust detection
    test_anomalies, health_parameters_test, history_test = detect_anomalies(test_normalized, healthy_cycles=20)
    
    #Create enhanced features using anomaly information
    train_enhanced = create_degradation_features(train_processed, anomalies)
    train_enhanced = create_health_index_features(train_enhanced)
    # train_enhanced = create_operational_features(train_enhanced)
    
    # For test data, we need to estimate anomaly points
    test_anomalies_preAnomaly = estimate_test_anomalies(test_processed, anomalies, health_parameters)
    
    # in the test data, for each engine, we distinguish the case where the last cycle is post-anomaly or not
    # for each engine in test_anomalies, we check if the last cycle is post-anomaly 
    # whcih is equivalent to checking if the pos key in test_anomalies is not empty

    is_post_anomaly = {}
    for unit in test_processed['unit'].unique():
        unit_mask = test_processed['unit'] == unit
        unit_data = test_processed[unit_mask].copy()
        last_cycle = unit_data['time_cycles'].max()
        anomaly_info = test_anomalies.get(unit, {})
        anomaly_positions = anomaly_info['pos'] if isinstance(anomaly_info, dict) and 'pos' in anomaly_info else []
        
        if anomaly_positions and last_cycle >= anomaly_positions[0]:
            is_post_anomaly[unit] = True # if the last cycle is post-anomaly
            test_enhanced = create_degradation_features(test_processed, test_anomalies)
            test_enhanced = create_health_index_features(test_enhanced)
            # test_enhanced = create_operational_features(test_enhanced)
        else:
            is_post_anomaly[unit] = False
            test_enhanced = create_degradation_features(test_processed, test_anomalies_preAnomaly)
            test_enhanced = create_health_index_features(test_enhanced)
            # test_enhanced = create_operational_features(test_enhanced)
    
    
    # test_enhanced = create_degradation_features(test_processed, test_anomalies)
    # test_enhanced = create_health_index_features(test_enhanced, test_anomalies, health_parameters)
    # test_enhanced = create_operational_features(test_enhanced)
    
    #Rolling statistics on enhanced features
    train_enhanced = rolling_stat(train_enhanced)
    test_enhanced = rolling_stat(test_enhanced)
    
    #Final normalization
    feature_cols = [col for col in train_enhanced.columns if col not in ['unit', 'time_cycles'] and train_enhanced[col].dtype in [np.float64, np.int64]]
    
    scaler_final = StandardScaler()
    train_enhanced[feature_cols] = scaler_final.fit_transform(train_enhanced[feature_cols])
    test_enhanced[feature_cols] = scaler_final.transform(test_enhanced[feature_cols])
    
    return train_enhanced, test_enhanced, anomalies, scaler_final


def enhanced_preprocessing_pipeline_V3(train_df, test_df, dataset_name):
    """
    Enhanced preprocessing pipeline that leverages CUSUM-Autoencoder effectively
    """
    print(f"Processing {dataset_name}...")
    print("the modif was loaded 6")
    
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    #Remove low variance features
    train_processed = remove_low_variance_features(train_processed)

    # Apply same feature removal to test
    low_var_features = [col for col in test_processed.columns if col not in train_processed.columns and 'sensor' in col]
    test_processed = test_processed.drop(low_var_features, axis=1, errors='ignore')
    
    # #Scale by engine 
    # train_processed = scale_by_engine(train_processed)
    # test_processed = scale_by_engine(test_processed)
    
    #CUSUM-Autoencoder anomaly detection
    get_ftr_names = make_column_selector(pattern='sensor')
    features = get_ftr_names(train_processed)
    
    #Normalize for anomaly detection
    scaler_anomaly = StandardScaler()
    train_normalized = train_processed.copy()
    train_normalized[features] = scaler_anomaly.fit_transform(train_processed[features])
    
    test_normalized = test_processed.copy()
    test_normalized[features] = scaler_anomaly.transform(test_processed[features])
    
    #Create enhanced features using anomaly information
    # train_enhanced = create_degradation_features(train_processed, anomalies)
    train_enhanced = create_health_index_features(train_processed)
    # train_enhanced = create_operational_features(train_enhanced)
    
    # # For test data, we need to estimate anomaly points
    # test_anomalies_preAnomaly = estimate_test_anomalies(test_processed, anomalies, health_parameters)
    
    # test_enhanced = create_degradation_features(test_processed, test_anomalies)
    test_enhanced = create_health_index_features(test_processed)
    # test_enhanced = create_operational_features(test_enhanced)
    
    #Rolling statistics on enhanced features
    train_enhanced = scale_by_engine(train_enhanced)
    test_enhanced = scale_by_engine(test_enhanced)
    # train_enhanced = rolling_stat(train_enhanced)
    # test_enhanced = rolling_stat(test_enhanced)
    
    #Final normalization
    feature_cols = [col for col in train_enhanced.columns if col not in ['unit', 'time_cycles'] and train_enhanced[col].dtype in [np.float64, np.int64]]
    
    scaler_final = StandardScaler()
    train_enhanced[feature_cols] = scaler_final.fit_transform(train_enhanced[feature_cols])
    test_enhanced[feature_cols] = scaler_final.transform(test_enhanced[feature_cols])
    
    return train_enhanced, test_enhanced, scaler_final