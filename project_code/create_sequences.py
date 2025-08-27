import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.compose import make_column_selector

from utils_MyProject import addRUL

# data preparation for lstm
def create_sequences(df, sequence_length):
    """
    Convert DataFrame to LSTM-ready sequences
    
    Args:
        df: DataFrame containing all data, without RUL
        sequence_length: number of timesteps per sample
    
    Returns:
        X: 3D array of shape (n_samples, sequence_length, n_features)
        y: 1D array of shape (n_samples,)
        groups: array indicating which unit each sample belongs to
    """
    X, y, groups = [], [], []
    
    # identify feature columns
    get_ftr_names = make_column_selector(pattern='sensor')
    feature_columns = get_ftr_names(df)
    feature_array = df[feature_columns].values
    
    # add target column to the data
    if "RUL" not in df.columns:
        df = addRUL(df)
    
    # take the max time cycles for all the engines
    max_cycle_per_engine = df.groupby('unit')['time_cycles'].max().reset_index()
    max_seq_length = int(max_cycle_per_engine['time_cycles'].max())
    
    # take the minimum life time for all the engines
    max_cycle_per_engine = df.groupby('unit')['time_cycles'].max().reset_index()
    total_cycles = max_cycle_per_engine.sum()
    # print(f"sum of all the life of engines: {total_cycles}")
    min_seq_length = max_cycle_per_engine['time_cycles'].min()
    # print(f"max cycle:  {max_seq_length}, min cycle:  {min_seq_length}")
    
    # check if 
    assert min_seq_length >= sequence_length, "Sequence length should be smaller than or equal to the minimum lifetime of all engines (e.g., 100)"

    # extract features data by group of unit
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit]
        unit_features = unit_data[feature_columns].values
        unit_target = unit_data['RUL'].values
        
        # zero-padding for shorter engines
        if len(unit_features) < max_seq_length:
            pad_length = max_seq_length - len(unit_features)
            # print(f"pad length:  {pad_length}")
            # print(f"max seq:  {max_seq_length}")
            # print(f"len(unit_features):  {len(unit_features)}")
            unit_features = np.pad(unit_features, ((0,pad_length),(0,0)), 'constant')
            # unit_target = np.pad(unit_target, (0,pad_length), 'constant')    
        
        for i in range(len(unit_data) - sequence_length + 1):
            X.append(unit_features[i:i+sequence_length])  # Automatically becomes 3D
            y.append(unit_target[i+sequence_length-1])  # Last RUL in window
            groups.append(unit)
            
            # je crois que sequence length est la longeur dans le time_cyle vers le quel on revienrait en arriere
    
    return np.array(X), np.array(y), np.array(groups)



def create_test_sequences(test_df, sequence_length, rul_df=None):
    """
    Prepare test data for LSTM evaluation
    
    Args:
        test_df: DataFrame with test sensor data (not run to failure)
        sequence_length: same as used in training
        rul_df: Optional DataFrame with true RUL for each engine
        
    Returns:
        X_test: 3D array (n_engines, sequence_length, n_features)
        y_test: 1D array of true RUL values
        engine_ids: list of engine IDs
    """
    # Get feature columns
    feature_columns = make_column_selector(pattern='sensor')(test_df)
    
    X_test, y_test, engine_ids = [], [], []
    
    for unit in test_df['unit'].unique():
        unit_data = test_df[test_df['unit'] == unit]
        
        # Take last sequence_length cycles
        unit_features = unit_data[feature_columns].values[-sequence_length:]
        
        # check if the seq length is not larger than the test set length for each engine
        max_cycle_per_engine = test_df.groupby('unit')['time_cycles'].max().reset_index()
        min_seq_length = max_cycle_per_engine['time_cycles'].min()
        # print(f"unit {unit} min seq length:  {min_seq_length}")
        # print(f"unit {unit}: unit_features shape = {unit_features.shape}")
        # assert min_seq_length >= sequence_length, "Sequence length should be smaller than or equal to the minimum lifetime of all engines (e.g., 100)"
            
        # Pad at the beginning if not enough cycles
        if unit_features.shape[0] < sequence_length:
            pad_length = sequence_length - unit_features.shape[0]
            unit_features = np.pad(unit_features, ((pad_length, 0), (0, 0)), 'constant')
        else:
            unit_features = unit_features[-sequence_length:]

        X_test.append(unit_features)
        engine_ids.append(unit)
    
    X_test = np.array(X_test)
    
    # If RUL values are provided
    if rul_df is not None:
        y_test = rul_df.set_index('unit').loc[engine_ids, 'RUL'].values
        return X_test, y_test, engine_ids
    
    return X_test, engine_ids