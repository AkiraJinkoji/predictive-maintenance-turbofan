import os

import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector


OP_SETTING_COLUMNS = ['op_setting_{}'.format(x) for x in range(1, 4)]
SENSOR_COLUMNS = ['sensor_{}'.format(x) for x in range(1, 22)]

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'CMAPSSData')


def read_data(filepath):
    '''
    Reads `filepath` as space separated file and returns pd.DataFrame
    '''
    col_names = ['unit', 'time_cycles'] + OP_SETTING_COLUMNS + SENSOR_COLUMNS
    return pd.read_csv(
        filepath,
        sep='\s+',
        header=None,
        names=col_names
    )

def read_dataset(dataset_name):
    '''
    Reads TRAIN, TEST and RUL datasets for specified dataset name

    Parameters
    ----------
    dataset_name : str, name of the dataset, e.g. 'FD001'

    Returns
    -------
    a tuple of (pd.DataFrame, pd.DataFrame, np.array) for TRAIN, TEST AND RUL
    datasets correspondingly
    '''
    TRAIN_FILE = os.path.join(DATA_DIR, f'train_{dataset_name}.txt')
    TEST_FILE = os.path.join(DATA_DIR, f'test_{dataset_name}.txt')
    TEST_RUL_FILE = os.path.join(DATA_DIR, f'RUL_{dataset_name}.txt')

    train_data = read_data(TRAIN_FILE)
    test_data = read_data(TEST_FILE)
    test_rul = np.loadtxt(TEST_RUL_FILE)

    return train_data, test_data, test_rul

    
def calculate_RUL(X, upper_threshold=None):
    '''
    Calculate Remaining Useful Life per `unit`

    Parameters
    ----------
    X : pd.DataFrame, with `unit` and `time_cycles` columns
    upper_threshold: int, limit maximum RUL valus, default is None

    Returns
    -------
    np.array with Remaining Useful Life values
    '''
    lifetime = X.groupby(['unit'])['time_cycles'].transform(max)
    rul = lifetime - X['time_cycles']

    if upper_threshold:
        rul = np.where(rul > upper_threshold, upper_threshold, rul)

    return rul

    

def drop_constant_value(dataframe):
    '''
    Function:
        - Deletes constant value columns in the data set.
        - A constant value is a value that is the same for all data in the data set.
        - A value is considered constant if the minimum (min) and maximum (max) values in the column are the same.
    Args:
        dataframe -> dataset to validate
    Returned value:
        dataframe -> dataset cleared of constant values
    '''
    
    constant_column = []

    # The process of finding a constant value by looking at the minimum and maximum values
    for col in dataframe.columns:
        min = dataframe[col].min()
        max = dataframe[col].max()

        # Append the column name if the min and max values are equal.
        if min == max:
            constant_column.append(col)

    
    dataframe.drop(columns=constant_column, inplace=True)
    return dataframe, col

def drop_uncorrelated(dataframe, threshold=0.2):
    """ drops columns that are not correlated with the target variable 'status'
        thus assume that 'status' has been set up

    Args:
        dataframe (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    # Show predictor that have correlation value >= threshold
    correlation = dataframe.corr()
    relevant_features = correlation[abs(correlation['status']) >= threshold]
    relevant_features['status']

    # Keep a relevant features (correlation value >= threshold)
    list_relevant_features = list(relevant_features.index[1:])

    # Applying feature selection
    dataframe = dataframe[list_relevant_features]
    
    return dataframe, list_relevant_features

def sensor_columns(df):
    get_ftr_names = make_column_selector(pattern='sensor')
    return get_ftr_names(df)

SENSOR_COLUMNS = ['sensor_{}'.format(x) for x in range(1, 22)]


def addRUL(df):
    """add the RUL column to the dataframe

    Args:
        df (pandas df): training data 

    Returns:
        pandas df: df with the column "RUL"
    """
    train_with_rul = df.copy()
    train_with_rul['RUL'] = calculate_RUL(train_with_rul)
    return train_with_rul

def convert_pd2txt(df, name):
    df.to_csv(name, index=False, sep='\t')
    return "dataframe converted as", name


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from target_metrics_baseline import rul_score

def split_train_validate(X, y, groups, test_size=0.2, random_state=42):
    """
    split the data between training and validation set
    Parameters:
        X (array-like): feature data
        y (array-like): target data
        groups (array-like): group data
        test_size (float, optional): proportion of data to include in the test set. Defaults to
        0.2.
        random_state (int, optional): seed for random number generator. Defaults to 42.
    Returns:

    """
    
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
    
    train_idx = np.where(np.isin(groups, train_groups))[0]
    test_idx = np.where(np.isin(groups, test_groups))[0]
    
    X_train, X_val = X[train_idx], X[test_idx]
    y_train, y_val = y[train_idx], y[test_idx]

    return X_train, y_train, X_val, y_val




def get_scores_model(model, X, y_true):
    """
    """
    y_pred = model.predict(X).flatten()
    # print(f"shape ypred:  {np.shape(y_pred)}, y_true:  {np.shape(y_true)}")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    custom_score = rul_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Custom RUL Score: {custom_score:.4f}")

    res = {}
    res['rmse'] = rmse
    res['mae'] = mae
    res['mape'] = mape
    res['custom_score'] = custom_score
    return res


def get_last_cycle_df(test_df):
    """
    Return a DataFrame with only the last cycle for each engine unit in the test data.
    Args:
        test_df (pd.DataFrame): Test data with 'unit' and 'time_cycles' columns.
    Returns:
        pd.DataFrame: DataFrame with only the last cycle for each engine.
    """
    # Find the last cycle for each unit
    idx = test_df.groupby('unit')['time_cycles'].idxmax()
    last_cycle_df = test_df.loc[idx].sort_values('unit').reset_index(drop=True)

    return last_cycle_df


    
def smart_data_splitting(train_df, anomalies, strategy='post_anomaly_weighted'):
    """
    Create training data that emphasizes the most informative part of degradation
    """
    
    if strategy == 'post_anomaly_weighted':
        # Weight post-anomaly data more heavily, but don't completely discard pre-anomaly
        train_weighted = []

        for unit in train_df['unit'].unique():
            unit_data = train_df[train_df['unit'] == unit].copy()

            if unit in anomalies and  anomalies[unit]['pos']:
                anomaly_cycle = anomalies[unit]['pos'][-1]

                # Pre-anomaly data (sample less frequently)
                pre_anomaly = unit_data[unit_data['time_cycles'] < anomaly_cycle]
                if len(pre_anomaly) > 10:
                    pre_anomaly_sampled = pre_anomaly.iloc[::3]  # Every 3rd sample
                else:
                    pre_anomaly_sampled = pre_anomaly

                # Post-anomaly data (use all)
                post_anomaly = unit_data[unit_data['time_cycles'] >= anomaly_cycle]

                # Combine
                unit_combined = pd.concat([pre_anomaly_sampled, post_anomaly])
            else:
                # For units without clear anomalies, use all data but weight later cycles more
                unit_combined = unit_data

            train_weighted.append(unit_combined)

        return pd.concat(train_weighted, ignore_index=True)

    elif strategy == 'full_sequence':
        return train_df

    
    """
    else:  # post_anomaly_only
        train_post_anomaly = []
        for unit in train_df['unit'].unique():
            unit_data = train_df[train_df['unit'] == unit]
            
            if unit in anomalies:
                anomaly_cycle = anomalies[unit]['pos'][-1] #last cycle dtetction ie failure cycle 
                post_anomaly = unit_data[unit_data['time_cycles'] >= anomaly_cycle]
                train_post_anomaly.append(post_anomaly)
            else:
                # Use last 50% of cycles for units without detected anomalies
                max_cycle = unit_data['time_cycles'].max()
                threshold_cycle = max_cycle * 0.5
                late_cycles = unit_data[unit_data['time_cycles'] >= threshold_cycle]
                train_post_anomaly.append(late_cycles)
        
        return pd.concat(train_post_anomaly, ignore_index=True)
        """