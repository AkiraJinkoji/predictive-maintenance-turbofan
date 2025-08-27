from sklearn.compose import make_column_selector
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from target_metrics_baseline import rul_score

def evaluate(model, X, y, groups, cv, 
             scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'], 
             n_jobs=None, 
             verbose=False):
    '''
    Evaluate a model with Cross-Validation
    Parameters:
    model: model to be evaluated
    X: feature matrix
    y: target vector
    groups: group vector
    cv: cross-validation object
    '''
    # cv_results = cross_validate(
    #     model, 
    #     X=X,
    #     y=y,
    #     groups=groups,
    #     scoring=scoring,
    #     cv=cv,
    #     return_train_score=True,
    #     return_estimator=True,
    #     n_jobs=n_jobs,
    #     verbose=verbose
    # )
    
    # for k, v in cv_results.items():
    #     if k.startswith('train_') or k.startswith('test_'):
    #         k_sp = k.split('_')
    #         print(f'[{k_sp[0]}] :: {" ".join(k_sp[2:])} : {np.abs(v.mean()):.2f} +- {v.std():.2f}')

    test_preds = []
    test_targets = []
    train_preds = []
    train_targets = []
    gkf = GroupKFold(n_splits=cv)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        # Test predictions
        y_pred_test = model.predict(X[test_idx])
        y_true_test = y[test_idx]
        test_preds.append(y_pred_test)
        test_targets.append(y_true_test)
        # Train predictions
        y_pred_train = model.predict(X[train_idx])
        y_true_train = y[train_idx]
        train_preds.append(y_pred_train)
        train_targets.append(y_true_train)

    # Concatenate all folds
    y_pred_test_all = np.concatenate(test_preds)
    y_true_test_all = np.concatenate(test_targets)
    y_pred_train_all = np.concatenate(train_preds)
    y_true_train_all = np.concatenate(train_targets)

    # Test metrics
    rmse_test = np.sqrt(mean_squared_error(y_true_test_all, y_pred_test_all))
    mae_test = mean_absolute_error(y_true_test_all, y_pred_test_all)
    mape_test = mean_absolute_percentage_error(y_true_test_all, y_pred_test_all)
    custom_score_test = rul_score(y_true_test_all, y_pred_test_all)

    # Train metrics
    rmse_train = np.sqrt(mean_squared_error(y_true_train_all, y_pred_train_all))
    mae_train = mean_absolute_error(y_true_train_all, y_pred_train_all)
    mape_train = mean_absolute_percentage_error(y_true_train_all, y_pred_train_all)
    custom_score_train = rul_score(y_true_train_all, y_pred_train_all)

    cv_results = {
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'mape_test': mape_test,
        'custom_score_test': custom_score_test,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'mape_train': mape_train,
        'custom_score_train': custom_score_train
    }
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Train MAE: {mae_train:.4f}")
    print(f"Train MAPE: {mape_train:.4f}")
    print(f"Train Custom RUL Score: {custom_score_train:.4f}")

    print(f"Validation RMSE: {rmse_test:.4f}")
    print(f"Validation MAE: {mae_test:.4f}")
    print(f"Validation MAPE: {mape_test:.4f}")
    print(f"Validation Custom RUL Score: {custom_score_test:.4f}")
   
    return cv_results


"""
Exemple of usage:
train_fd01_path = r'..\CMAPSSData\train_FD001.txt'
train = read_data(train_fd01_path)

dataset_name = 'FD001'
train, test, test_rul = read_dataset(dataset_name)
train.columns

get_ftr_names = make_column_selector(pattern='sensor')
data_to_train = train.copy()
data_to_train_name = 'train_FD001'

model = Pipeline([
    ('scale', StandardScaler()),
    ('model', PoissonRegression())
])

print(f"Evaluation by PoisonRegressor with the data: {data_to_train_name}")
cv_result = evaluate(
    model,
    X=train[get_ftr_names(data_to_train)].values, # No training using the operational setting
    y=calculate_RUL(data_to_train, upper_threshold=RUL_THRESHOLD), 
    groups=train['unit'], # ensures that all observations from the same engine (unit) stay together durtingh the Kfolding in either the training or test set across folds.
    cv=10 # Kfold splitter with 5 folds
)
"""