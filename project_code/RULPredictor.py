from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor  # Uncomment if using baseline model
from target_metrics_baseline import rul_score
from dataproc_evaluation import evaluate

class EnhancedRULPredictor:
    def __init__(self, rul_threshold=135):
        self.rul_threshold = rul_threshold
        self.models = {
            
            'baseline': Pipeline([
                ('scale', StandardScaler()),
                ('model', PoissonRegressor())
            ]),
            
            'gbm': Pipeline([
                ('scale', StandardScaler()),
                ('model', GradientBoostingRegressor(n_estimators=200//2, random_state=42))
            ])
        }
        self.best_model = None
        self.best_score = float('inf')
        
    def train_and_evaluate(self, X_train, y_train, groups, models_to_train = ['baseline'], cv_folds=5):
        """
        Train multiple models and select the best one
        """
        results = {}
        for name, model in self.models.items():
            print(f"Training {name} model {model}")
            if name not in models_to_train:
                continue
            cv_result = evaluate(
                model,
                X=X_train,
                y=y_train,
                groups=groups,
                cv=cv_folds
            )
            
            results[name] = cv_result
            mean_rmse = cv_result['rmse_test']  # Updated to match the returned key
            
            if mean_rmse < self.best_score:
                self.best_score = mean_rmse
                self.best_model = model
                
            print(f"{name} - RMSE: {mean_rmse:.2f}")
            
        print(f"\nBest model selected with RMSE: {self.best_score:.2f}")
        return results
    
    def predict(self, X_test):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train_and_evaluate first.")
        return self.best_model.predict(X_test)