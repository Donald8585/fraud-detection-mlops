import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(data):
    """Preprocess fraud detection data"""
    # Separate features and target
    if 'Class' in data.columns:
        X = data.drop('Class', axis=1)
        y = data['Class']
    else:
        X = data
        y = None
    
    # Fill NaN values with column mean
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train, params=None):
    """Train fraud detection model"""
    try:
        import xgboost as xgb
        
        if params is None:
            params = {
                'max_depth': 5,
                'eta': 0.2,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        # Wrap model with custom predict method
        class XGBoostWrapper:
            def __init__(self, booster):
                self.booster = booster
            
            def predict(self, X):
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                return self.booster.predict(dtest)
        
        return XGBoostWrapper(model)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

