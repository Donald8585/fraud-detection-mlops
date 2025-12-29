import json
import numpy as np
import xgboost as xgb

def model_fn(model_dir):
    """Load the XGBoost model"""
    model = xgb.Booster()
    model.load_model(f"{model_dir}/xgboost-model")
    return model

def input_fn(request_body, content_type='application/json'):
    """Parse input data"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['features']).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run prediction"""
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    return prediction

def output_fn(prediction, accept='application/json'):
    """Format output"""
    if accept == 'application/json':
        return json.dumps({
            'fraud_probability': float(prediction[0]),
            'is_fraud': bool(prediction[0] > 0.5)
        }), accept
    raise ValueError(f"Unsupported accept type: {accept}")
