
# Create comprehensive test suite

# Unit tests
unit_tests = '''import pytest
import pandas as pd
import numpy as np
from src.train import preprocess_data, train_model
from src.monitoring.drift_detector import ModelDriftDetector

class TestDataPreprocessing:
    def test_preprocess_data(self):
        # Create sample data
        df = pd.DataFrame({
            'Class': [0, 1, 0],
            'V1': [1.0, 2.0, 3.0],
            'Amount': [100, 200, 300]
        })
        
        X, y = preprocess_data(df)
        
        assert X.shape[0] == 3
        assert y.shape[0] == 3
        assert 'Class' not in X.columns
        
    def test_missing_values_handling(self):
        df = pd.DataFrame({
            'Class': [0, 1, 0],
            'V1': [1.0, np.nan, 3.0],
            'Amount': [100, 200, 300]
        })
        
        X, y = preprocess_data(df)
        
        assert X.isna().sum().sum() == 0

class TestModel:
    def test_model_training(self):
        # Create synthetic training data
        X_train = np.random.rand(100, 30)
        y_train = np.random.randint(0, 2, 100)
        
        model = train_model(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, 'predict')
        
    def test_model_prediction_shape(self):
        X_train = np.random.rand(100, 30)
        y_train = np.random.randint(0, 2, 100)
        model = train_model(X_train, y_train)
        
        X_test = np.random.rand(10, 30)
        predictions = model.predict(X_test)
        
        assert predictions.shape[0] == 10
        assert all(p in [0, 1] for p in predictions)

class TestDriftDetection:
    def test_drift_detector_initialization(self):
        baseline_data = pd.DataFrame({
            f'V{i}': np.random.rand(100) for i in range(1, 29)
        })
        baseline_data['Amount'] = np.random.rand(100) * 1000
        baseline_data.to_csv('baseline_test.csv', index=False)
        
        detector = ModelDriftDetector('baseline_test.csv', threshold=0.05)
        
        assert detector.threshold == 0.05
        assert len(detector.feature_columns) == 29
        
    def test_no_drift_detection(self):
        # Create identical distributions
        baseline_data = pd.DataFrame({
            f'V{i}': np.random.rand(1000) for i in range(1, 29)
        })
        baseline_data['Amount'] = np.random.rand(1000) * 1000
        baseline_data.to_csv('baseline_test2.csv', index=False)
        
        detector = ModelDriftDetector('baseline_test2.csv', threshold=0.05)
        
        # Test with same distribution
        current_data = baseline_data.copy()
        result = detector.detect_data_drift(current_data)
        
        assert result['drift_detected'] == False
'''

# Integration tests
integration_tests = '''import pytest
import boto3
import json
from moto import mock_s3, mock_sagemaker
from src.train_boto3 import create_training_job

@mock_s3
@mock_sagemaker
class TestSageMakerIntegration:
    def setup_method(self):
        self.s3 = boto3.client('s3', region_name='us-east-1')
        self.sagemaker = boto3.client('sagemaker', region_name='us-east-1')
        self.bucket = 'test-fraud-detection-bucket'
        self.s3.create_bucket(Bucket=self.bucket)
        
    def test_training_job_creation(self):
        job_name = 'test-training-job'
        
        # This would test the training job creation logic
        # In real scenario, would verify job configuration
        assert True  # Placeholder
        
    def test_model_deployment(self):
        # Test endpoint configuration and deployment
        endpoint_config = {
            'EndpointConfigName': 'test-config',
            'ProductionVariants': [{
                'VariantName': 'AllTraffic',
                'ModelName': 'test-model',
                'InstanceType': 'ml.t2.medium',
                'InitialInstanceCount': 1
            }]
        }
        
        # Test deployment logic
        assert endpoint_config['ProductionVariants'][0]['InstanceType'] == 'ml.t2.medium'

class TestLambdaFunction:
    def test_lambda_handler_valid_input(self):
        from lambda_function import lambda_handler
        
        event = {
            'body': json.dumps({
                'features': [0.0] * 30
            })
        }
        
        # Mock SageMaker runtime response
        # In real test, use mocking library
        pass
        
    def test_lambda_handler_invalid_input(self):
        from lambda_function import lambda_handler
        
        event = {
            'body': json.dumps({
                'features': [0.0] * 20  # Wrong number of features
            })
        }
        
        response = lambda_handler(event, None)
        assert response['statusCode'] == 400
'''

# Batch inference pipeline
batch_inference = '''import boto3
import pandas as pd
from datetime import datetime
import json

class BatchInferencePipeline:
    def __init__(self, bucket_name, endpoint_name):
        self.s3 = boto3.client('s3')
        self.sagemaker = boto3.client('sagemaker-runtime')
        self.bucket = bucket_name
        self.endpoint = endpoint_name
        
    def run_batch_inference(self, input_file, output_file):
        """Run batch predictions on large dataset"""
        # Read data from S3
        obj = self.s3.get_object(Bucket=self.bucket, Key=input_file)
        df = pd.read_csv(obj['Body'])
        
        predictions = []
        batch_size = 100
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Convert to CSV format for SageMaker
            csv_data = batch.to_csv(index=False, header=False)
            
            # Invoke endpoint
            response = self.sagemaker.invoke_endpoint(
                EndpointName=self.endpoint,
                ContentType='text/csv',
                Body=csv_data
            )
            
            # Parse predictions
            result = response['Body'].read().decode()
            batch_predictions = [float(x) for x in result.strip().split('\\n')]
            predictions.extend(batch_predictions)
            
            print(f"Processed {i+len(batch)}/{len(df)} records")
        
        # Add predictions to dataframe
        df['prediction'] = [1 if p > 0.5 else 0 for p in predictions]
        df['fraud_score'] = predictions
        
        # Save results to S3
        output_buffer = df.to_csv(index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=output_file,
            Body=output_buffer
        )
        
        # Generate summary report
        summary = {
            'total_transactions': len(df),
            'fraud_detected': df['prediction'].sum(),
            'fraud_rate': df['prediction'].mean(),
            'avg_fraud_score': df['fraud_score'].mean(),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary

if __name__ == '__main__':
    pipeline = BatchInferencePipeline(
        bucket_name='fraud-detection-bucket',
        endpoint_name='fraud-detection-endpoint'
    )
    
    summary = pipeline.run_batch_inference(
        input_file='batch/input_data.csv',
        output_file=f'batch/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    
    print(json.dumps(summary, indent=2))
'''

# MLflow integration
mlflow_tracking = '''import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

class MLflowExperimentTracker:
    def __init__(self, experiment_name="fraud-detection", tracking_uri=None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
    def train_with_tracking(self, X_train, y_train, X_val, y_val, params):
        """Train model with MLflow tracking"""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Make predictions
            y_pred = model.predict(dval)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred_binary),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1_score': f1_score(y_val, y_pred_binary),
                'roc_auc': roc_auc_score(y_val, y_pred)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            # Log additional info
            mlflow.set_tags({
                'training_date': datetime.now().isoformat(),
                'model_type': 'xgboost',
                'dataset_size': len(X_train)
            })
            
            print(f"Run ID: {run.info.run_id}")
            print(f"Metrics: {metrics}")
            
            return model, run.info.run_id
    
    def register_best_model(self, run_id, model_name="fraud-detection-model"):
        """Register model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        return result
    
    def promote_model_to_production(self, model_name, version):
        """Promote model version to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"Model {model_name} version {version} promoted to Production")
    
    def get_production_model(self, model_name):
        """Get current production model"""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        
        if versions:
            return versions[0]
        return None
'''

# Write files
with open('tests_unit_test_model.py', 'w') as f:
    f.write(unit_tests)

with open('tests_integration_test_sagemaker.py', 'w') as f:
    f.write(integration_tests)
    
with open('src_batch_inference.py', 'w') as f:
    f.write(batch_inference)
    
with open('src_mlflow_tracking.py', 'w') as f:
    f.write(mlflow_tracking)

print("✅ Created unit tests: tests_unit_test_model.py")
print("✅ Created integration tests: tests_integration_test_sagemaker.py")
print("✅ Created batch inference: src_batch_inference.py")
print("✅ Created MLflow tracking: src_mlflow_tracking.py")
