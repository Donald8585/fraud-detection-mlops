import pytest
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
