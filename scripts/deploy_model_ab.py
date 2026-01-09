import boto3
import time
import json
from datetime import datetime

class ModelDeploymentOrchestrator:
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker')
        self.cloudwatch = boto3.client('cloudwatch')

    def deploy_model_with_ab_testing(self, model_a_uri, model_b_uri, traffic_split=0.5):
        """Deploy two model versions with A/B testing"""
        timestamp = int(time.time())

        # Create model A
        model_a_name = f"fraud-model-a-{timestamp}"
        self.sagemaker.create_model(
            ModelName=model_a_name,
            PrimaryContainer={
                'Image': self._get_xgboost_image(),
                'ModelDataUrl': model_a_uri
            },
            ExecutionRoleArn=self._get_execution_role()
        )

        # Create model B
        model_b_name = f"fraud-model-b-{timestamp}"
        self.sagemaker.create_model(
            ModelName=model_b_name,
            PrimaryContainer={
                'Image': self._get_xgboost_image(),
                'ModelDataUrl': model_b_uri
            },
            ExecutionRoleArn=self._get_execution_role()
        )

        # Create endpoint config with both models
        config_name = f"fraud-ab-config-{timestamp}"
        self.sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'ModelA',
                    'ModelName': model_a_name,
                    'InstanceType': 'ml.t2.medium',
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': traffic_split
                },
                {
                    'VariantName': 'ModelB',
                    'ModelName': model_b_name,
                    'InstanceType': 'ml.t2.medium',
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1 - traffic_split
                }
            ]
        )

        # Create or update endpoint
        endpoint_name = 'fraud-detection-endpoint'

        try:
            self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            # Update existing endpoint
            self.sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            print(f"Updating endpoint {endpoint_name}...")
        except:
            # Create new endpoint
            self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            print(f"Creating endpoint {endpoint_name}...")

        # Wait for deployment
        self._wait_for_endpoint(endpoint_name)

        print(f"✅ Deployed A/B testing endpoint: {endpoint_name}")
        return endpoint_name

    def _wait_for_endpoint(self, endpoint_name, timeout=900):
        """Wait for endpoint to be InService"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']

            print(f"Status: {status}")

            if status == 'InService':
                return True
            elif status in ['Failed', 'RolledBack']:
                raise Exception(f"Endpoint deployment failed: {status}")

            time.sleep(30)

        raise Exception("Endpoint deployment timeout")

    def _get_xgboost_image(self):
        region = boto3.Session().region_name
        return f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1"

    def _get_execution_role(self):
        account_id = boto3.client('sts').get_caller_identity()['Account']
        return f"arn:aws:iam::{account_id}:role/LambdaSageMakerExecutionRole"

if __name__ == '__main__':
    orchestrator = ModelDeploymentOrchestrator()

    # Deploy with A/B testing
    orchestrator.deploy_model_with_ab_testing(
        model_a_uri='s3://bucket/models/model_a/model.tar.gz',
        model_b_uri='s3://bucket/models/model_b/model.tar.gz',
        traffic_split=0.5
    )
