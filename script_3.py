
# Create remaining MLOps files

# Updated requirements.txt with all dependencies
requirements_prod = '''# Core ML Libraries
sagemaker>=2.100.0
xgboost>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0

# AWS SDK
boto3>=1.26.0
awscli>=1.27.0

# MLOps & Experiment Tracking
mlflow>=2.0.0
evidently>=0.2.0

# Monitoring & Observability
prometheus-client>=0.16.0
sentry-sdk>=1.14.0

# API & Web
fastapi>=0.95.0
uvicorn>=0.20.0
pydantic>=1.10.0

# Testing
pytest>=7.2.0
pytest-cov>=4.0.0
pytest-asyncio>=0.20.0
moto>=4.1.0

# Data Validation
great-expectations>=0.15.0
pandera>=0.13.0

# Utilities
python-dotenv>=0.21.0
pyyaml>=6.0
'''

requirements_dev = '''# Development dependencies
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
isort>=5.12.0
pre-commit>=3.0.0

# Documentation
mkdocs>=1.4.0
mkdocs-material>=9.0.0

# Profiling
py-spy>=0.3.14
memory-profiler>=0.60.0
'''

# Model versioning system
model_versioning = '''import boto3
import json
import hashlib
from datetime import datetime
from typing import Dict, List

class ModelVersionManager:
    def __init__(self, s3_bucket, dynamodb_table='model-versions'):
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(dynamodb_table)
        self.bucket = s3_bucket
        
    def register_model_version(self, model_path, metadata):
        """Register new model version with metadata"""
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Generate version ID
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Upload model to S3
        s3_key = f"models/{version_id}/model.tar.gz"
        self.s3.upload_file(model_path, self.bucket, s3_key)
        
        # Store metadata in DynamoDB
        item = {
            'version_id': version_id,
            'model_hash': model_hash,
            's3_uri': f"s3://{self.bucket}/{s3_key}",
            'created_at': datetime.now().isoformat(),
            'status': 'registered',
            'metadata': metadata
        }
        
        self.table.put_item(Item=item)
        
        print(f"✅ Registered model version: {version_id}")
        return version_id
    
    def get_model_version(self, version_id):
        """Retrieve model version details"""
        response = self.table.get_item(Key={'version_id': version_id})
        return response.get('Item')
    
    def list_model_versions(self, status=None):
        """List all model versions, optionally filtered by status"""
        if status:
            response = self.table.scan(
                FilterExpression='#status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': status}
            )
        else:
            response = self.table.scan()
        
        return response.get('Items', [])
    
    def promote_version(self, version_id, stage='production'):
        """Promote model version to specific stage"""
        # Update version status
        self.table.update_item(
            Key={'version_id': version_id},
            UpdateExpression='SET #status = :status, promoted_at = :timestamp',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': stage,
                ':timestamp': datetime.now().isoformat()
            }
        )
        
        print(f"✅ Promoted version {version_id} to {stage}")
    
    def rollback_to_version(self, version_id):
        """Rollback to previous model version"""
        # Get version details
        version_info = self.get_model_version(version_id)
        
        if not version_info:
            raise ValueError(f"Version {version_id} not found")
        
        # Download model from S3
        s3_uri = version_info['s3_uri']
        local_path = f"/tmp/model_{version_id}.tar.gz"
        
        # Extract bucket and key from S3 URI
        bucket = s3_uri.split('/')[2]
        key = '/'.join(s3_uri.split('/')[3:])
        
        self.s3.download_file(bucket, key, local_path)
        
        # Deploy this version (implement deployment logic here)
        print(f"✅ Rolled back to version {version_id}")
        return local_path
    
    def compare_versions(self, version_a, version_b):
        """Compare two model versions"""
        va = self.get_model_version(version_a)
        vb = self.get_model_version(version_b)
        
        comparison = {
            'version_a': {
                'id': va['version_id'],
                'created_at': va['created_at'],
                'metadata': va.get('metadata', {})
            },
            'version_b': {
                'id': vb['version_id'],
                'created_at': vb['created_at'],
                'metadata': vb.get('metadata', {})
            }
        }
        
        return comparison

# Example usage
if __name__ == '__main__':
    manager = ModelVersionManager(s3_bucket='fraud-detection-bucket')
    
    # Register new model
    metadata = {
        'accuracy': 0.98,
        'precision': 0.95,
        'recall': 0.97,
        'training_data_size': 284807,
        'hyperparameters': {
            'max_depth': 5,
            'eta': 0.2
        }
    }
    
    version_id = manager.register_model_version('model.tar.gz', metadata)
    
    # Promote to production
    manager.promote_version(version_id, 'production')
'''

# Deployment automation script
deployment_script = '''import boto3
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
'''

# Write all files
with open('requirements.txt', 'w') as f:
    f.write(requirements_prod)
    
with open('requirements-dev.txt', 'w') as f:
    f.write(requirements_dev)

with open('src_model_versioning.py', 'w') as f:
    f.write(model_versioning)
    
with open('scripts_deploy_model_ab.py', 'w') as f:
    f.write(deployment_script)

print("✅ Created requirements.txt")
print("✅ Created requirements-dev.txt")
print("✅ Created model versioning: src_model_versioning.py")
print("✅ Created deployment script: scripts_deploy_model_ab.py")
