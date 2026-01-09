import boto3
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
