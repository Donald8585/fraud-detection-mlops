import pandas as pd
import numpy as np
import boto3
import time
from sklearn.model_selection import train_test_split

print("🚀 Fraud Detection - Using Raw Boto3 (No SageMaker SDK)")

# Setup
boto_session = boto3.Session()
sagemaker_client = boto_session.client('sagemaker')
s3_client = boto_session.client('s3')
region = boto_session.region_name
account_id = boto_session.client('sts').get_caller_identity()['Account']
role = f"arn:aws:iam::{account_id}:role/LambdaSageMakerExecutionRole"
bucket = 'your-fraud-detection-bucket-1767008231'

print(f"Region: {region}, Account: {account_id}")

# Preprocess
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_df = pd.concat([y_train, X_train], axis=1)
train_df.to_csv('train.csv', index=False, header=False)
s3_client.upload_file('train.csv', bucket, 'data/train.csv')
print("✓ Data uploaded")

# Get XGBoost image
ecr_image = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1"

# Create training job
job_name = f"fraud-detection-{int(time.time())}"
print(f"Creating training job: {job_name}")

sagemaker_client.create_training_job(
    TrainingJobName=job_name,
    RoleArn=role,
    AlgorithmSpecification={
        'TrainingImage': ecr_image,
        'TrainingInputMode': 'File'
    },
    InputDataConfig=[{
        'ChannelName': 'train',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': f"s3://{bucket}/data/train.csv",
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        'ContentType': 'text/csv'
    }],
    OutputDataConfig={'S3OutputPath': f"s3://{bucket}/models/"},
    ResourceConfig={
        'InstanceType': 'ml.m5.large',
        'InstanceCount': 1,
        'VolumeSizeInGB': 10
    },
    StoppingCondition={'MaxRuntimeInSeconds': 3600},
    HyperParameters={
        'max_depth': '5',
        'eta': '0.2',
        'objective': 'binary:logistic',
        'num_round': '50',
        'scale_pos_weight': '10'
    }
)

# Wait for completion
print("Training... (5-8 mins)")
while True:
    status = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    if status['TrainingJobStatus'] in ['Completed', 'Failed', 'Stopped']:
        print(f"Status: {status['TrainingJobStatus']}")
        break
    time.sleep(30)

if status['TrainingJobStatus'] != 'Completed':
    print("❌ Training failed")
    exit(1)

# Deploy endpoint
model_name = f"fraud-model-{int(time.time())}"
sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': ecr_image,
        'ModelDataUrl': status['ModelArtifacts']['S3ModelArtifacts']
    },
    ExecutionRoleArn=role
)

config_name = f"fraud-config-{int(time.time())}"
sagemaker_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InstanceType': 'ml.t2.medium',
        'InitialInstanceCount': 1
    }]
)

print("Deploying endpoint... (3-5 mins)")
sagemaker_client.create_endpoint(
    EndpointName='fraud-detection-endpoint',
    EndpointConfigName=config_name
)

while True:
    status = sagemaker_client.describe_endpoint(EndpointName='fraud-detection-endpoint')
    if status['EndpointStatus'] in ['InService', 'Failed']:
        print(f"Endpoint: {status['EndpointStatus']}")
        break
    time.sleep(30)

print("🎉 SUCCESS!")
print("API: https://tc1u6gp0hh.execute-api.us-east-1.amazonaws.com")
