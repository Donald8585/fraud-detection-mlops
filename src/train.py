import pandas as pd
import numpy as np
import boto3
import sagemaker
from sklearn.model_selection import train_test_split

print("🚀 Fraud Detection MLOps - Production Ready")

# Get credentials using boto3 (works locally)
boto_session = boto3.Session()
region = boto_session.region_name
account_id = boto_session.client('sts').get_caller_identity()['Account']
ROLE_ARN = f"arn:aws:iam::{account_id}:role/LambdaSageMakerExecutionRole"
bucket = 'your-fraud-detection-bucket-1767008231'

print(f"Region: {region}")
print(f"Role: {ROLE_ARN}")
print(f"Bucket: {bucket}")

def preprocess_data():
    print("1. Preprocessing data...")
    df = pd.read_csv('creditcard.csv')
    print(f"Dataset loaded: {df.shape}")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    s3 = boto3.client('s3')
    train_df = pd.concat([y_train, X_train], axis=1)
    train_df.to_csv('train.csv', index=False, header=False)
    
    s3.upload_file('train.csv', bucket, 'data/train.csv')
    print("✓ Data uploaded to s3://{}/data/train.csv".format(bucket))
    return f"s3://{bucket}/data/train.csv"

print("2. Setting up SageMaker...")
session = sagemaker.Session(boto_session=boto_session)
container = sagemaker.image_uris.retrieve("xgboost", region, "1.7-1")

print("3. Creating estimator...")
estimator = sagemaker.estimator.Estimator(
    image_uri=container,
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.t3.medium",
    output_path=f"s3://{bucket}/models/",
    sagemaker_session=session,
    hyperparameters={
        "max_depth": "5",
        "eta": "0.2",
        "objective": "binary:logistic",
        "num_round": "50",
        "scale_pos_weight": "10"
    }
)

print("4. Starting training job...")
train_data = preprocess_data()
estimator.fit({'train': train_data})

print("5. Deploying endpoint...")
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="fraud-detection-endpoint"
)

print("🎉 SUCCESS!")
print(f"✅ SageMaker Endpoint: fraud-detection-endpoint")
print(f"✅ API Gateway: https://tc1u6gp0hh.execute-api.us-east-1.amazonaws.com")
print("\nTest your API:")
print('curl -X POST https://tc1u6gp0hh.execute-api.us-east-1.amazonaws.com \\')
print('  -H "Content-Type: application/json" \\')
print('  -d \'{"features": [-1.36,-0.07,2.54,1.38,-0.34,0.46,0.24,0.1,0.36,0.09,-0.55,-0.62,-0.99,-0.31,1.47,-0.47,0.21,0.03,0.4,0.25,-0.02,0.24,0.85,0.16,0.15,0.14,0.15,-0.17,0.02,0.02]}\'')
