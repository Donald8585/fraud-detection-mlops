import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Test AWS connection
print("Testing AWS connection...")
s3 = boto3.client('s3')
print("✓ AWS credentials work!")

# Load and preprocess data
print("\nLoading credit card fraud dataset...")
df = pd.read_csv('creditcard.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Quick preprocessing
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
print(f"Fraud cases in train: {y_train.sum()}")

# Train a simple local model first
print("\nTraining local Random Forest model...")
model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n" + "="*50)
print("LOCAL MODEL PERFORMANCE")
print("="*50)
print(classification_report(y_test, y_pred))

# Save model locally
with open('local_fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✓ Model saved to local_fraud_model.pkl")

# Upload to S3 for later use
print("\nUploading to S3...")
bucket = 'your-fraud-detection-bucket-1767008231'

# Upload data
train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

train_data.to_csv('train.csv', index=False, header=False)
test_data.to_csv('test.csv', index=False, header=False)

s3.upload_file('train.csv', bucket, 'data/train.csv')
s3.upload_file('test.csv', bucket, 'data/test.csv')
s3.upload_file('local_fraud_model.pkl', bucket, 'models/local_model.pkl')

print(f"✓ Files uploaded to s3://{bucket}/")
print("\n✅ LOCAL TESTING COMPLETE!")
print("Now you can deploy to SageMaker when ready.")
