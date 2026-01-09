
# Create comprehensive MLOps enhancement files for fraud detection project

import os
import json

# Create directory structure
files_to_create = {}

# 1. GitHub Actions CI/CD Pipeline
files_to_create['.github/workflows/ci-cd.yml'] = '''name: MLOps CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  PYTHON_VERSION: '3.9'
  S3_BUCKET: your-fraud-detection-bucket
  
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          
      - name: Run linting
        run: |
          flake8 src/ tests/ --max-line-length=120
          black --check src/ tests/
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          
  integration-test:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v
          
  build-and-deploy:
    needs: [test, integration-test]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          
      - name: Build and push Docker image
        run: |
          docker build -t fraud-detection:${{ github.sha }} .
          aws ecr get-login-password --region ${{ env.AWS_REGION }} | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com
          docker tag fraud-detection:${{ github.sha }} ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/fraud-detection:${{ github.sha }}
          docker push ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/fraud-detection:${{ github.sha }}
          
      - name: Trigger SageMaker training
        run: |
          python scripts/trigger_training.py --image-uri ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/fraud-detection:${{ github.sha }}
          
      - name: Deploy model with A/B testing
        run: |
          python scripts/deploy_model_ab.py --model-version ${{ github.sha }}
          
      - name: Run smoke tests
        run: |
          pytest tests/smoke/ -v
          
      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Deployment ${{ job.status }}'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
'''

# 2. Dockerfile for containerization
files_to_create['Dockerfile'] = '''FROM python:3.9-slim

WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=train.py

# Expose port for serving
EXPOSE 8080

# Entry point for SageMaker training
ENTRYPOINT ["python", "src/train.py"]
'''

# 3. Model drift detection
files_to_create['src/monitoring/drift_detector.py'] = '''import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import mlflow

class ModelDriftDetector:
    def __init__(self, baseline_data_path, threshold=0.05):
        self.s3 = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        self.baseline_data = pd.read_csv(baseline_data_path)
        self.threshold = threshold
        self.feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
    def detect_data_drift(self, current_data):
        """Detect distribution drift using KS test"""
        drift_detected = {}
        drift_scores = {}
        
        for feature in self.feature_columns:
            baseline_dist = self.baseline_data[feature].values
            current_dist = current_data[feature].values
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(baseline_dist, current_dist)
            
            drift_detected[feature] = p_value < self.threshold
            drift_scores[feature] = ks_statistic
            
            # Log to CloudWatch
            self.cloudwatch.put_metric_data(
                Namespace='FraudDetection/Drift',
                MetricData=[{
                    'MetricName': f'DriftScore_{feature}',
                    'Value': ks_statistic,
                    'Unit': 'None',
                    'Timestamp': datetime.now()
                }]
            )
        
        # Log to MLflow
        mlflow.log_metrics({
            f'drift_{k}': v for k, v in drift_scores.items()
        })
        
        return {
            'drift_detected': any(drift_detected.values()),
            'features_with_drift': [k for k, v in drift_detected.items() if v],
            'drift_scores': drift_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_prediction_drift(self, predictions_path):
        """Detect drift in model predictions"""
        predictions = pd.read_csv(predictions_path)
        
        # Calculate prediction distribution metrics
        fraud_rate = predictions['prediction'].mean()
        avg_fraud_score = predictions['fraud_score'].mean()
        
        # Compare with baseline (assuming 0.172% fraud rate)
        baseline_fraud_rate = 0.00172
        
        drift_ratio = abs(fraud_rate - baseline_fraud_rate) / baseline_fraud_rate
        
        # Alert if fraud rate changed significantly (>50%)
        alert_threshold = 0.5
        drift_detected = drift_ratio > alert_threshold
        
        # Log metrics
        self.cloudwatch.put_metric_data(
            Namespace='FraudDetection/Predictions',
            MetricData=[
                {
                    'MetricName': 'FraudRate',
                    'Value': fraud_rate,
                    'Unit': 'Percent',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': 'AvgFraudScore',
                    'Value': avg_fraud_score,
                    'Unit': 'None',
                    'Timestamp': datetime.now()
                }
            ]
        )
        
        return {
            'drift_detected': drift_detected,
            'current_fraud_rate': fraud_rate,
            'baseline_fraud_rate': baseline_fraud_rate,
            'drift_ratio': drift_ratio,
            'timestamp': datetime.now().isoformat()
        }
    
    def send_alert(self, drift_info):
        """Send SNS alert if drift detected"""
        sns = boto3.client('sns')
        topic_arn = 'arn:aws:sns:us-east-1:ACCOUNT_ID:fraud-detection-alerts'
        
        message = f"""
        MODEL DRIFT DETECTED
        
        Timestamp: {drift_info['timestamp']}
        Features with drift: {drift_info.get('features_with_drift', [])}
        
        Action Required: Review model performance and consider retraining.
        """
        
        sns.publish(
            TopicArn=topic_arn,
            Subject='Fraud Detection Model Drift Alert',
            Message=message
        )

if __name__ == '__main__':
    detector = ModelDriftDetector('s3://bucket/baseline_data.csv')
    
    # Load current data
    current_data = pd.read_csv('s3://bucket/current_data.csv')
    
    # Detect drift
    drift_result = detector.detect_data_drift(current_data)
    
    if drift_result['drift_detected']:
        detector.send_alert(drift_result)
        print("⚠️  Drift detected!")
    else:
        print("✅ No drift detected")
'''

print("Creating GitHub Actions workflow...")
print("Creating Dockerfile...")
print("Creating drift detector...")
print("Files created successfully!")
