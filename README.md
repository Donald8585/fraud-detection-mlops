# Credit Card Fraud Detection - Production ML API

Real-time fraud detection using XGBoost on AWS SageMaker with serverless inference.

## Live API
**Endpoint:** `https://aa17j8hu64.execute-api.us-east-1.amazonaws.com/prod/predict`

## Architecture
Kaggle Dataset → S3 → SageMaker Training (XGBoost) → Model Artifacts → SageMaker Endpoint → Lambda → API Gateway

## Tech Stack
- Python (pandas, scikit-learn, boto3)
- AWS SageMaker (XGBoost training + real-time inference)
- AWS Lambda (serverless orchestration)
- AWS S3 (data lake)
- API Gateway (REST API)

## Usage
curl -X POST https://aa17j8hu64.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
