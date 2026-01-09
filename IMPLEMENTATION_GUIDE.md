# MLOps Enhancement Implementation Guide

## 🚀 Quick Start

This guide walks through implementing all MLOps enhancements for the fraud detection system.

## 📋 Prerequisites

- AWS Account with appropriate permissions
- Python 3.9+
- Docker installed
- Terraform installed (optional)
- GitHub repository access

## 🛠️ Implementation Steps

### 1. Set Up Local Development Environment

```bash
# Clone repository
git clone https://github.com/Donald8585/fraud-detection-mlops.git
cd fraud-detection-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start local services (MLflow, Prometheus, Grafana)
docker-compose up -d
```

### 2. Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Set environment variables
export AWS_REGION=us-east-1
export S3_BUCKET=your-fraud-detection-bucket
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Set Up CI/CD Pipeline

```bash
# Create GitHub secrets
gh secret set AWS_ACCESS_KEY_ID
gh secret set AWS_SECRET_ACCESS_KEY
gh secret set AWS_ACCOUNT_ID
gh secret set SLACK_WEBHOOK  # Optional

# Push code to trigger CI/CD
git add .
git commit -m "Add MLOps enhancements"
git push origin main
```

### 4. Deploy Infrastructure with Terraform

```bash
cd terraform/

# Initialize Terraform
terraform init

# Review planned changes
terraform plan \
  -var="s3_bucket_name=your-bucket" \
  -var="mlflow_tracking_uri=http://mlflow:5000" \
  -var="alert_email=your@email.com"

# Apply infrastructure
terraform apply -auto-approve
```

### 5. Train Model with MLflow Tracking

```python
from src.mlflow_tracking import MLflowExperimentTracker

# Initialize tracker
tracker = MLflowExperimentTracker(
    experiment_name="fraud-detection",
    tracking_uri="http://localhost:5000"
)

# Train with tracking
params = {
    'max_depth': 5,
    'eta': 0.2,
    'objective': 'binary:logistic',
    'scale_pos_weight': 10
}

model, run_id = tracker.train_with_tracking(
    X_train, y_train, X_val, y_val, params
)

# Register model
tracker.register_best_model(run_id, "fraud-detection-model")
```

### 6. Deploy with A/B Testing

```python
from scripts.deploy_model_ab import ModelDeploymentOrchestrator

orchestrator = ModelDeploymentOrchestrator()

orchestrator.deploy_model_with_ab_testing(
    model_a_uri='s3://bucket/models/v1/model.tar.gz',
    model_b_uri='s3://bucket/models/v2/model.tar.gz',
    traffic_split=0.5  # 50/50 split
)
```

### 7. Set Up Model Monitoring

```python
from src.monitoring.drift_detector import ModelDriftDetector

detector = ModelDriftDetector(
    baseline_data_path='s3://bucket/baseline_data.csv',
    threshold=0.05
)

# Run drift detection (schedule this with EventBridge)
current_data = pd.read_csv('s3://bucket/current_week_data.csv')
drift_result = detector.detect_data_drift(current_data)

if drift_result['drift_detected']:
    detector.send_alert(drift_result, sns_topic_arn)
```

### 8. Run Tests

```bash
# Unit tests
pytest tests/unit/ -v --cov=src

# Integration tests
pytest tests/integration/ -v

# Full test suite
pytest tests/ -v --cov=src --cov-report=html
```

### 9. Access Dashboards

- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **CloudWatch**: AWS Console > CloudWatch > Dashboards > FraudDetection-Production

## 📊 Key Features Implemented

✅ **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
✅ **Model Monitoring**: Drift detection with CloudWatch alerts
✅ **A/B Testing**: Traffic splitting between model versions
✅ **Model Versioning**: Complete version management with rollback
✅ **Batch Inference**: Process large datasets efficiently
✅ **MLflow Integration**: Experiment tracking and model registry
✅ **Infrastructure as Code**: Terraform for reproducible deployments
✅ **Comprehensive Testing**: Unit, integration, and smoke tests
✅ **Observability**: CloudWatch dashboards and metrics

## 🔄 Daily Operations

### Retrain Model
```bash
python scripts/trigger_training.py --dataset s3://bucket/new_data.csv
```

### Check Drift
```bash
python src/monitoring/drift_detector.py --current-data s3://bucket/current.csv
```

### Rollback Model
```python
from src.model_versioning import ModelVersionManager

manager = ModelVersionManager(s3_bucket='bucket')
manager.rollback_to_version('v20260109_143000')
```

### Run Batch Inference
```python
from src.batch_inference import BatchInferencePipeline

pipeline = BatchInferencePipeline('bucket', 'endpoint-name')
summary = pipeline.run_batch_inference('input.csv', 'output.csv')
```

## 📈 Cost Optimization

- Shut down SageMaker endpoints when not in use
- Use spot instances for training
- Enable S3 lifecycle policies
- Monitor CloudWatch costs

## 🆘 Troubleshooting

### CI/CD Pipeline Fails
- Check GitHub Actions logs
- Verify AWS credentials
- Ensure IAM permissions are correct

### Endpoint Deployment Fails
- Check SageMaker logs in CloudWatch
- Verify model artifacts in S3
- Check IAM role permissions

### Drift Detection Not Working
- Verify baseline data exists
- Check CloudWatch permissions
- Ensure SNS topic is created

## 📚 Additional Resources

- [AWS SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

## 🎓 Skills Demonstrated

- End-to-end MLOps pipeline automation
- Cloud infrastructure management
- Model monitoring and observability
- A/B testing and experimentation
- DevOps best practices
- Production ML system design

---

Built with ❤️  for production-grade ML systems
