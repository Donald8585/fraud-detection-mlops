
# Create final MLOps files

# Docker Compose for local development
docker_compose = '''version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow_server
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://fraud-detection-bucket/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - mlflow-data:/mlflow

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  localstack:
    image: localstack/localstack:latest
    container_name: localstack
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3,sagemaker,lambda,dynamodb
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - localstack-data:/tmp/localstack

volumes:
  mlflow-data:
  prometheus-data:
  grafana-data:
  localstack-data:
'''

# CloudWatch dashboard configuration
cloudwatch_dashboard = '''import boto3
import json

def create_fraud_detection_dashboard():
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["FraudDetection/Predictions", "FraudRate", {"stat": "Average"}],
                        [".", ".", {"stat": "Maximum"}],
                        [".", ".", {"stat": "Minimum"}]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Fraud Detection Rate",
                    "yAxis": {
                        "left": {
                            "min": 0,
                            "max": 1
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/SageMaker", "ModelLatency", {"stat": "Average"}],
                        [".", ".", {"stat": "p50"}],
                        [".", ".", {"stat": "p90"}],
                        [".", ".", {"stat": "p99"}]
                    ],
                    "period": 60,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Model Inference Latency (ms)",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["FraudDetection/ABTesting", "Predictions_Model_A", {"stat": "Sum"}],
                        [".", "Predictions_Model_B", {"stat": "Sum"}]
                    ],
                    "period": 3600,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "A/B Testing Traffic Distribution"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["FraudDetection/Drift", "DriftScore_V1"],
                        [".", "DriftScore_V2"],
                        [".", "DriftScore_Amount"]
                    ],
                    "period": 3600,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Feature Drift Detection",
                    "annotations": {
                        "horizontal": [{
                            "label": "Drift Threshold",
                            "value": 0.05
                        }]
                    }
                }
            },
            {
                "type": "log",
                "x": 0,
                "y": 12,
                "width": 24,
                "height": 6,
                "properties": {
                    "query": """SOURCE '/aws/lambda/fraud-detection-api'
| fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 20""",
                    "region": "us-east-1",
                    "stacked": false,
                    "title": "Recent Errors",
                    "view": "table"
                }
            }
        ]
    }
    
    cloudwatch.put_dashboard(
        DashboardName='FraudDetection-Production',
        DashboardBody=json.dumps(dashboard_body)
    )
    
    print("✅ CloudWatch dashboard created: FraudDetection-Production")

if __name__ == '__main__':
    create_fraud_detection_dashboard()
'''

# Drift detector standalone file
drift_detector_file = '''import pandas as pd
import numpy as np
from scipy import stats
import boto3
from datetime import datetime, timedelta
import mlflow

class ModelDriftDetector:
    def __init__(self, baseline_data_path, threshold=0.05):
        self.s3 = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        self.sns = boto3.client('sns')
        
        # Load baseline data
        if baseline_data_path.startswith('s3://'):
            bucket = baseline_data_path.split('/')[2]
            key = '/'.join(baseline_data_path.split('/')[3:])
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            self.baseline_data = pd.read_csv(obj['Body'])
        else:
            self.baseline_data = pd.read_csv(baseline_data_path)
            
        self.threshold = threshold
        self.feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
    def detect_data_drift(self, current_data):
        """Detect distribution drift using KS test"""
        drift_detected = {}
        drift_scores = {}
        
        for feature in self.feature_columns:
            if feature not in current_data.columns:
                continue
                
            baseline_dist = self.baseline_data[feature].values
            current_dist = current_data[feature].values
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(baseline_dist, current_dist)
            
            drift_detected[feature] = p_value < self.threshold
            drift_scores[feature] = float(ks_statistic)
            
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
        
        # Log to MLflow if tracking URI is set
        try:
            mlflow.log_metrics({
                f'drift_{k}': v for k, v in drift_scores.items()
            })
        except:
            pass
        
        return {
            'drift_detected': any(drift_detected.values()),
            'features_with_drift': [k for k, v in drift_detected.items() if v],
            'drift_scores': drift_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def send_alert(self, drift_info, sns_topic_arn):
        """Send SNS alert if drift detected"""
        if not drift_info['drift_detected']:
            return
            
        message = f"""
MODEL DRIFT DETECTED

Timestamp: {drift_info['timestamp']}
Features with drift: {', '.join(drift_info['features_with_drift'])}

Drift Scores:
{json.dumps(drift_info['drift_scores'], indent=2)}

Action Required: Review model performance and consider retraining.

Dashboard: https://console.aws.amazon.com/cloudwatch/deeplink.js?region=us-east-1#dashboards:name=FraudDetection-Production
        """
        
        self.sns.publish(
            TopicArn=sns_topic_arn,
            Subject='⚠️  Fraud Detection Model Drift Alert',
            Message=message
        )
        
        print("✅ Alert sent via SNS")
'''

# Comprehensive implementation guide
implementation_guide = '''# MLOps Enhancement Implementation Guide

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

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
terraform plan \\
  -var="s3_bucket_name=your-bucket" \\
  -var="mlflow_tracking_uri=http://mlflow:5000" \\
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
'''

# Write final files
with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose)
    
with open('scripts_create_cloudwatch_dashboard.py', 'w') as f:
    f.write(cloudwatch_dashboard)
    
with open('src_monitoring_drift_detector.py', 'w') as f:
    f.write(drift_detector_file)
    
with open('IMPLEMENTATION_GUIDE.md', 'w') as f:
    f.write(implementation_guide)

print("✅ Created docker-compose.yml")
print("✅ Created CloudWatch dashboard script: scripts_create_cloudwatch_dashboard.py")
print("✅ Created drift detector: src_monitoring_drift_detector.py")
print("✅ Created IMPLEMENTATION_GUIDE.md")
print("\n🎉 ALL MLOPS FILES CREATED SUCCESSFULLY!")
