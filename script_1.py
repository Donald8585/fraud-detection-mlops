
# Continue creating MLOps files

# 4. A/B Testing Framework
ab_testing_code = '''import boto3
import json
from datetime import datetime
import random

class ABTestingFramework:
    def __init__(self, endpoint_name_a, endpoint_name_b, traffic_split=0.5):
        self.runtime = boto3.client('sagemaker-runtime')
        self.cloudwatch = boto3.client('cloudwatch')
        self.endpoint_a = endpoint_name_a
        self.endpoint_b = endpoint_name_b
        self.traffic_split = traffic_split
        
    def route_request(self, features):
        """Route request to model A or B based on traffic split"""
        use_model_a = random.random() < self.traffic_split
        endpoint = self.endpoint_a if use_model_a else self.endpoint_b
        model_version = 'A' if use_model_a else 'B'
        
        # Invoke endpoint
        csv_data = ','.join(map(str, features))
        response = self.runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType='text/csv',
            Body=csv_data
        )
        
        result = float(response['Body'].read().decode().strip())
        prediction = 1 if result > 0.5 else 0
        
        # Log metrics for both models
        self.cloudwatch.put_metric_data(
            Namespace='FraudDetection/ABTesting',
            MetricData=[
                {
                    'MetricName': f'Predictions_Model_{model_version}',
                    'Value': 1,
                    'Unit': 'Count',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': f'FraudScore_Model_{model_version}',
                    'Value': result,
                    'Unit': 'None',
                    'Timestamp': datetime.now()
                }
            ]
        )
        
        return {
            'prediction': prediction,
            'fraud_score': result,
            'model_version': model_version,
            'endpoint': endpoint
        }
    
    def get_performance_metrics(self, days=7):
        """Compare performance of models A and B"""
        metrics = {}
        
        for model in ['A', 'B']:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='FraudDetection/ABTesting',
                MetricName=f'FraudScore_Model_{model}',
                StartTime=datetime.now() - timedelta(days=days),
                EndTime=datetime.now(),
                Period=3600,
                Statistics=['Average', 'SampleCount']
            )
            
            metrics[f'model_{model}'] = {
                'avg_fraud_score': sum(d['Average'] for d in response['Datapoints']) / len(response['Datapoints']),
                'total_predictions': sum(d['SampleCount'] for d in response['Datapoints'])
            }
        
        return metrics
    
    def decide_winner(self, alpha=0.05):
        """Statistical test to determine winning model"""
        import scipy.stats as stats
        
        # Get performance data
        metrics = self.get_performance_metrics()
        
        # Perform t-test (placeholder - need actual performance data)
        # In production, use metrics like precision, recall, F1-score
        
        print(f"Model A: {metrics['model_A']}")
        print(f"Model B: {metrics['model_B']}")
        
        # Decision logic here
        return "model_A"  # Placeholder

# Lambda function for A/B testing
def lambda_handler_ab(event, context):
    ab_tester = ABTestingFramework(
        endpoint_name_a='fraud-detection-endpoint-v1',
        endpoint_name_b='fraud-detection-endpoint-v2',
        traffic_split=0.5
    )
    
    body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
    features = body.get('features', [])
    
    if len(features) != 30:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Expected 30 features'})
        }
    
    result = ab_tester.route_request(features)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(result)
    }
'''

# 5. Terraform Infrastructure as Code
terraform_main = '''terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "terraform-state-fraud-detection"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for data and models
resource "aws_s3_bucket" "fraud_detection" {
  bucket = var.s3_bucket_name
  
  tags = {
    Name        = "Fraud Detection MLOps"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "fraud_detection" {
  bucket = aws_s3_bucket.fraud_detection.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_execution" {
  name = "fraud-detection-sagemaker-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Lambda Function
resource "aws_lambda_function" "fraud_detection_api" {
  filename      = "lambda_deployment.zip"
  function_name = "fraud-detection-api"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  timeout       = 30
  memory_size   = 512
  
  environment {
    variables = {
      SAGEMAKER_ENDPOINT = var.sagemaker_endpoint_name
      MLFLOW_TRACKING_URI = var.mlflow_tracking_uri
    }
  }
  
  tags = {
    Name        = "Fraud Detection API"
    Environment = var.environment
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "fraud_detection" {
  name        = "fraud-detection-api"
  description = "API for fraud detection inference"
}

resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.fraud_detection.id
  parent_id   = aws_api_gateway_rest_api.fraud_detection.root_resource_id
  path_part   = "predict"
}

resource "aws_api_gateway_method" "predict_post" {
  rest_api_id   = aws_api_gateway_rest_api.fraud_detection.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id = aws_api_gateway_rest_api.fraud_detection.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method
  
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.fraud_detection_api.invoke_arn
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/fraud-detection-api"
  retention_in_days = 7
}

# SNS Topic for alerts
resource "aws_sns_topic" "model_alerts" {
  name = "fraud-detection-model-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.model_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "fraud_detection" {
  dashboard_name = "FraudDetection-Production"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["FraudDetection/Predictions", "FraudRate"],
            [".", "AvgFraudScore"]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Prediction Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/SageMaker", "ModelLatency", {"stat": "Average"}],
            [".", ".", {"stat": "p99"}]
          ]
          period = 60
          stat   = "Average"
          region = var.aws_region
          title  = "Model Latency"
        }
      }
    ]
  })
}

# Outputs
output "api_gateway_url" {
  value = aws_api_gateway_deployment.production.invoke_url
}

output "s3_bucket_name" {
  value = aws_s3_bucket.fraud_detection.bucket
}
'''

terraform_variables = '''variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "s3_bucket_name" {
  description = "S3 bucket for MLOps artifacts"
  type        = string
}

variable "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  type        = string
  default     = "fraud-detection-endpoint"
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  type        = string
}

variable "alert_email" {
  description = "Email for alerts"
  type        = string
}
'''

# Write files
with open('src_monitoring_ab_testing.py', 'w') as f:
    f.write(ab_testing_code)

with open('terraform_main.tf', 'w') as f:
    f.write(terraform_main)
    
with open('terraform_variables.tf', 'w') as f:
    f.write(terraform_variables)

print("✅ Created A/B testing framework: src_monitoring_ab_testing.py")
print("✅ Created Terraform main: terraform_main.tf")
print("✅ Created Terraform variables: terraform_variables.tf")
