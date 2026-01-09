terraform {
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
