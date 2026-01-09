variable "aws_region" {
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
