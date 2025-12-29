#!/bin/bash

# 1. Set your AWS region and bucket name
export AWS_REGION=us-east-1
export BUCKET_NAME=your-fraud-detection-bucket-$(date +%s)

# 2. Create S3 bucket
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION

# 3. Create ECR repository
aws ecr create-repository --repository-name fraud-detection-ml --region $AWS_REGION

# 4. Create Lambda execution role
aws iam create-role \
  --role-name LambdaSageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# 5. Attach policies to Lambda role
aws iam attach-role-policy \
  --role-name LambdaSageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name LambdaSageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 6. Create Lambda function
cd lambda
zip -r lambda.zip lambda_function.py
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name LambdaSageMakerExecutionRole --query 'Role.Arn' --output text)

aws lambda create-function \
  --function-name FraudDetectionAPI \
  --runtime python3.9 \
  --role $LAMBDA_ROLE_ARN \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda.zip \
  --timeout 30 \
  --memory-size 512 \
  --environment Variables={SAGEMAKER_ENDPOINT=fraud-detection-endpoint}

cd ..

# 7. Create API Gateway
API_ID=$(aws apigatewayv2 create-api \
  --name FraudDetectionAPI \
  --protocol-type HTTP \
  --target arn:aws:lambda:$AWS_REGION:$(aws sts get-caller-identity --query Account --output text):function:FraudDetectionAPI \
  --query 'ApiId' --output text)

# 8. Give API Gateway permission to invoke Lambda
aws lambda add-permission \
  --function-name FraudDetectionAPI \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:$AWS_REGION:$(aws sts get-caller-identity --query Account --output text):$API_ID/*/*"

echo "========================================="
echo "Setup Complete!"
echo "S3 Bucket: $BUCKET_NAME"
echo "API Endpoint: https://$API_ID.execute-api.$AWS_REGION.amazonaws.com"
echo "========================================="
echo "Next steps:"
echo "1. Update src/train.py with bucket name: $BUCKET_NAME"
echo "2. Add GitHub secrets: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
echo "3. Run: python src/train.py"
