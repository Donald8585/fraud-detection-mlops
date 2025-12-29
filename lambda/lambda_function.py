import json
import boto3
import os

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT']

def lambda_handler(event, context):
    """
    Lambda function to invoke SageMaker endpoint
    Triggered by API Gateway
    """
    try:
        # Parse input from API Gateway
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
        
        # Extract features (expects 30 features for credit card fraud)
        features = body.get('features', [])
        
        if len(features) != 30:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Expected 30 features'})
            }
        
        # Prepare payload for SageMaker
        payload = json.dumps({'features': features})
        
        # Invoke SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Log to CloudWatch for monitoring
        print(f"Prediction: {result}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
