import json
import boto3

runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    try:
        # Parse input from API Gateway
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
        features = body.get('features', [])
        
        if len(features) != 30:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Expected 30 features'})
            }
        
        # XGBoost expects CSV format (comma-separated values)
        csv_data = ','.join(map(str, features))
        
        # Invoke SageMaker endpoint with CSV
        response = runtime.invoke_endpoint(
            EndpointName='fraud-detection-endpoint',
            ContentType='text/csv',
            Body=csv_data
        )
        
        # Parse prediction (XGBoost returns raw score)
        result = float(response['Body'].read().decode().strip())
        prediction = 1 if result > 0.5 else 0
        
        print(f"Score: {result}, Prediction: {prediction}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': prediction,
                'fraud_score': result,
                'message': 'Fraud detected' if prediction == 1 else 'Legitimate transaction'
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
