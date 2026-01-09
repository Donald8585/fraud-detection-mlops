import boto3
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
