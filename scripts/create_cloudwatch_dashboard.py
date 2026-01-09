import boto3
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
