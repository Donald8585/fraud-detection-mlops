import pandas as pd
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
