import boto3
import pandas as pd
from datetime import datetime
import json

class BatchInferencePipeline:
    def __init__(self, bucket_name, endpoint_name):
        self.s3 = boto3.client('s3')
        self.sagemaker = boto3.client('sagemaker-runtime')
        self.bucket = bucket_name
        self.endpoint = endpoint_name

    def run_batch_inference(self, input_file, output_file):
        """Run batch predictions on large dataset"""
        # Read data from S3
        obj = self.s3.get_object(Bucket=self.bucket, Key=input_file)
        df = pd.read_csv(obj['Body'])

        predictions = []
        batch_size = 100

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            # Convert to CSV format for SageMaker
            csv_data = batch.to_csv(index=False, header=False)

            # Invoke endpoint
            response = self.sagemaker.invoke_endpoint(
                EndpointName=self.endpoint,
                ContentType='text/csv',
                Body=csv_data
            )

            # Parse predictions
            result = response['Body'].read().decode()
            batch_predictions = [float(x) for x in result.strip().split('\n')]
            predictions.extend(batch_predictions)

            print(f"Processed {i+len(batch)}/{len(df)} records")

        # Add predictions to dataframe
        df['prediction'] = [1 if p > 0.5 else 0 for p in predictions]
        df['fraud_score'] = predictions

        # Save results to S3
        output_buffer = df.to_csv(index=False)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=output_file,
            Body=output_buffer
        )

        # Generate summary report
        summary = {
            'total_transactions': len(df),
            'fraud_detected': df['prediction'].sum(),
            'fraud_rate': df['prediction'].mean(),
            'avg_fraud_score': df['fraud_score'].mean(),
            'timestamp': datetime.now().isoformat()
        }

        return summary

if __name__ == '__main__':
    pipeline = BatchInferencePipeline(
        bucket_name='fraud-detection-bucket',
        endpoint_name='fraud-detection-endpoint'
    )

    summary = pipeline.run_batch_inference(
        input_file='batch/input_data.csv',
        output_file=f'batch/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

    print(json.dumps(summary, indent=2))
