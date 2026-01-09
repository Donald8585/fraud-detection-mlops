import pandas as pd
import boto3

# Load your test data and take a sample as "baseline"
df = pd.read_csv('test.csv')
baseline = df.sample(n=10000, random_state=42)  # 10k rows as baseline

# Save locally first
baseline.to_csv('baseline_data.csv', index=False)
print(f"✅ Created baseline_data.csv with {len(baseline)} rows")

# Create S3 bucket for your project
s3_client = boto3.client('s3')
bucket_name = 'fraud-detection-mlops-alfred'  # Must be globally unique

try:
    s3_client.create_bucket(Bucket=bucket_name)
    print(f"✅ Created S3 bucket: {bucket_name}")
except Exception as e:
    if 'BucketAlreadyOwnedByYou' in str(e):
        print(f"✅ Bucket {bucket_name} already exists")
    else:
        print(f"❌ Error creating bucket: {e}")

# Upload baseline to S3
s3_client.upload_file('baseline_data.csv', bucket_name, 'baseline_data.csv')
print(f"✅ Uploaded baseline_data.csv to s3://{bucket_name}/baseline_data.csv")

# Create current data (slightly different for testing drift)
current = df.sample(n=5000, random_state=99)
current.to_csv('current_data.csv', index=False)
s3_client.upload_file('current_data.csv', bucket_name, 'current_data.csv')
print(f"✅ Uploaded current_data.csv to s3://{bucket_name}/current_data.csv")

print("\n🎯 Ready to test drift detection!")

