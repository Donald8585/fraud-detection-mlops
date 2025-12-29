FROM python:3.9-slim

WORKDIR /opt/ml

# Install dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY src/inference.py /opt/ml/code/
ENV SAGEMAKER_PROGRAM inference.py

# Set permissions
RUN chmod +x /opt/ml/code/inference.py

CMD ["serve"]
