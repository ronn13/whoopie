FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install CPU-only torch first (much smaller than CUDA wheel — ~180 MB vs 530 MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY app/requirements-app.txt .
RUN pip install --no-cache-dir flask flask-cors transformers

# Copy application code
COPY app/server.py app/inference.py ./
COPY app/static/ ./static/

# Copy model configs and weights (directory may contain only .json if .pt not yet trained)
COPY outputs/models/ ./outputs/models/

# Expose the API port
EXPOSE 8080

# Run the Flask server
CMD ["python", "server.py"]
