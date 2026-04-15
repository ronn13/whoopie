# API-Driven Model Integration Guide

This document outlines the architecture, requirements, and deployment steps for deploying our Aviation Safety Classification models as distributed, standalone microservices on AWS.

## 1. Why API-Driven?

Instead of centralizing all `.pt` weights into a single massive UI server (which causes severe Memory/VRAM exhaustions and dependency conflicts across PyTorch versions), we are moving to a **Microservices Architecture**.

In this architecture, **the Demo UI is just a "dumb" frontend**. When a user inputs an incident narrative, the UI fires concurrent HTTP POST requests to the separate cloud instances hosting each team member's model. 
*   **Total Isolation**: You can use whatever framework or dependencies you want (TensorFlow, PyTorch, Scikit-Learn, custom pipelines) without conflicting with other teammates.
*   **Distributed Compute**: No single machine is crushed under the weight of loading four transformer models simultaneously.

---

## 2. The API Contract (What Your Endpoint Must Do)

To ensure the central UI can digest your model's predictions, everyone must adhere to a strict API schema. We recommend using **FastAPI** to build your endpoint.

### Expected Request (Input to your API)
The UI will send a `POST` request to your `.../predict` endpoint with the following JSON body:
```json
{
    "narrative": "The aircraft experienced severe turbulence leading to cabin crew injury...",
    "event_id": "optional-1234"
}
```

### Expected Response (Output from your API)
Your model must process the narrative, run inference, and return a JSON payload exactly matching this structure:
```json
{
    "model_id": "team_alpha_distilbert",
    "display_name": "Team Alpha DistilBERT",
    "prediction": {
        "top_class": "TURB",
        "confidence": 0.942,
        "top_5": [
            {"class": "TURB", "confidence": 0.942},
            {"class": "WSTRW", "confidence": 0.031},
            {"class": "LOC-I", "confidence": 0.012},
            {"class": "MAC", "confidence": 0.005},
            {"class": "ICE", "confidence": 0.001}
        ]
    },
    "inference_time_ms": 145
}
```
> **Note**: The `top_5` array is required for the UI to generate comparison graphs. The classes must be the standard ICAO ADREP taxonomy strings (e.g., `CFIT`, `MAC`, `RE`).

---

## 3. Example FastAPI Implementation

Here is a bare-bones template to convert your local Jupyter/PyTorch architecture into an API:

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import time

# 1. Define Request/Response schema structures
class IncidentRequest(BaseModel):
    narrative: str
    event_id: str = None

app = FastAPI()

# 2. Load your model GLOBALLY so it only loads into memory once on startup
print("Loading model...")
# model = load_my_custom_architecture()
# model.eval()

@app.post("/predict")
async def predict(request: IncidentRequest):
    start_time = time.time()
    
    # 3. Process the incoming string
    # inputs = my_tokenizer(request.narrative, return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**inputs).logits
    #     probs = torch.softmax(logits, dim=-1)
    
    # 4. Format the output according to the API contract
    inference_time = int((time.time() - start_time) * 1000)
    
    return {
        "model_id": "my_team_model",
        "display_name": "My Champion Architecture",
        "prediction": {
            "top_class": "CFIT",  # Replace with actual logic
            "confidence": 0.85,
            "top_5": [
                {"class": "CFIT", "confidence": 0.85},
                # ... 4 more classes
            ]
        },
        "inference_time_ms": inference_time
    }
```
You can test this locally by running: `pip install fastapi uvicorn` followed by `uvicorn main:app --host 0.0.0.0 --port 8000`

---

## 4. Deploying from Your Laptop to AWS

To get your API off your laptop and onto a public AWS endpoint, we highly recommend **Containerizing** your app and deploying it via **AWS App Runner**. App Runner automatically balances load, provisions secure HTTPS URLs, and requires zero DevOps instance management.

### Step 1: Dockerize Your App
Create a `Dockerfile` in the root of your project:
```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your scripts and model weights
COPY . .

# Expose the API port
EXPOSE 8080

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Step 2: Push to AWS ECR (Elastic Container Registry)
1. Install the AWS CLI and log in.
2. In the AWS Console, navigate to **Elastic Container Registry (ECR)** and click *Create repository* (e.g., `team-alpha-model`).
3. Run the "View push commands" provided by AWS in your terminal to build your Docker image and push it to the ECR cloud.

### Step 3: Deploy via AWS App Runner
1. In the AWS Console, navigate to **AWS App Runner** and click *Create an App Runner service*.
2. **Repository type**: Select `Amazon ECR`.
3. Choose the container image you just pushed in Step 2.
4. **Service Settings**:
    * Compute capacity: Assign adequate memory (e.g., 2 GB or 4 GB RAM if your model requires it).
    * Port: Set to `8080` (matching your Dockerfile).
5. Click **Deploy**.

### Step 4: Hooking it to the UI
Once App Runner finishes deploying (takes ~5 minutes), it will give you a public "Default domain" URL (e.g., `https://xyz123.us-east-1.awsapprunner.com`). 

Send this domain to the UI integration team so they can point their requests to:
`POST https://xyz123.us-east-1.awsapprunner.com/predict`
