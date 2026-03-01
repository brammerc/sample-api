# Sound Realty Valuation API  
### From Manual to Model-Driven Property Valuation

This project deploys a machine learning model as a **production-ready REST API** that provides real-time property value predictions for the Seattle housing market.

The service:

- Accepts JSON input representing home features
- Automatically enriches requests with zipcode demographic data
- Returns a predicted home value
- Logs predictions for observability
- Supports model versioning and shadow testing
- Runs inside Docker for portability and scalability

---

# Running the API Locally

The application is fully containerized using Docker. This ensures reproducibility and makes scaling straightforward.

---

## Build the Docker Image

```bash
docker build -t sample-api .
```

## Run the Container (Primary Model Only)

```bash
docker run -d \
  -v "/${PWD}/model:/app/model" \
  -v "/${PWD}/logs:/app/logs" \
  -p 8080:8080 \
  -e HOST=0.0.0.0 \
  -e PORT=8080 \
  -e WORKERS=3 \
  -e LOG_LEVEL=debug \
  -e MODEL_ACTIVE_VERSION=v1 \
  --name sample-api-container \
  sample-api
```

What this does:
  - Mounts model directory (/model) into the container, enabling model versioning without rebuilding the image.
  - Mounts logs directory (/logs), allows prediction logs and shadow results to persist outside the container.
  - Exposes port 8080, making the API available at: `http://localhost:8080`
  - Starts 3 Gunicorn workers for parallel request handling.
  - Sets the production model version.

## View the container logs

Useful for performance debugging and monitoring.

```bash
docker logs -f sample-api-container
```

This streams:
  - Request logs
  - Prediction outputs
  - Shadow model results (if enabled)
  - Errors
  - Latency information

## Testing the API

A simple test client is included to demonstrate endpoint functionality.

```bash
python client.py ./mle-project-challenge-2/data/future_unseen_examples.csv
```

This script:
  1. Loads sample homes from future_unseen_examples.csv
  2. Sends POST requests to the /predict endpoint
  3. Prints:
     - Status code
     - JSON prediction response
     - Any errors


## Running with Shadow Models (Advanced)

To evaluate new model versions safely in production, you can enable shadow testing.

```bash
docker run -d \
  -v "/${PWD}/model:/app/model" \
  -v "/${PWD}/logs:/app/logs" \
  -p 8080:8080 \
  -e HOST=0.0.0.0 \
  -e PORT=8080 \
  -e WORKERS=3 \
  -e LOG_LEVEL=debug \
  -e MODEL_ACTIVE_VERSION=v1 \
  -e SHADOW_MODELS=v2 \
  --name sample-api-container \
  sample-api
```

What **_Shadow Mode_** Does
  - v1 remains the live production model.
  - v2 runs silently in the background.
  - The client only receives predictions from v1.
  - Predictions from v2 are logged for comparison.

Benefits of shadow testing:
  - Safe performance evaluation
  - Drift detection
  - Seamless model upgrades
  - Zero-downtime experimentation
