import time
import json
import logging
from pathlib import Path
from os import getenv
from datetime import datetime, timezone
import sys
import inspect

from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from numpy import array
from pandas import read_csv

import transformers
from utils import (
    load_model_and_features, 
    build_dynamic_feature_model, 
    load_shadow_models, 
    require_basic_auth, 
    run_shadow_models, 
    write_prediction_to_file
)

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------

app = FastAPI(title="Housing Price Prediction API")


# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
APP_DIR = Path("/app")
LOG_DIR = APP_DIR / "logs"
LOG_PATH = LOG_DIR / "prediction_requests.log"
PREDICTION_PATH = LOG_DIR / "predictions.jsonl"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Load Primary Model + Feature Metadata
# -------------------------------------------------------------------
MODEL_ACTIVE_VERSION = getenv("MODEL_ACTIVE_VERSION", "v1")
MODEL_BASE= APP_DIR / "model"
MODEL_DIR = MODEL_BASE / MODEL_ACTIVE_VERSION
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"
DEMOGRAPHICS_PATH = MODEL_BASE / "zipcode_demographics.csv"

model, model_features = load_model_and_features(MODEL_PATH, FEATURES_PATH)

HouseFeatures: type[BaseModel] = build_dynamic_feature_model(model_features)

# -------------------------------------------------------------------
# Load ZIP code demographics once at startup
# -------------------------------------------------------------------

# Identify which demographics columns are required by the model (excluding 'zipcode')
demographics_columns_to_use = ["zipcode"] + [col for col in model_features if col != "zipcode"]
actual_columns = read_csv(DEMOGRAPHICS_PATH, nrows=0).columns.tolist()
valid_cols = [c for c in demographics_columns_to_use if c in actual_columns]

# Read only needed columns
demographics_df = read_csv(
    DEMOGRAPHICS_PATH,
    usecols=valid_cols,
    dtype={"zipcode": str}
)
# Convert to dict of dicts for fast lookup: {zipcode: {col: value, ...}, ...}
demographics_dict = demographics_df.set_index("zipcode").to_dict(orient="index")


shadow_models = load_shadow_models(MODEL_BASE)


# -------------------------------------------------------------------
# Prediction Endpoint with ZIP code demographics merge
# -------------------------------------------------------------------
@app.post("/predict")
async def predict(
    features: HouseFeatures,
    request: Request,
    background_tasks: BackgroundTasks,
    username: str = Depends(require_basic_auth),
):
    start_time = time.time()

    try:
        # Convert input to dict
        features_dict = features.dict()

        # Ensure ZIP code is a string for lookup
        zipcode_str = str(features_dict.get("zipcode"))

        # Merge demographics from preloaded dict
        zip_demo = demographics_dict.get(zipcode_str, {})  # empty if ZIP not found
        merged_features = {**features_dict, **zip_demo}

        # Verify that all model-required features exist
        missing_features = [f for f in model_features if f not in merged_features]
        if missing_features:
            raise ValueError(f"Missing required model features: {missing_features}")

        # Build input vector in the exact order of model_features.json
        input_vector = array([[merged_features[f] for f in model_features]])

        # Predict
        # Instead of: prediction = model.predict(input_vector)[0]
        # prevent blocking with awaitable
        prediction = await run_in_threadpool(model.predict, input_vector)
        prediction = prediction[0]

        latency_ms = round((time.time() - start_time) * 1000, 2)

        metadata = {
            "user": username,
            "client_ip": request.client.host,
            "endpoint": "/predict",
            "model_version": MODEL_ACTIVE_VERSION,
            "latency_ms": latency_ms,
            "status": "success",
            "prediction_timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(json.dumps(metadata))

        record = {
            "metadata": metadata,
            "input": features_dict,
            "prediction": float(prediction),
        }

        # Save prediction asynchronously
        background_tasks.add_task(write_prediction_to_file, PREDICTION_PATH, record)

        # Run shadow models in background using merged_features
        if shadow_models:
            background_tasks.add_task(
                run_shadow_models,
                shadow_models,
                merged_features,
                metadata
            )

        return {
                "prediction": float(prediction),
                "metadata": metadata 
            }

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)

        error_meta = {
            "user": username,
            "client_ip": request.client.host,
            "endpoint": "/predict",
            "model_version": MODEL_ACTIVE_VERSION,
            "latency_ms": latency_ms,
            "status": "error",
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }

        logger.error(json.dumps(error_meta))

        # Return the error_meta directly as top-level keys
        # We use JSONResponse to force the 500 status code
        return JSONResponse(
            status_code=500,
            content={
                "prediction": None,
                **error_meta  # Unpacks status, error message, latency, etc.
            }
        )