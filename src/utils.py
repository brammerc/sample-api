from os import getenv
import secrets
import pickle
from pathlib import Path
import json
from typing import Optional, Type, Dict, Any, Tuple
import logging
import importlib
import sys
import traceback

from pandas import DataFrame
from numpy import array
from pydantic import BaseModel, Field, create_model
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


BASIC_USER = "admin"
BASIC_PASS = "changeme"

APP_DIR = Path("/app")
MODEL_DIR = APP_DIR / "model"
LOG_DIR = APP_DIR / "logs"


logger = logging.getLogger(__name__)
gunicorn_logger = logging.getLogger("gunicorn.error")
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)

security = HTTPBasic()

# -------------------------------------------------------------------
# Define Input Schema
# -------------------------------------------------------------------
INPUT_FEATURES: Dict[str, Tuple[Any, Field]] = {
    "price": (float, Field(..., gt=0)),
    "bedrooms": (int, Field(..., ge=0)),
    "bathrooms": (float, Field(..., ge=0)),
    "sqft_living": (int, Field(..., ge=0)),
    "sqft_lot": (int, Field(..., ge=0)),
    "floors": (float, Field(..., ge=0)),
    "waterfront": (int, Field(..., ge=0)),
    "view": (int, Field(..., ge=0)),
    "condition": (int, Field(..., ge=0)),
    "grade": (int, Field(..., ge=0)),
    "sqft_above": (int, Field(..., ge=0)),
    "sqft_basement": (int, Field(..., ge=0)),
    "yr_built": (int, Field(..., ge=0)),
    "yr_renovated": (int, Field(..., ge=0)),
    "zipcode": (int, Field(..., ge=0)),
    "lat": (float, Field(...)),
    "long": (float, Field(...)),
    "sqft_living15": (int, Field(..., ge=0)),
    "sqft_lot15": (int, Field(..., ge=0)),
}


def require_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    username_ok = secrets.compare_digest(credentials.username, BASIC_USER)
    password_ok = secrets.compare_digest(credentials.password, BASIC_PASS)

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def load_optional_module(module_path: Path, module_name: str):
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_and_features(model_path: Path, features_path: Path):
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    with open(features_path, "r") as f:
        feature_list = json.load(f)

    return loaded_model, feature_list


def build_dynamic_feature_model(model_features: list[str]) -> Type[BaseModel]:
    dynamic_fields = {}

    for field_name, (field_type, field_def) in INPUT_FEATURES.items():

        if field_name in model_features:
            # Required
            dynamic_fields[field_name] = (field_type, field_def)

        else:
            # Optional
            dynamic_fields[field_name] = (
                Optional[field_type],
                Field(None)
            )

    return create_model("HouseFeatures", **dynamic_fields)

# Dynamically created HouseFeatures input model looks something like:
#
# class HouseFeatures(BaseModel):
#     price: float = Field(..., gt=0)

#     bedrooms: int = Field(..., ge=0)
#     bathrooms: float = Field(..., ge=0)
#     sqft_living: int = Field(..., ge=0)
#     sqft_lot: int = Field(..., ge=0)
#     floors: float = Field(..., ge=0)

#     waterfront: int = Field(..., ge=0)
#     view: int = Field(..., ge=0)
#     condition: int = Field(..., ge=0)
#     grade: int = Field(..., ge=0)

#     sqft_above: int = Field(..., ge=0)
#     sqft_basement: int = Field(..., ge=0)

#     yr_built: int = Field(..., ge=0)
#     yr_renovated: int = Field(..., ge=0)

#     zipcode: int = Field(..., ge=0)
#     lat: float
#     long: float

#     sqft_living15: Optional[int] = Field(None, ge=0)
#     sqft_lot15: Optional[int] = Field(None, ge=0)


# -------------------------------------------------------------------
# Background Writer
# -------------------------------------------------------------------
def write_prediction_to_file(file_location, record: dict):
    with open(file_location, "a") as f:
        f.write(json.dumps(record) + "\n")


# -------------------------------------------------------------------
# Shadow Model Configuration
# -------------------------------------------------------------------
def get_shadow_model_versions() -> list[str]:
    value = getenv("SHADOW_MODELS", "")
    if not value.strip():
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def load_shadow_models(model_base: Path):
    models = {}
    versions = get_shadow_model_versions()

    for version in versions:
        try:
            model_dir = model_base / version

            model_path = model_dir / f"model.pkl"
            features_path = model_dir / f"model_features.json"

            shadow_model, shadow_features = load_model_and_features(
                model_path, features_path
            )

            models[version] = {
                "model": shadow_model,
                "features": shadow_features,
            }

            logger.info(f"Loaded shadow model: {version}")

        except Exception as e:
            logger.error(f"Failed to load shadow model {version}: {e}")

    return models


def run_shadow_models(shadow_models, features_dict, metadata):
    for version, shadow_obj in shadow_models.items():
        try:
            shadow_model = shadow_obj["model"]
            shadow_features = shadow_obj["features"]

            missing_features = [
                f for f in shadow_features if f not in features_dict
            ]
            if missing_features:
                raise ValueError(
                    f"Missing required shadow model features: {missing_features}"
                )

            input_df = DataFrame(
                [[features_dict[f] for f in shadow_features]], 
                columns=shadow_features
            )
            # input_vector = array(
            #     [[features_dict[f] for f in shadow_features]]
            # )

            prediction = shadow_model.predict(input_df)

            shadow_log = {
                "shadow_model_version": version,
                "metadata": metadata,
                "input": features_dict,
                "prediction": float(prediction),
            }
            
            shadow_predict_path = LOG_DIR / f"predictions_{version}.jsonl"
            write_prediction_to_file(shadow_predict_path, shadow_log)

            logger.info(json.dumps({"shadow_prediction": shadow_log}))

        except Exception as e:
            logger.error(json.dumps({
                "shadow_model_version": version,
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
