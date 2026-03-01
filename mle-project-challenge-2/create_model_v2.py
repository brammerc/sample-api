
import json
import pickle
import pathlib
from typing import List, Tuple
from datetime import date
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, neighbors, pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn import set_config
set_config(transform_output="pandas")

from xgboost import XGBRegressor, plot_importance

from transformers import GeoCluster, FeatureEngineer

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
OUTPUT_DIR = "model/v2"  # Directory where output artifacts will be saved

DROP_COLS = [
    "urbn_ppltn_qty", "ppltn_qty", 
    "per_bchlr", "per_prfsnl", 
    "sqft_basement", "yr_renovated"
]

FINAL_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", 
    "waterfront", "view", "condition", "grade", "sqft_above", "yr_built", 
    "lat", "long", "sqft_living15", "sqft_lot15", "sbrbn_ppltn_qty", 
    "farm_ppltn_qty", "non_farm_qty", "medn_hshld_incm_amt", "hous_val_amt", 
    "edctn_less_than_9_qty", "edctn_9_12_qty", "edctn_prfsnl_qty", "per_urbn", 
    "per_sbrbn", "per_farm", "per_non_farm", "per_9_to_12", "per_some_clg", 
    "per_assoc", 
    "urbn_ppltn_qty", "ppltn_qty", 
    "per_bchlr", "per_prfsnl", 
    "sqft_basement", "yr_renovated"
]

CURRENT_YEAR = date.today().year


def load_data(
    sales_path: str,
    demographics_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:

    data = pd.read_csv(
        sales_path,
        dtype={"zipcode": str},
    )

    data["date"] = pd.to_datetime(data["date"])

    demographics = pd.read_csv(
        demographics_path,
        dtype={"zipcode": str},
    )

    merged = data.merge(
        demographics,
        how="left",
        on="zipcode",
    )

    # Target
    merged["price_per_sqft"] = (
        merged["price"] / merged["sqft_living"]
    )
    merged["log_price_per_sqft"] = np.log1p(merged["price_per_sqft"])
    merged["log_price"] = np.log1p(merged["price"])
    
    print(merged.columns)
    y = merged.pop("price")

    merged = merged.drop(columns=["id", "date", "log_price", "price_per_sqft", "log_price_per_sqft", "zipcode"])
    merged = merged[FINAL_COLS]
    return merged, y


# ============================================================
# Training Pipeline
# ============================================================

def main():

    X, y = load_data(
        SALES_PATH,
        DEMOGRAPHICS_PATH,
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                preprocessing.RobustScaler(),
                make_column_selector(dtype_include=np.number),
            )
        ]
    )

    # Swapped KNN for XGBoost
    xgb = XGBRegressor(
        n_estimators=500,        # Higher number of trees, controlled by early stopping or learning rate
        random_state=42,
        tree_method='hist'       # Faster histogram-based algorithm
    )

    pipe = pipeline.Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer(CURRENT_YEAR)),
            ("geo_cluster", GeoCluster()), 
            ("preprocess", preprocessor),
            ("model", xgb), 
        ]
    )


    # Wrap the pipeline in TransformedTargetRegressor
    # func=np.log1p will be applied to y before training
    # inverse_func=np.expm1 will be applied to y_pred automatically
    full_model = TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Hyperparameter tuning
    # IMPORTANT: Because we wrapped the pipeline, you must add the 
    # prefix "regressor__" to all your existing parameter keys.
    param_grid = {
        f"regressor__{k}": v for k, v in {
            "geo_cluster__n_clusters": [10, 15, 20],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.01, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }.items()
    }
    # {'regressor__geo_cluster__n_clusters': 10, 'regressor__model__colsample_bytree': 1.0, 
    # 'regressor__model__learning_rate': 0.1, 'regressor__model__max_depth': 4, 'regressor__model__subsample': 0.8}

    grid = GridSearchCV(
        full_model,
        param_grid,
        cv=3, 
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    best_model = grid.best_estimator_

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(output_dir / "model_features.json", "w") as f:
        json.dump(list(X_train.columns), f)

    
    # Feature importances
    inner_pipeline = best_model.regressor_
    xgb_model = inner_pipeline.named_steps["model"]
    # Get feature names AFTER preprocessing
    model_features = inner_pipeline.named_steps["preprocess"].get_feature_names_out()
    print("Best model pipeline features:", model_features.tolist())

    # Attach list of names to the booster
    xgb_model.get_booster().feature_names = model_features
    # Get feature importances
    importances = xgb_model.feature_importances_

    #  Create sorted dataframe
    feat_importance_df = pd.DataFrame({
        "feature": model_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    feat_importance_df.to_csv(output_dir / "feature_importances.csv", index=False)

    ####################
    plt.figure()
    plot_importance(
        xgb_model,
        max_num_features=20,
        importance_type="gain"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()

    # explainer = shap.TreeExplainer(xgb_model)
    # shap_values = explainer.shap_values(
    #     inner_pipeline[:-1].transform(X_train)
    # )

    # shap.summary_plot(shap_values, feature_names=model_features)

    print("Model training complete. Artifacts saved.")


if __name__ == "__main__":
    main()

