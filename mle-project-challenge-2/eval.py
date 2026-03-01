"""
Model evaluation script for house price model.

Outputs:
- Printed regression metrics
- Cross-validation scores
- Saved plots:
    - actual_vs_pred.png
    - residuals.png
    - error_distribution.png
- Saved metrics.json
"""

import json
import pickle
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn import model_selection, metrics
from sklearn.compose import TransformedTargetRegressor
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------
# CONFIG
# -----------------------
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
MODEL_DIR = "model/v2"
OUTPUT_DIR = "artifacts_v2/"
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

class GeoCluster(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, X, y=None):
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42
        )
        self.kmeans.fit(X[["lat", "long"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["geo_cluster"] = self.kmeans.predict(
            X[["lat", "long"]]
        )
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, current_year: int):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Ratios
        X["basement_ratio"] = (
            X["sqft_basement"] / X["sqft_living"]
        )

        X["lot_to_living_ratio"] = (
            X["sqft_lot"] / X["sqft_living"]
        )

        # Age features
        X["house_age"] = (
            self.current_year - X["yr_built"]
        )

        X["is_renovated"] = (
            X["yr_renovated"] > 0
        ).astype(int)

        X["years_since_renovation"] = (
            self.current_year
            - X["yr_renovated"].mask(
                X["yr_renovated"] == 0,
                X["yr_built"],
            )
        )

        # Demographic features
        X["urban_density"] = (
            X["urbn_ppltn_qty"] / X["ppltn_qty"]
        )

        X["education_index"] = (
            X["per_bchlr"] + X["per_prfsnl"]
        )

        return X


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pd.read_csv(sales_path,
                       #usecols=sales_column_selection,
                       dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    # merged_data["log_price"] = np.log1p(merged_data["price"])
    # y = merged_data.pop('log_price')
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def evaluate():
    # -----------------------
    # Load data
    # -----------------------
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=42
    )

    # -----------------------
    # Load trained model
    # -----------------------
    model_path = pathlib.Path(MODEL_DIR) / "model.pkl"
    model = pickle.load(open(model_path, "rb"))

    # -----------------------
    # Predictions
    # -----------------------
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # -----------------------
    # Metrics
    # -----------------------
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))

    train_mape = metrics.mean_absolute_percentage_error(y_train, y_pred_train)
    test_mape = metrics.mean_absolute_percentage_error(y_test, y_pred_test)

    train_median_ape = np.median(np.abs((y_train - y_pred_train) / y_train))
    test_median_ape = np.median(np.abs((y_test - y_pred_test) / y_test))

    results = {
        # Core metrics
        "train_r2": metrics.r2_score(y_train, y_pred_train),
        "test_r2": metrics.r2_score(y_test, y_pred_test),

        "train_rmse": train_rmse,
        "test_rmse": test_rmse,

        "train_mae": metrics.mean_absolute_error(y_train, y_pred_train),
        "test_mae": metrics.mean_absolute_error(y_test, y_pred_test),

        # Percentage metrics (business friendly)
        "train_mape": train_mape,
        "test_mape": test_mape,

        "train_median_ape": train_median_ape,
        "test_median_ape": test_median_ape,
    }

    print("\nModel Performance:")
    for k, v in results.items():
        if "mape" in k or "ape" in k:
            print(f"{k}: {v:.4%}")   # format as percentage
        else:
            print(f"{k}: {v:,.4f}")

    # -----------------------
    # Cross Validation
    # -----------------------
    cv_r2 = model_selection.cross_val_score(
        model, X, y, cv=5, scoring="r2"
    )

    cv_rmse = -model_selection.cross_val_score(
        model, X, y, cv=5, scoring="neg_root_mean_squared_error"
    )

    print("\nCross-Validation R-squared Scores:")
    print(cv_r2)
    print(f"Mean CV R-squared: {cv_r2.mean():.4f}")

    print("\nCross-Validation RMSE Scores:")
    print(cv_rmse)
    print(f"Mean CV RMSE: {cv_rmse.mean():,.4f}")

    results["cv_mean_r2"] = cv_r2.mean()
    results["cv_std_r2"] = cv_r2.std()

    results["cv_mean_rmse"] = cv_rmse.mean()
    results["cv_std_rmse"] = cv_rmse.std()

    # -----------------------
    # Save Metrics
    # -----------------------
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # -----------------------
    # Visualization
    # -----------------------

    y_test_plot = y_test
    y_pred_plot = y_pred_test

    # -----------------------
    # Actual vs Predicted
    # -----------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_plot, y_pred_plot, alpha=0.4)

    min_val = min(y_test_plot.min(), y_pred_plot.min())
    max_val = max(y_test_plot.max(), y_pred_plot.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs Predicted Home Prices")

    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_pred.png")
    plt.close()


    # -----------------------
    # Residual Plot
    # -----------------------
    residuals = y_test_plot - y_pred_plot

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_plot, residuals, alpha=0.4)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual (Actual - Predicted) ($)")
    plt.title("Residuals vs Predicted Price")

    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png")
    plt.close()


    # -----------------------
    # Error Distribution
    # -----------------------
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40)

    plt.xlabel("Prediction Error ($)")
    plt.ylabel("Number of Homes")
    plt.title("Distribution of Prediction Errors")

    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png")
    plt.close()

    print("\nEvaluation artifacts saved to:", output_dir)


if __name__ == "__main__":
    evaluate()