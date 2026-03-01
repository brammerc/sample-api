from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin



DROP_COLS = [
    "urbn_ppltn_qty", "ppltn_qty", 
    "per_bchlr", "per_prfsnl", 
    "sqft_basement", "yr_renovated"
]

# ============================================================
# Custom Geo Cluster Transformer 
# ============================================================

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
        # X["house_age"] = (
        #     self.current_year - X["yr_built"]
        # )

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

        X = X.drop(columns=DROP_COLS)

        return X