import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path


SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
OUTPUT_DIR = "artifacts_v2/"
current_year = date.today().year

def load_merged():
    sales = pd.read_csv(SALES_PATH, dtype={"zipcode": str})
    sales["date"] = pd.to_datetime(sales["date"])

    demo = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})

    merged = sales.merge(demo, how="left", on="zipcode")

    merged["price_per_sqft"] = (
        merged["price"] / merged["sqft_living"]
    )
    merged["log_price"] = np.log1p(merged["price"])

    # Ratios
    merged["basement_ratio"] = (
        merged["sqft_basement"] / merged["sqft_living"]
    )

    merged["lot_to_living_ratio"] = (
        merged["sqft_lot"] / merged["sqft_living"]
    )

    # Age features
    merged["house_age"] = (
        current_year - merged["yr_built"]
    )

    merged["is_renovated"] = (
        merged["yr_renovated"] > 0
    ).astype(int)

    merged["years_since_renovation"] = (
        current_year
        - merged["yr_renovated"].mask(
            merged["yr_renovated"] == 0,
            merged["yr_built"],
        )
    )

    # Demographic features
    merged["urban_density"] = (
        merged["urbn_ppltn_qty"] / merged["ppltn_qty"]
    )

    merged["education_indemerged"] = (
        merged["per_bchlr"] + merged["per_prfsnl"]
    )

    merged = merged.drop(columns=["id"])

    merged["month"] = merged["date"].dt.month
    merged["year"] = merged["date"].dt.year
    return merged


def run_eda(df):

    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== SUMMARY STATS =====")
    print(df.describe().T)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # =========================
    # Target distribution
    # =========================
    plt.figure()
    plt.hist(df["price"].dropna(), bins=50)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "price_dist.png")
    plt.close()
    # =========================
    # Log target distribution
    # =========================
    plt.figure()
    plt.hist(df["log_price"].dropna(), bins=50)
    plt.title("Log Price Distribution")
    plt.xlabel("Log(Price + 1)")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "log_price_dist.png")
    plt.close()

    # =========================
    # Correlation with target
    # =========================
    corr = df.corr(numeric_only=True)

    target_corr = corr["log_price"].sort_values(ascending=False)

    print("\n===== CORRELATION WITH TARGET =====")
    print(target_corr.head(20))

    print("\n===== MOST NEGATIVE CORRELATIONS =====")
    print(target_corr.tail(20))

    # =========================
    # One-Hot Encoding Discovery
    # =========================
    print("\n===== ONE-HOT ENCODING CANDIDATES =====")
    
    # Identify non-numeric columns OR numeric columns with very few values (e.g., floors, condition)
    # Threshold: Usually 2 to 10 unique values is the "sweet spot" for OHE
    ohe_threshold = 10
    
    potential_cols = []
    for col in df.columns:
        # Skip the target variable
        if col in ["price", "log_price"]:
            continue
            
        n_unique = df[col].nunique()
        
        # Criteria: Object/Category types OR small integer sets
        if n_unique <= ohe_threshold:
            potential_cols.append({'column': col, 'nunique': n_unique, 'dtype': df[col].dtype})
            
    ohe_df = pd.DataFrame(potential_cols)
    if not ohe_df.empty:
        print(ohe_df.sort_values(by="nunique"))
    else:
        print("No low-cardinality columns found.")


    # =========================
    # Pairs Plot
    # =========================
    print("\n===== GENERATING PAIRS PLOT =====")
    # Pick top 5 numerical features for clarity
    plot_cols = target_corr.head(5).index.tolist()
    n = len(plot_cols)
    
    fig, axes = plt.subplots(n, n, figsize=(15, 15), tight_layout=True)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            col_x = plot_cols[j]
            col_y = plot_cols[i]
            
            if i == j:
                # Diagonal: Histogram
                ax.hist(df[col_x].dropna(), bins=20, color='skyblue', edgecolor='black')
            else:
                # Off-diagonal: Scatter plot
                ax.scatter(df[col_x], df[col_y], alpha=0.3, s=1)
            
            # Labeling only the outer edges to keep it clean
            if i == n - 1:
                ax.set_xlabel(col_x)
            if j == 0:
                ax.set_ylabel(col_y)
                
            # Keep font sizes small for readability
            ax.tick_params(labelsize=8)

    plt.suptitle("Feature Pairs Matrix", y=1.02, fontsize=16)
    plt.savefig(output_dir / "pairs_plot.png")
    plt.close()


    # =========================
    # Time Series Analysis
    # =========================
    if 'date' in df.columns:
        print("\n===== TIME SERIES TRENDS =====")
        
        # Set date as index for easier resampling
        temp_df = df.set_index('date')
        
        # Resample by Month (M) or Quarter (Q) for smoother lines
        # 'price' for average cost, 'sqft_living' for volume/count
        monthly_trends = temp_df.resample('M').agg({
            'price': 'mean',
            'log_price': 'count' # Using count to see volume of sales
        }).rename(columns={'log_price': 'sales_volume'})

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Average Price
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Avg Sale Price', color=color)
        ax1.plot(monthly_trends.index, monthly_trends['price'], color=color, marker='o', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for Sales Volume
        ax2 = ax1.twinx()
        color = 'tab:gray'
        ax2.set_ylabel('Number of Sales', color=color)
        ax2.bar(monthly_trends.index, monthly_trends['sales_volume'], color=color, alpha=0.3, width=20)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Price Trends and Sales Volume Over Time")
        fig.tight_layout()
        plt.savefig(output_dir / "sales_by_month.png")
        plt.close()

    # =========================
    # Seasonality Analysis
    # =========================
    seasonal_data = df.groupby('month')['price'].median()
    
    plt.figure(figsize=(10, 5))
    seasonal_data.plot(kind='bar', color='skyblue')
    # Just start the axis at 400,000
    plt.ylim(400000, seasonal_data.max() * 1.05)
    plt.title("Median Price by Month (Seasonality)")
    plt.xlabel("Month (1=Jan, 12=Dec)")
    plt.ylabel("Median Price (Starting at $400k)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "med_price_month.png")
    plt.close()




    # =========================
    # Top Growing Zipcodes
    # =========================
    # Pivot to get avg price per zip per year
    zip_growth = df.pivot_table(
        values='price', 
        index='year', 
        columns='zipcode', 
        aggfunc='mean'
    )

    # Calculate % growth from first year to last year available
    growth_pct = ((zip_growth.iloc[-1] - zip_growth.iloc[0]) / zip_growth.iloc[0]).sort_values(ascending=False)
    top_10_zips = growth_pct.head(10).index

    plt.figure(figsize=(12, 6))
    for zip_code in top_10_zips:
        plt.plot(zip_growth.index, zip_growth[zip_code], marker='o', label=f"Zip: {zip_code}")

    plt.title("Price Growth: Top 10 Performing Zipcodes")
    plt.xlabel("Year")
    plt.ylabel("Average Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "zipcode_price_change.png")
    plt.close()


    # =========================
    # Correlation heatmap (top features only)
    # =========================
    top_features = (
        target_corr.abs()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    top_corr_matrix = df[top_features].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(top_corr_matrix)
    plt.title("Top Correlated Features")
    plt.xticks(
        ticks=range(len(top_features)),
        labels=top_features,
        rotation=90
    )
    plt.yticks(
        ticks=range(len(top_features)),
        labels=top_features
    )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()


def drop_high_corr(df, threshold=0.9):
    corr = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Identify pairs and print them
    print(f"{'Feature 1':<20} {'Feature 2':<20} {'Correlation'}")
    print("-" * 55)
    
    to_drop = set()
    for column in upper.columns:
        for index, row in upper.iterrows():
            if upper.loc[index, column] > threshold:
                print(f"{index:<20} {column:<20} {upper.loc[index, column]:.4f}")
                to_drop.add(column)

    print(f"\nTotal columns to drop: {list(to_drop)}")
    return df.drop(columns=to_drop)

if __name__ == "__main__":
    df = load_merged()
    run_eda(df)
    df = drop_high_corr(df, threshold=0.9)
    print(df.columns)
