import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_processing import prepare_features


def create_rfm_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy target variable using RFM + KMeans clustering
    """

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionStartTime", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    high_risk_cluster = cluster_summary.sort_values(
        by=["Recency", "Frequency", "Monetary"],
        ascending=[False, True, True]
    ).index[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    df = df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    return df


if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/raw/data.csv")

    # Create proxy target
    df = create_rfm_target(df)

    # Generate features
    X, y, iv_report, _, _ = prepare_features(
        data_path="data/raw/data.csv"
    )

    print("Target distribution:")
    print(df["is_high_risk"].value_counts(normalize=True))
