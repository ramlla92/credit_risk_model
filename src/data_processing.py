import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# ===============================
# Config
# ===============================
TARGET = "FraudResult"

DROP_COLS = [
    "TransactionId",
    "BatchId",
    "CurrencyCode",
    "CountryCode"
]

# IDs should NEVER go into WoE
ID_COLS = [
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "ProductId",
    "ProviderId",
    "ChannelId"
]


# ===============================
# Data Loading & Cleaning
# ===============================
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_COLS, errors="ignore")


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df.drop(columns=["TransactionStartTime"])


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_amount=("Amount", "std")
        )
        .reset_index()
    )
    return agg_df


# ===============================
# Feature / Target Split
# ===============================
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_feature_types(df: pd.DataFrame, max_unique: int = 30):
    """
    IMPORTANT:
    - Numerical features → scaled
    - Categorical features → WoE ONLY if low cardinality
    """

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove ID columns
    numerical_features = [c for c in numerical_features if c not in ID_COLS]

    categorical_features = [
        c for c in categorical_features
        if c not in ID_COLS and df[c].nunique() <= max_unique
    ]

    return numerical_features, categorical_features


# ===============================
# WoE & IV Transformer
# ===============================
class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_list, target_col=TARGET, iv_threshold=0.02):
        self.feature_list = feature_list
        self.target_col = target_col
        self.iv_threshold = iv_threshold
        self.woe_maps = {}
        self.iv_values = {}

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = y

        total_good = (df[self.target_col] == 0).sum()
        total_bad = (df[self.target_col] == 1).sum()
        eps = 1e-6

        for feature in self.feature_list:
            grouped = (
                df.groupby(feature)[self.target_col]
                .value_counts()
                .unstack(fill_value=0)
            )

            grouped.columns = ["good", "bad"]

            grouped["dist_good"] = grouped["good"] / total_good
            grouped["dist_bad"] = grouped["bad"] / total_bad

            grouped["woe"] = np.log(
                (grouped["dist_good"] + eps) /
                (grouped["dist_bad"] + eps)
            )

            grouped["iv"] = (
                (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
            )

            self.woe_maps[feature] = grouped["woe"].to_dict()
            self.iv_values[feature] = grouped["iv"].sum()

        return self

    def transform(self, X):
        X_trans = X.copy()

        selected_features = [
            f for f, iv in self.iv_values.items()
            if iv >= self.iv_threshold
        ]

        for feature in selected_features:
            X_trans[feature] = X_trans[feature].map(self.woe_maps[feature])

        return X_trans[selected_features]


# ===============================
# Full Feature Preparation
# ===============================
def prepare_features(data_path: str):
    # Load & clean
    df = load_data(data_path)
    df = drop_columns(df)
    df = extract_datetime_features(df)

    agg_df = create_aggregate_features(df)
    df = df.merge(agg_df, on="CustomerId", how="left")

    # Split
    X, y = split_features_target(df)

    # Feature typing (CRITICAL CHANGE)
    num_features, cat_features = get_feature_types(X, max_unique=30)

    # WoE for categorical
    woe_transformer = WoETransformer(cat_features)
    woe_transformer.fit(X, y)
    X_woe = woe_transformer.transform(X)

    # Scale numerical
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_num = pd.DataFrame(
        scaler.fit_transform(num_imputer.fit_transform(X[num_features])),
        columns=num_features,
        index=X.index
    )

    # Final dataset
    X_final = pd.concat([X_num, X_woe], axis=1)

    iv_report = (
        pd.DataFrame(
            list(woe_transformer.iv_values.items()),
            columns=["Feature", "IV"]
        )
        .sort_values("IV", ascending=False)
    )

    return X_final, y, iv_report, woe_transformer, scaler
