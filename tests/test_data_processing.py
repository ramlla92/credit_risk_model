import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.data_processing import prepare_features, create_aggregate_features

def test_prepare_features_returns_shapes():
    # Dummy CSV with multiple transactions per customer to avoid NaN std_amount
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2],
        'TransactionId': [11, 12, 21, 22],
        'BatchId': [1, 1, 1, 1],
        'CurrencyCode': ['USD', 'USD', 'USD', 'USD'],
        'CountryCode': ['US', 'US', 'US', 'US'],
        'TransactionStartTime': [
            '2025-01-01 10:00:00', '2025-01-01 12:00:00',
            '2025-01-02 10:00:00', '2025-01-02 12:00:00'
        ],
        'Amount': [100, 200, 300, 400],
        'FraudResult': [0, 1, 0, 1]
    })
    dummy_csv = Path("tests/dummy.csv")
    df.to_csv(dummy_csv, index=False)

    X, y, iv_report, woe_transformer, scaler = prepare_features(str(dummy_csv))

    # Assertions
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y)
    assert not X.empty
    assert 'IV' in iv_report.columns or 'Feature' in iv_report.columns

    dummy_csv.unlink()  # Clean up

def test_create_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2],
        'Amount': [100, 200, 300, 400]
    })

    agg_df = create_aggregate_features(df)

    # Should return one row per customer
    assert len(agg_df) == 2

    # Check columns
    expected_cols = ['CustomerId', 'total_amount', 'avg_amount', 'transaction_count', 'std_amount']
    for col in expected_cols:
        assert col in agg_df.columns

    # Check values
    customer1 = agg_df[agg_df['CustomerId'] == 1].iloc[0]
    assert customer1['total_amount'] == 300
    assert customer1['avg_amount'] == 150
    assert customer1['transaction_count'] == 2
    assert round(customer1['std_amount'], 6) == 70.710678  # pandas std uses ddof=1
