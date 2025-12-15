from data_processing import prepare_features

if __name__ == "__main__":
    # Run full Task 3 pipeline with WoE & IV
    X, y, iv_report, woe_transformer, scaler = prepare_features(
        data_path="data/raw/data.csv"
    )

    print(" Feature matrix shape:", X.shape)
    print(" Target shape:", y.shape)
    print("\nTop features by IV:")
    print(iv_report.head(10))
