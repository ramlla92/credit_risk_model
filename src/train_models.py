import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

from .data_processing import prepare_features


RANDOM_STATE = 42

if __name__ == "__main__":
    # Prepare features and target
    X, y, iv_report, woe_transformer, scaler = prepare_features("data/raw/data.csv")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment("Credit_Risk_Modeling")

    best_model = None
    best_score = 0

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
        }
        print("Logistic Regression Metrics:", metrics)
        mlflow.log_params(lr.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "model")

        if metrics["roc_auc"] > best_score:
            best_model = lr
            best_score = metrics["roc_auc"]

    # --- Random Forest ---
    with mlflow.start_run(run_name="Random_Forest"):
        rf = RandomForestClassifier(random_state=RANDOM_STATE)

        # Hyperparameter tuning
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        rf_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=5, cv=3,
            scoring="roc_auc", random_state=RANDOM_STATE
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_

        y_pred = best_rf.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1])
        }
        print("Random Forest Metrics:", metrics)
        mlflow.log_params(rf_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "model")

        if metrics["roc_auc"] > best_score:
            best_model = best_rf
            best_score = metrics["roc_auc"]

    # Register the best model
    mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="Credit_Risk_Model")
    print(f"Best model registered with ROC-AUC: {best_score:.4f}")
