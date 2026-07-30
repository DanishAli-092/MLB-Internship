"""
Day 10 - Step 2: Baseline Model

Training a plain Logistic Regression model with default settings so we
have a baseline score to compare against after tuning.
"""

import json

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42  # keeping this fixed so the split is the same every run

# In this dataset, class 0 = malignant, class 1 = benign. sklearn's
# precision/recall/f1 default to scoring class 1 (pos_label=1) unless we
# say otherwise, which would mean we're measuring how well the model
# catches BENIGN cases, not cancer. For a cancer dataset the number that
# actually matters is how many malignant cases we catch, so pos_label is
# set to 0 everywhere below.
POS_LABEL = 0


def load_data():
    try:
        bunch = load_breast_cancer()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")
    return X, y


def train_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # only fit on train data

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    return model, X_test_scaled, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=POS_LABEL),
        "recall": recall_score(y_test, predictions, pos_label=POS_LABEL),
        "f1_score": f1_score(y_test, predictions, pos_label=POS_LABEL),
    }

    print("Baseline model metrics (malignant = positive class):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["malignant", "benign"]))

    return metrics, cm


if __name__ == "__main__":
    X, y = load_data()

    if X is None:
        print("Could not load dataset, stopping.")
    else:
        model, X_test, y_test = train_baseline(X, y)
        metrics, cm = evaluate_model(model, X_test, y_test)

        try:
            results = {"metrics": metrics, "confusion_matrix": cm.tolist()}
            with open("../outputs/baseline_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("\nSaved results to outputs/baseline_results.json")
        except Exception as e:
            print(f"Could not save results: {e}")