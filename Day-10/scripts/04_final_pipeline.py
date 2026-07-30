"""
Day 10 - Step 4: Final Prediction Pipeline

Runs the whole thing from start to finish - load data, train a baseline
model, tune with GridSearchCV, evaluate both, and save a confusion
matrix heatmap comparing them side by side.
"""

import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    make_scorer,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET_NAMES = ["malignant", "benign"]

# class 0 = malignant, class 1 = benign in this dataset. We care about
# catching malignant cases, not benign ones, so pos_label is set to 0
# everywhere instead of relying on sklearn's default of 1.
POS_LABEL = 0


def load_data():
    try:
        bunch = load_breast_cancer()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")

    if X.isnull().any().any():
        print("Dataset has missing values, cannot continue.")
        return None, None

    return X, y


def prepare_splits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_baseline(X_train, y_train):
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def tune_model(X_train, y_train):
    param_grid = [
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]},
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1"], "solver": ["liblinear"]},
    ]
    base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)

    scorer = make_scorer(fbeta_score, beta=2, pos_label=POS_LABEL)

    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring=scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search


def compute_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=POS_LABEL),
        "recall": recall_score(y_test, predictions, pos_label=POS_LABEL),
        "f1_score": f1_score(y_test, predictions, pos_label=POS_LABEL),
    }
    cm = confusion_matrix(y_test, predictions)
    return metrics, cm


def plot_confusion_matrices(baseline_cm, tuned_cm, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrices = [baseline_cm, tuned_cm]
    titles = ["Baseline Model", "Tuned Model (GridSearchCV)"]

    for ax, cm, title in zip(axes, matrices, titles):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, cbar=False, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Running full pipeline: load -> split -> baseline -> tune -> evaluate\n")

    X, y = load_data()

    if X is None:
        print("Could not load dataset, stopping.")
    else:
        X_train, X_test, y_train, y_test = prepare_splits(X, y)

        baseline_model = train_baseline(X_train, y_train)
        baseline_metrics, baseline_cm = compute_metrics(baseline_model, X_test, y_test)

        print("Running GridSearchCV (scoring on F2-score, malignant = positive class)...")
        grid_search = tune_model(X_train, y_train)
        tuned_model = grid_search.best_estimator_
        tuned_metrics, tuned_cm = compute_metrics(tuned_model, X_test, y_test)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F2-score: {grid_search.best_score_:.4f}\n")

        print(f"{'Metric':<12}{'Baseline':>12}{'Tuned':>12}")
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            base_val = baseline_metrics[metric]
            tuned_val = tuned_metrics[metric]
            print(f"{metric:<12}{base_val:>12.4f}{tuned_val:>12.4f}")

        try:
            save_path = "../outputs/confusion_matrix_comparison.png"
            plot_confusion_matrices(baseline_cm, tuned_cm, save_path)
            print(f"\nConfusion matrix heatmap saved to {save_path}")
        except Exception as e:
            print(f"Could not save confusion matrix image: {e}")