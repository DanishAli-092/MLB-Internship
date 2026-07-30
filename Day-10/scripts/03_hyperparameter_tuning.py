"""
Day 10 - Step 3: Hyperparameter Tuning

Using GridSearchCV to find better settings for Logistic Regression
instead of just guessing, then comparing the tuned model against the
baseline from script 2.
"""

import json
import warnings

import pandas as pd
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

# class 0 = malignant, class 1 = benign in this dataset. We care about
# catching malignant cases, not benign ones, so pos_label is set to 0
# everywhere instead of relying on sklearn's default of 1.
POS_LABEL = 0


def load_and_split():
    try:
        bunch = load_breast_cancer()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None

    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def run_grid_search(X_train, y_train):
    param_grid = [
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]},
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1"], "solver": ["liblinear"]},
    ]

    base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)

    # Plain F1 treats missing a cancer case (false negative) and a false
    # alarm (false positive) as equally bad, but they aren't - a missed
    # cancer case is much worse than an extra test. F2-score fixes this
    # by weighting recall twice as much as precision, so GridSearchCV
    # picks parameters that are better at catching malignant cases,
    # without going as far as optimizing recall alone (which could just
    # predict "malignant" for everything to get a perfect recall score).
    scorer = make_scorer(fbeta_score, beta=2, pos_label=POS_LABEL)

    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring=scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=POS_LABEL),
        "recall": recall_score(y_test, predictions, pos_label=POS_LABEL),
        "f1_score": f1_score(y_test, predictions, pos_label=POS_LABEL),
    }
    cm = confusion_matrix(y_test, predictions)
    return metrics, cm


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()

    if X_train is None:
        print("Could not load dataset, stopping.")
    else:
        print("Running GridSearchCV with 5-fold cross validation...")
        print("Scoring on F2-score (malignant = positive class)\n")
        grid_search = run_grid_search(X_train, y_train)

        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validated F2-score: {grid_search.best_score_:.4f}")

        tuned_model = grid_search.best_estimator_
        tuned_metrics, tuned_cm = evaluate_model(tuned_model, X_test, y_test)

        print("\nTuned model metrics on test set (malignant = positive class):")
        for name, value in tuned_metrics.items():
            print(f"  {name}: {value:.4f}")

        try:
            with open("../outputs/baseline_results.json") as f:
                baseline_results = json.load(f)
            baseline_metrics = baseline_results["metrics"]

            print("\nBaseline vs Tuned comparison:")
            print(f"{'Metric':<12}{'Baseline':>12}{'Tuned':>12}{'Change':>12}")
            for metric_name in baseline_metrics:
                base_val = baseline_metrics[metric_name]
                tuned_val = tuned_metrics[metric_name]
                change = tuned_val - base_val
                print(f"{metric_name:<12}{base_val:>12.4f}{tuned_val:>12.4f}{change:>+12.4f}")

        except FileNotFoundError:
            print("\nbaseline_results.json not found - run 02_baseline_model.py first to compare.")
        except Exception as e:
            print(f"\nCould not load baseline results: {e}")

        try:
            output = {
                "best_params": grid_search.best_params_,
                "best_cv_f2": grid_search.best_score_,
                "tuned_metrics": tuned_metrics,
                "tuned_confusion_matrix": tuned_cm.tolist(),
            }
            with open("../outputs/tuning_results.json", "w") as f:
                json.dump(output, f, indent=2)
            print("\nSaved results to outputs/tuning_results.json")
        except Exception as e:
            print(f"Could not save results: {e}")