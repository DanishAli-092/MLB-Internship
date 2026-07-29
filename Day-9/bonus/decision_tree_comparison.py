"""
Day 9 - Bonus: Decision Tree vs Logistic Regression Comparison
-----------------------------------------------------------------
script that trains BOTH models on the Iris dataset and
compares their performance side by side. Kept separate from the main
project so the comparison logic is easy to find and reuse.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names


def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_scores(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-Score": f1_score(y_test, y_pred, average="macro"),
    }


def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train both models
    log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)

    # Score both models
    log_reg_scores = get_scores(log_reg, X_test, y_test)
    tree_scores = get_scores(tree, X_test, y_test)

    # Build a clean comparison table
    results = pd.DataFrame({
        "Logistic Regression": log_reg_scores,
        "Decision Tree": tree_scores
    })

    print("=" * 50)
    print("LOGISTIC REGRESSION vs DECISION TREE")
    print("=" * 50)
    print(results.round(4))

    print("\nObservation:")
    print("On a small, clean dataset like Iris, both models tend to perform")
    print("similarly well. Logistic Regression draws smooth linear decision")
    print("boundaries, while Decision Trees split the data step-by-step and")
    print("can overfit more easily if allowed to grow too deep.")


if __name__ == "__main__":
    main()