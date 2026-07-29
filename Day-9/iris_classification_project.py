"""
Day 9 - Mini Project: Iris Flower Classification System
---------------------------------------------------------
Predicts the species of an Iris flower (Setosa / Versicolor / Virginica)
based on 4 physical measurements, using Logistic Regression.

Bonus: Also trains a Decision Tree and compares performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# STEP 1: Load and explore the dataset

def load_and_explore_data():
    iris = load_iris()

    # Build a readable DataFrame just for exploring/printing
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[label] for label in iris.target]

    print("=" * 55)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 55)
    print(f"Total samples : {df.shape[0]}")
    print(f"Total features: {len(iris.feature_names)}")
    print(f"Feature names : {list(iris.feature_names)}")
    print(f"Target classes: {list(iris.target_names)}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nHow many samples per species:")
    print(df["species"].value_counts())

    return iris, df


# STEP 2: Split data into train/test and scale it

def split_and_scale(iris):
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% of data reserved for testing
        random_state=42,    # reproducible results
        stratify=y            # equal class ratio in train and test
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 55)
    print("STEP 2: TRAIN / TEST SPLIT")
    print("=" * 55)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    return X_train_scaled, X_test_scaled, y_train, y_test


# STEP 3: Train a model (used for both Logistic Regression & Tree)


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


# STEP 4: Evaluate a trained model

def evaluate_model(model_name, model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"EVALUATION: {model_name}")
    print("=" * 55)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Sample Predictions (Actual vs Predicted):")
    for i in range(10):
        actual = class_names[y_test[i]]
        predicted = class_names[y_pred[i]]
        result = "Correct" if actual == predicted else "Wrong"
        print(f"  Actual: {actual:12s} | Predicted: {predicted:12s} | {result}")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": y_pred
    }


 
# STEP 5: Save confusion matrix as an image (for the deliverable)

def save_confusion_matrix_plot(cm, class_names, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nConfusion matrix image saved as: {filename}")


# STEP 6: Compare Logistic Regression vs Decision Tree (Bonus)

def compare_models(result_1, result_2):
    print("\n" + "=" * 55)
    print("BONUS: MODEL COMPARISON")
    print("=" * 55)

    comparison = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1"], 4),
        }
        for r in [result_1, result_2]
    ])
    print(comparison.to_string(index=False))

    winner = result_1 if result_1["accuracy"] >= result_2["accuracy"] else result_2
    print(f"\nBest performing model on this test split: {winner['model_name']}")
    print("Note: Iris is a small, clean dataset, so both models usually score")
    print("similarly high. Decision Trees can overfit more easily on small")
    print("data, while Logistic Regression tends to generalize more smoothly.")

# MAIN

def main():
    iris, df = load_and_explore_data()
    X_train, X_test, y_train, y_test = split_and_scale(iris)
    class_names = iris.target_names

    # ---- Logistic Regression (main model) ----
    log_reg_model = train_logistic_regression(X_train, y_train)
    log_reg_result = evaluate_model(
        "Logistic Regression", log_reg_model, X_test, y_test, class_names
    )
    save_confusion_matrix_plot(
        log_reg_result["confusion_matrix"], class_names,
        "Confusion Matrix - Logistic Regression",
        "confusion_matrix.png"
    )

    # ---- Decision Tree (bonus model) ----
    tree_model = train_decision_tree(X_train, y_train)
    tree_result = evaluate_model(
        
        "Decision Tree", tree_model, X_test, y_test, class_names
    )
    save_confusion_matrix_plot(
        tree_result["confusion_matrix"], class_names,
        "Confusion Matrix - Decision Tree",
        "confusion_matrix_decision_tree.png"
    )

    # ---- Compare both models ----
    compare_models(log_reg_result, tree_result)


if __name__ == "__main__":
    main()