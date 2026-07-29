"""
Day 9 - Classification Practice
--------------------------------
Goal: Train a simple classification model and understand what each
evaluation metric (Accuracy, Precision, Recall, F1-Score) actually means.

Dataset used: Iris dataset (built into sklearn)
Task type: Multi-class Classification (Setosa / Versicolor / Virginica)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def load_data():
    """Load the Iris dataset and return features + labels."""
    data = load_iris()
    X = data.data                # features: sepal/petal length & width
    y = data.target              # labels: 0=setosa, 1=versicolor, 2=virginica
    return X, y, data.target_names

    

def prepare_data(X, y):
    """Split data into train/test sets and scale the features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 80% train, 20% test
        random_state=42,      # keeps results reproducible
        stratify=y             # keeps class ratio same in train & test
    )

    # Logistic Regression works better when features are on the same scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """Predict on test data and print all evaluation metrics."""
    y_pred = model.predict(X_test)

    # Note: Iris has 3 classes, so we use "macro" average -> it calculates
    # the metric for each class separately, then takes the plain average.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(cm)
    print("(rows = actual class, columns = predicted class)")

    print("\n===== FULL CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return accuracy, precision, recall, f1, cm


def explain_metrics():
    """Print a plain-English explanation of what each metric means."""
    print("\n===== WHAT DO THESE METRICS MEAN? =====")
    print("Accuracy  -> Out of all predictions, how many were correct overall.")
    print("             Not fully reliable alone if classes are imbalanced.")
    print()
    print("Precision -> Out of everything the model predicted as a certain")
    print("             class, how many were actually that class. High")
    print("             precision means fewer false alarms.")
    print()
    print("Recall    -> Out of all the ACTUAL samples of a class, how many")
    print("             did the model correctly catch. High recall means")
    print("             fewer real cases were missed.")
    print()
    print("F1-Score  -> A balance between Precision and Recall. Useful when")
    print("             you care about both false alarms and missed cases.")


def main():
    X, y, class_names = load_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {list(class_names)}")

    X_train, X_test, y_train, y_test = prepare_data(X, y)

    model = train_model(X_train, y_train)
    print("\nModel trained successfully (Logistic Regression).")

    evaluate_model(model, X_test, y_test, class_names)
    explain_metrics()


if __name__ == "__main__":
    main()