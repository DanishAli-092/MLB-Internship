# Day 9 - Model Evaluation & Classification

## Overview
Yesterday's task (Day 8) covered the basic ML workflow using Linear Regression on a Regression problem. Today's task moves into Classification — predicting a category/label instead of a number — along with learning how to properly evaluate any classification model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## What is Classification?
Classification is a supervised Machine Learning technique where the model predicts a **category or class label** for a given input, instead of predicting a continuous number. The model learns from labeled examples during training, then assigns new, unseen inputs to one of the known categories. Example: predicting whether an email is spam or not spam, or predicting which species an Iris flower belongs to based on its measurements.

## Folder Structure
```
Day-9/
├── bonus/
│   └── decision_tree_comparison.py
├── classification_practice.py
├── iris_classification_project.py
├── streamlit_app.py
├── confusion_matrix.png
├── confusion_matrix_decision_tree.png
├── requirements.txt
└── README.md
```

## Files in this folder

| File | Purpose |
|---|---|
| `classification_practice.py` | Trains a Logistic Regression model on the Iris dataset and evaluates it using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix |
| `iris_classification_project.py` | Mini Project — full end-to-end pipeline: load data, explore, train, predict, evaluate |
| `bonus/decision_tree_comparison.py` | Bonus — trains a Decision Tree and compares it against Logistic Regression |
| `streamlit_app.py` | Interactive dashboard version of the project (dataset explorer, metrics, live prediction) |
| `confusion_matrix.png` | Confusion Matrix screenshot for Logistic Regression |
| `confusion_matrix_decision_tree.png` | Confusion Matrix screenshot for Decision Tree (bonus) |
| `requirements.txt` | Python dependencies for the project |
| `README.md` | This file |

## Difference between Regression and Classification

| Aspect | Regression (Day 8) | Classification (Day 9) |
|---|---|---|
| Output type | Continuous number (e.g. average score) | Discrete category (e.g. species) |
| Example algorithm | Linear Regression | Logistic Regression, Decision Tree |
| Evaluation metrics | MAE, MSE, R² | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Question it answers | "How much?" | "Which class?" |

## Why Model Evaluation Matters
Just like train-test splitting in Day 8 prevented the model from "memorizing" answers, using multiple classification metrics (not just Accuracy) prevents a false sense of confidence. A model can hit 90% accuracy while still completely failing to catch one specific class — Precision, Recall, and the Confusion Matrix are what expose that.

## Evaluation Metrics Used
- **Accuracy** — Percentage of total predictions that were correct overall.
- **Precision** — Out of everything predicted as a certain class, how many were actually that class (controls false alarms).
- **Recall** — Out of all actual samples of a class, how many the model correctly caught (controls missed cases).
- **F1-Score** — Harmonic mean of Precision and Recall, useful when you need a balance between the two.
- **Confusion Matrix** — A grid comparing actual vs predicted labels, showing exactly where the model succeeds or fails per class.

## Dataset & Feature Choice
**Dataset used:** Iris dataset (built into scikit-learn) — 150 samples, 3 target classes (Setosa, Versicolor, Virginica), 50 samples per class (perfectly balanced).

**Features (X):** Sepal length, Sepal width, Petal length, Petal width (all 4 columns used).

**Target (y):** Species (Setosa / Versicolor / Virginica).

## Model Performance & Observations (from this run)

| Metric | Logistic Regression | Decision Tree (Bonus) |
|---|---|---|
| Accuracy | 93.3% | 93.3% |
| Precision (macro) | 93.3% | 93.3% |
| Recall (macro) | 93.3% | 93.3% |
| F1-Score (macro) | 93.3% | 93.3% |

- **Setosa** is predicted correctly 100% of the time in both models — it is clearly separable from the other two species based on petal size alone.
- All misclassifications happen between **Versicolor** and **Virginica**, since these two species genuinely overlap in petal/sepal measurements. This is expected behavior, not a sign of a broken model.
- Both models perform almost identically on this dataset. Iris is small and clean, so a linear model (Logistic Regression) and a non-linear model (Decision Tree) converge to similar results. Decision Trees are more prone to overfitting on small datasets, so Logistic Regression is generally the safer default here.
- The Confusion Matrix (see `confusion_matrix.png`) confirms this: the only non-zero off-diagonal values are between Versicolor and Virginica.

## How to Run

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Run the classification practice script
python classification_practice.py

# 2. Run the Iris mini project (main deliverable)
python iris_classification_project.py

# 3. Run the bonus Decision Tree comparison
python bonus/decision_tree_comparison.py

# 4. Launch the interactive Streamlit dashboard
streamlit run streamlit_app.py
```

## 📚 Learning Outcomes
Through this project, I learned:
- What Classification is and how it differs from Regression
- How to train and evaluate a Logistic Regression classifier
- How to read and interpret a Confusion Matrix
- Why Accuracy alone can be misleading, and when Precision/Recall matter more
- How Decision Trees compare to Logistic Regression on the same data
- Building and deploying an interactive classification app using Streamlit