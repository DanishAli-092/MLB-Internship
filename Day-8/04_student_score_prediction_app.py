"""
Day 8 - Mini Project: Student Score Prediction System
--------------------------------------------------------
A single script that runs the complete workflow from start to finish:
load data -> preprocess -> train model -> predict -> evaluate -> visualize.

Predicts a student's Average_Score using Age, Program, and Attendance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load the dataset

df = pd.read_csv("01_student_performance.csv")
print(f"Loaded dataset with {len(df)} students.\n")

# STEP 2: Preprocess the data

# Target: Average_Score = average of the 4 subject marks
df["Average_Score"] = (
    df["Python"] + df["Mathematics"] + df["Statistics"] + df["Machine_Learning"]
) / 4

# Encode Program (AI, SE, DS) into numbers
program_encoder = LabelEncoder()
df["Program_Encoded"] = program_encoder.fit_transform(df["Program"])

# Features: Age, Program, Attendance (NOT the subject scores themselves,
# since Average_Score is directly calculated from those)
X = df[["Age", "Program_Encoded", "Attendance"]]
y = df["Average_Score"]

# STEP 3: Train-Test Split (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Scale features (fit on train only, to avoid data leakage)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 5: Train the Linear Regression model

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# STEP 6: Predict on the test set

y_pred = model.predict(X_test_scaled)

# STEP 7: Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--------- Model Evaluation --------------------------")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
print("---------------------------------------------------------\n")

# STEP 8: Actual vs Predicted comparison table

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2),
    "Difference": (y_test.values - y_pred).round(2),
})
print("Actual vs Predicted Scores:")
print(comparison)

# STEP 9: Visualize predictions (scatter plot)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color="#4CAF50", edgecolor="black", s=80, label="Predictions")

# Diagonal red line = where a perfect prediction would land (Actual == Predicted)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction Line")

plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Actual vs Predicted Student Scores")
plt.legend()
plt.tight_layout()
plt.savefig("05_charts/actual_vs_predicted_scatter.png")
print("\nScatter plot saved to 05_charts/actual_vs_predicted_scatter.png")