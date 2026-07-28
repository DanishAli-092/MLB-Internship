"""
Day 8 - Linear Regression Model
---------------------------------
This script loads the prepared data (from 02_data_preprocessing.py),
trains a Linear Regression model, and evaluates how well it predicts
Average_Score using Age, Program, and Attendance.

Steps:
    1. Load preprocessed data
    2. Train the model
    3. Predict on the test set
    4. Compare Actual vs Predicted
    5. Calculate MAE, MSE, R2
    
"""

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: Load the preprocessed data

with open("prepared_data/preprocessed_data.pkl", "rb") as f:
    
    data = pickle.load(f)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Preprocessed data loaded.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# STEP 2: Train the Linear Regression model

# Average_Score is a continuous number, not a category, so this is a
# REGRESSION problem (not classification). Linear Regression tries to find
# the best straight-line relationship between our features
# (Age, Program, Attendance) and the target (Average_Score).

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained.")
print("Coefficients (Age, Program_Encoded, Attendance):", model.coef_)
print("Intercept:", model.intercept_)

# STEP 3: Predict on the test set

y_pred = model.predict(X_test)

# STEP 4: Compare Actual vs Predicted

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2),
    "Difference": (y_test.values - y_pred).round(2),
})

print("\nActual vs Predicted:")

print(comparison)

comparison.to_csv("actual_vs_predicted.csv", index=False)

print("\nSaved comparison table to actual_vs_predicted.csv")


# STEP 5: Evaluate the model
# MAE - average error size, in the same units as Average_Score (marks).
# MSE - like MAE, but squares errors first, so big mistakes count more.
# R2  - how much of the variation in Average_Score the model explains.
#       1.0 = perfect, 0.0 = model is no better than guessing the mean.
#       With only 20 rows of data, don't expect a very high R2 -- small
#       datasets make regression metrics noisy.

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--------- Model Evaluation --------=--------")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
print("----------------------------------=------------")

# Save the trained model for reuse in the mini project / streamlit app

with open("prepared_data/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to prepared_data/trained_model.pkl")