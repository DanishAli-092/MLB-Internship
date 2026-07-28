"""
MLB-INTERNSHIP 

Day 8 - Data Preprocessing
---------------------------
Dataset columns: Student_ID, Name, Age, Program, Python, Mathematics,
Statistics, Machine_Learning, Attendance

Goal: prepare this data so it can be used to train a Linear Regression
model that predicts a student's Average_Score.

Steps:
    1. Load the dataset
    2. Create the Average_Score column from the 4 subject scores
    3. Encode the 'Program' column (it's text: AI, SE, DS)
    4. Choose Features (X) and Target (y)
    5. Train-Test split (80/20)
    6. Feature scaling (Standardization)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os


# STEP 1: Load the dataset
df = pd.read_csv("01_student_performance.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# STEP 2: Create the Average_Score column

# Average_Score is the average of the 4 subject marks: Python, Mathematics,
# Statistics, and Machine_Learning. This will be our TARGET variable
# the thing we want the model to predict.

df["Average_Score"] = (df["Python"] + df["Mathematics"] + df["Statistics"] + df["Machine_Learning"]) / 4

print("\nAverage_Score column created.")
print(df[["Name", "Python", "Mathematics", "Statistics", "Machine_Learning", "Average_Score"]].head())



# STEP 3: Encode the categorical column ('Program')


# 'Program' contains text values: AI, SE, DS. Models can't work with text,
# so we convert it into numbers using Label Encoding.
# Example mapping (alphabetical order): AI -> 0, DS -> 1, SE -> 2

program_encoder = LabelEncoder()
df["Program_Encoded"] = program_encoder.fit_transform(df["Program"])

print("\nProgram encoding mapping:")
for original, encoded in zip(program_encoder.classes_, range(len(program_encoder.classes_))):
    print(f"  {original} -> {encoded}")

# STEP 4: Select Features (X) and Target (y)

# We deliberately do NOT use Python, Mathematics, Statistics, or
# Machine_Learning as features, because Average_Score is directly
# calculated from them -- using them as inputs would make the model
# trivial (it would just re-learn the averaging formula, not actually
# "predict" anything meaningful).
#
# Instead, we predict Average_Score using more general student
# information: Age, Program, and Attendance. This is a more realistic
# ML problem -- "can we predict performance from a student's profile?"

X = df[["Age", "Program_Encoded", "Attendance"]]
y = df["Average_Score"]

print("\nFeatures (X):", list(X.columns))
print("Target (y): Average_Score")

# STEP 5: Train-Test Split (80% train, 20% test)

# We split the data BEFORE scaling. This avoids Data Leakage -- if we
# scaled the full dataset first, the scaler would "see" the test data's
# statistics (mean/std) during training, which artificially inflates
# performance. The test set must stay completely unseen until evaluation.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")

print(f"Testing samples: {len(X_test)}")


# STEP 6: Feature Scaling (Standardization)

# Age, Program_Encoded, and Attendance are on very different numeric
# scales (Age is ~20-23, Attendance is ~75-100, Program_Encoded is 0-2).
# Standardization rescales every feature to mean=0, std=1, so no single
# feature dominates the model just because its raw numbers are bigger.
#
# We FIT the scaler only on X_train, then TRANSFORM both X_train and
# X_test using those same training statistics -- again, to avoid leakage.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling complete.")


# STEP 7: Save the prepared data for the next script

os.makedirs("prepared_data", exist_ok=True)

with open("prepared_data/preprocessed_data.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "program_encoder": program_encoder,
        "feature_names": ["Age", "Program_Encoded", "Attendance"],
    }, f)

print("\nPreprocessing done. Saved to prepared_data/preprocessed_data.pkl")