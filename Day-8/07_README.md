# Day 8 - Data Preparation & First Machine Learning Model

## Overview
Today's task moved from data cleaning/visualization (Day 7) into the actual
Machine Learning workflow: preparing data correctly, training a Linear
Regression model, and evaluating how well it predicts a student's
`Average_Score`.

## Folder Structure
```
Day-8/
├── 05_charts/
│   └── actual_vs_predicted_scatter.png
├── prepared_data/
├── recording/
├── 01_student_performance.csv
├── 02_data_preprocessing.py
├── 03_linear_regression_model.py
├── 04_student_score_prediction_app.py
├── 06_streamlit_app.py
├── 07_README.md
├── actual_vs_predicted.csv
└── requirements.txt
```

## Files in this folder
| File | Purpose |
|---|---|
| `01_student_performance.csv` | Dataset used throughout the day |
| `02_data_preprocessing.py` | Encoding, target creation, train-test split, scaling |
| `03_linear_regression_model.py` | Trains the model and prints evaluation metrics |
| `04_student_score_prediction_app.py` | End-to-end Mini Project (single-run pipeline) |
| `05_charts/` | Actual vs Predicted scatter plot |
| `prepared_data/` | Processed/prepared dataset outputs from the preprocessing step |
| `recording/` | Walkthrough recording explaining the implementation |
| `06_streamlit_app.py` | Interactive dashboard version of the project |
| `actual_vs_predicted.csv` | Saved predictions vs actual values for evaluation |
| `requirements.txt` | Python dependencies for the project |
| `07_README.md` | This file |

## What I Learned About Data Preprocessing
- Raw data is rarely ready for a model. Categorical columns (like Gender or
  ParentalEducation) need to be converted to numbers using **Label
  Encoding** before a model can use them.
- Creating a clear **target variable** (`Average_Score`) upfront makes the
  rest of the pipeline much simpler — everything else becomes "features
  used to predict this one number."
- **Feature Scaling (Standardization)** matters for algorithms like Linear
  Regression because features with larger numeric ranges (e.g. attendance
  percentage) can otherwise dominate features with smaller ranges (e.g.
  study hours), even if both are equally important.
- **Data Leakage** happens when information from the test set accidentally
  influences training — for example, fitting a scaler on the *entire*
  dataset before splitting. I made sure to always `fit` the scaler only on
  the training set and `transform` (not `fit_transform`) the test set.

## Why Train-Test Splitting Is Important
If a model is evaluated on the same data it was trained on, it can simply
"memorize" the answers instead of learning general patterns — this gives a
falsely high performance score. Splitting the data (80% train / 20% test)
means the model is tested on data it has never seen before, giving an
honest measure of how it will perform on new, real-world students.

## Evaluation Metrics Used
- **MAE (Mean Absolute Error)** — average magnitude of prediction error, in
  the same units as the score itself. Easy to interpret directly.
- **MSE (Mean Squared Error)** — similar to MAE, but squares errors first,
  so it penalizes larger mistakes more heavily.
- **R² Score** — measures how much of the variation in `Average_Score` the
  model is able to explain (closer to 1.0 = better fit).

## Dataset & Feature Choice
The dataset (`01_student_performance.csv`) has 20 students with columns:
`Student_ID, Name, Age, Program, Python, Mathematics, Statistics,
Machine_Learning, Attendance`.

- **Target (y):** `Average_Score` — created by averaging the four subject
  columns (Python, Mathematics, Statistics, Machine_Learning).
- **Features (X):** `Age`, `Program` (encoded), `Attendance`.
- I deliberately did **not** use the individual subject scores as features,
  since `Average_Score` is directly calculated from them — including them
  would make the model trivially "cheat" by re-learning the averaging
  formula instead of genuinely predicting performance from a student's
  profile (age, program, attendance).

## Model Performance & Observations (from this run)
| Metric | Value |
|---|---|
| MAE | 2.53 |
| MSE | 13.46 |
| R² Score | 0.80 |

- An R² of 0.80 means the model explains about 80% of the variation in
  students' average scores using just Age, Program, and Attendance —
  Attendance turned out to have by far the strongest coefficient, which
  makes sense (more class attendance → better grades).
- MAE of ~2.53 means predictions are, on average, within about 2.5 marks
  of the actual average score — quite reasonable given only 20 students.
- With only 20 rows and a 4-row test set, these metrics can shift a lot
  if the random split changes. This dataset is really only enough to
  demonstrate the ML workflow, not to draw strong real-world conclusions.
- The scatter plot (`05_charts/actual_vs_predicted_scatter.png`) shows
  most predicted points sitting close to the red diagonal line (perfect
  prediction), with one point (a lower-attendance student) predicted a
  bit high — likely because the model hasn't seen many low-attendance
  examples to learn from.

## How to Run
```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Preprocess the data
python 02_data_preprocessing.py

# 2. Train and evaluate the model
python 03_linear_regression_model.py

# 3. Run the full mini project end-to-end
python 04_student_score_prediction_app.py

# 4. Launch the interactive dashboard
streamlit run 06_streamlit_app.py
```

## 📚 Learning Outcomes
Through this project, I learned:

- Data preprocessing techniques
- Label Encoding and One-Hot Encoding
- Feature scaling using StandardScaler
- The importance of train-test splitting
- Training a Linear Regression model
- Evaluating a regression model using MAE, MSE, RMSE, and R² Score
- Building and deploying a machine learning application using Streamlit