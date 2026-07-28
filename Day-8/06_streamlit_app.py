"""
Day 8 - Streamlit App: Student Score Prediction System
-----------------------------------------------------------
Interactive dashboard version of the Mini Project.

Run with:
    python -m streamlit run 06_streamlit_app.py

"""



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Score Prediction", layout="wide")
st.title("Student Score Prediction System")
st.caption("Day 8 Mini Project - Linear Regression with Scikit-learn")

# STEP 1: Load and preprocess data

df = pd.read_csv("Day-8/01_student_performance.csv")

df["Average_Score"] = (
    df["Python"] + df["Mathematics"] + df["Statistics"] + df["Machine_Learning"]
) / 4

program_encoder = LabelEncoder()
df["Program_Encoded"] = program_encoder.fit_transform(df["Program"])

X = df[["Age", "Program_Encoded", "Attendance"]]
y = df["Average_Score"]


# STEP 2: Train-test split, scale, train the model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# SECTION: Dataset preview

with st.expander("Preview dataset"):
    st.dataframe(df[["Name", "Age", "Program", "Attendance", "Average_Score"]])

# SECTION: Evaluation metrics

st.subheader("Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("R2 Score", f"{r2:.2f}")

st.caption(
    "Note: this dataset only has 20 students, so metrics (especially R2) "
    "can look unstable. More data would give a more reliable model."
)

# SECTION: Actual vs Predicted table

st.subheader("Actual vs Predicted Scores")
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2),
    "Difference": (y_test.values - y_pred).round(2),
})
st.dataframe(comparison, use_container_width=True)

# SECTION: Scatter plot

st.subheader("Actual vs Predicted Visualization")

# Wrap the chart in a narrower column so it doesn't stretch across the
# whole page width. Also using a small, fixed figsize so the plot itself
# stays compact regardless of screen size.
chart_col, _ = st.columns([1, 1])
with chart_col:
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=120)
    ax.scatter(y_test, y_pred, color="#4CAF50", edgecolor="black", s=50)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Perfect Prediction")
    ax.set_xlabel("Actual Average Score", fontsize=9)
    ax.set_ylabel("Predicted Average Score", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

# SECTION: Try your own prediction

st.subheader("Try a Live Prediction")

age_input = st.number_input("Age", min_value=15, max_value=40, value=21)
program_input = st.selectbox("Program", options=list(program_encoder.classes_))
attendance_input = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)

if st.button("Predict Average Score"):
    program_encoded_input = program_encoder.transform([program_input])[0]
    new_data = pd.DataFrame(
        [[age_input, program_encoded_input, attendance_input]],
        columns=["Age", "Program_Encoded", "Attendance"]
    )
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]
    st.success(f"Predicted Average Score: {prediction:.2f}")