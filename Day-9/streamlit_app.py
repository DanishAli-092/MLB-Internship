"""
Day 9 - Iris Flower Classification System (Streamlit Web App)

"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Iris Classification System", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classification System")
st.caption("Day 9 - MLB Internship | Logistic Regression vs Decision Tree")


# Load and prepare data (runs once, cached for speed)

@st.cache_resource
def load_and_train():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    
    tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train_scaled, y_train)

    return iris, scaler, X_test_scaled, y_test, log_reg, tree


iris, scaler, X_test, y_test, log_reg_model, tree_model = load_and_train()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🤖 Model Evaluation", "🔮 Try a Prediction"])



# TAB 1 - Dataset Overview

with tab1:
    st.subheader("Dataset Overview")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(15))
    with col2:
        st.write("**Samples per species**")
        st.bar_chart(df["species"].value_counts())

    st.write("**Statistical Summary**")
    st.dataframe(df.describe())


# TAB 2 - Model Evaluation

with tab2:
    model_choice = st.radio(
        "Choose a model to evaluate:",
        ["Logistic Regression", "Decision Tree"],
        horizontal=True
    )
    model = log_reg_model if model_choice == "Logistic Regression" else tree_model

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Precision", f"{precision:.1%}")
    col3.metric("Recall", f"{recall:.1%}")
    col4.metric("F1-Score", f"{f1:.1%}")

    fig, ax = plt.subplots(figsize=(4.0,3.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax,
        annot_kws={"size": 9}, cbar=False
    )
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(f"Confusion Matrix - {model_choice}", fontsize=9)
    ax.tick_params(labelsize=7)

    # Center it in a smaller column so it doesn't stretch full width
    col_left, col_mid, col_right = st.columns([1, 1, 1])
    with col_mid:
        st.pyplot(fig, use_container_width=True)

    st.write("**Sample Predictions**")
    sample_df = pd.DataFrame({
        "Actual": [iris.target_names[i] for i in y_test[:15]],
        "Predicted": [iris.target_names[i] for i in y_pred[:15]],
    })
    sample_df["Correct"] = sample_df["Actual"] == sample_df["Predicted"]
    st.dataframe(sample_df)


# TAB 3 - Live Prediction

with tab3:
    st.subheader("Enter flower measurements")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    with col2:
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 3.8)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2)

    model_choice_2 = st.radio(
        "Model to use for prediction:",
        ["Logistic Regression", "Decision Tree"],
        horizontal=True,
        key="prediction_model"
    )
    prediction_model = log_reg_model if model_choice_2 == "Logistic Regression" else tree_model

    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_class = prediction_model.predict(input_data)[0]
    probabilities = prediction_model.predict_proba(input_data)[0]

    st.success(f"Predicted species: **{iris.target_names[predicted_class]}**")

    proba_df = pd.DataFrame({
        "species": iris.target_names,
        "probability": probabilities
    })
    st.bar_chart(proba_df.set_index("species"))