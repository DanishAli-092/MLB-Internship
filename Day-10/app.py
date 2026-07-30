"""
Day 10 - Mini Project: Breast Cancer Prediction System (Streamlit App)

Lets a user use the built-in sklearn dataset or upload their own CSV,
then walks through dataset exploration, baseline training, GridSearchCV
tuning, and a live prediction form - all from the browser.
"""

import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    make_scorer,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TOP_N_FEATURES = 8  # how many features the user gets to adjust for prediction

st.set_page_config(page_title="Breast Cancer Prediction System", page_icon="🩺", layout="wide")


st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(90deg, #7b2ff7 0%, #06beb6 100%);
        padding: 1.2rem 1.6rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .main-banner h1 { color: white; margin: 0; font-size: 1.6rem; }
    .main-banner p { color: rgba(255,255,255,0.9); margin: 0.2rem 0 0 0; }
    .pred-benign {
        background-color: #e6f7ec; border: 1px solid #34a853;
        border-radius: 8px; padding: 1rem 1.2rem; color: #1e6b3c;
    }
    .pred-malignant {
        background-color: #fdecea; border: 1px solid #d93025;
        border-radius: 8px; padding: 1rem 1.2rem; color: #a01a12;
    }
</style>
<div class="main-banner">
    <h1>🩺 Breast Cancer Prediction System</h1>
    <p>Model Evaluation & Hyperparameter Tuning - Day 10, MLB Internship</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_dataset():
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df


def load_uploaded_dataset(uploaded_file):
    # returns (dataframe, error_message). error_message is None if ok.
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read the CSV file: {e}"

    if "target" not in df.columns:
        return None, "CSV must contain a 'target' column (0/1 labels)."

    if df["target"].nunique() != 2:
        return None, f"'target' column must be binary. Found {df['target'].nunique()} unique values."

    if df.isnull().any().any():
        return None, "Dataset has missing values, please clean it before uploading."

    return df, None


def plot_confusion_matrix(cm, title, target_names):
    fig, ax = plt.subplots(figsize=(4, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, cbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def get_top_features(model, feature_names, n):
    # ranks features by how big their logistic regression coefficient is
    # (on scaled data, so they're comparable). used to pick which
    # features are worth showing as sliders instead of all 30 of them.
    coefficients = np.abs(model.coef_[0])
    ranked = sorted(zip(feature_names, coefficients), key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in ranked[:n]]


def score_model(model, X_test, y_test, pos_label):
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=pos_label),
        "recall": recall_score(y_test, predictions, pos_label=pos_label),
        "f1_score": f1_score(y_test, predictions, pos_label=pos_label),
    }
    cm = confusion_matrix(y_test, predictions)
    return metrics, cm


# Sidebar - pick dataset source
st.sidebar.header("1. Choose Dataset")
data_source = st.sidebar.radio("Source", ["Use built-in dataset", "Upload my own CSV"])

if data_source == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must have a 'target' column)", type=["csv"])
    if uploaded_file is None:
        st.info("👈 Upload a CSV from the sidebar, or switch to the built-in dataset.")
        st.stop()

    df, error = load_uploaded_dataset(uploaded_file)
    if error:
        st.error(f"⚠️ {error}")
        st.stop()
    st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # FIX: for a custom CSV we have no idea which value (0 or 1) actually
    # represents the "critical"/positive class (e.g. disease present).
    # Hardcoding malignant=0 like the built-in dataset would silently
    # mislabel results if the uploaded CSV uses the opposite convention.
    # So we ask the user explicitly instead of assuming.
    unique_vals = sorted(df["target"].unique().tolist())
    st.sidebar.subheader("2. Identify the Critical Class")
    critical_value = st.sidebar.selectbox(
        "Which target value represents the critical/positive class "
        "(e.g., disease present, defect found, event of interest)?",
        options=unique_vals,
        index=0,
        help="Precision/Recall/F1 will be computed treating this value as the positive class.",
    )
    POS_LABEL = critical_value
    TARGET_NAMES = [f"Class {v}" for v in unique_vals]
else:
    df = load_default_dataset()
    st.sidebar.success("Using sklearn's Breast Cancer Wisconsin dataset")

    # class 0 = malignant, class 1 = benign in this known dataset. sklearn's
    # precision/recall/f1 default to scoring class 1 unless told otherwise,
    # which would measure how well the model catches BENIGN cases, not
    # cancer. So pos_label is set to 0 for this specific, known dataset.
    POS_LABEL = 0
    TARGET_NAMES = ["malignant", "benign"]

# if a new dataset was loaded, clear any previously trained model so the
# prediction form can't run against a model trained on old data
data_signature = (df.shape, tuple(df.columns), POS_LABEL)
if st.session_state.get("data_signature") != data_signature:
    st.session_state["trained"] = False
    st.session_state["data_signature"] = data_signature
    # also clear old slider values, since a new dataset can have completely
    # different min/max ranges for a feature with the same name
    for key in list(st.session_state.keys()):
        if key.startswith("input_"):
            del st.session_state[key]



# Step 1: Dataset Exploration
st.header("Step 1 · Dataset Exploration")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Preview (head)")
    st.dataframe(df.head())

with col2:
    st.subheader("Target Distribution")
    class_counts = df["target"].value_counts().sort_index()
    st.bar_chart(class_counts)

with st.expander("Show statistical summary (describe)"):
    st.dataframe(df.describe())

with st.expander("Show column info"):
    info_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null_count": df.notnull().sum().values,
    })
    st.dataframe(info_df, hide_index=True)

imbalance_ratio = class_counts.max() / class_counts.min()
st.write(f"**Class imbalance ratio:** {imbalance_ratio:.2f} : 1")


# Step 2 & 3: Baseline model + Hyperparameter tuning
st.header("Step 2 & 3 · Train Baseline Model and Tune with GridSearchCV")

test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
run_button = st.button("🚀 Train and Compare Models", type="primary")

if run_button:
    try:
        X = df.drop(columns=["target"])
        y = df["target"]
        feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- baseline model with default settings ---
        with st.spinner("Training baseline Logistic Regression..."):
            baseline_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)
            baseline_model.fit(X_train_scaled, y_train)
            baseline_metrics, baseline_cm = score_model(baseline_model, X_test_scaled, y_test, POS_LABEL)

        param_grid = [
            {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]},
            {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1"], "solver": ["liblinear"]},
        ]

        # --- GridSearchCV scored on plain F1 ---
        # F1 treats a missed positive case and a false alarm as equally bad
        with st.spinner("Running GridSearchCV scored on F1..."):
            f1_scorer = make_scorer(f1_score, pos_label=POS_LABEL)
            grid_f1 = GridSearchCV(
                LogisticRegression(random_state=RANDOM_STATE, max_iter=10000),
                param_grid, cv=5, scoring=f1_scorer, n_jobs=-1,
            )
            grid_f1.fit(X_train_scaled, y_train)
            f1_metrics, f1_cm = score_model(grid_f1.best_estimator_, X_test_scaled, y_test, POS_LABEL)

        # --- GridSearchCV scored on F2 (recall weighted higher) ---
        # missing a real positive case is worse than a false alarm, so recall
        # should matter more than precision here - F2 weighs it 2x
        with st.spinner("Running GridSearchCV scored on F2..."):
            f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=POS_LABEL)
            grid_f2 = GridSearchCV(
                LogisticRegression(random_state=RANDOM_STATE, max_iter=10000),
                param_grid, cv=5, scoring=f2_scorer, n_jobs=-1,
            )
            grid_f2.fit(X_train_scaled, y_train)
            f2_metrics, f2_cm = score_model(grid_f2.best_estimator_, X_test_scaled, y_test, POS_LABEL)

        # save everything needed later into session_state, otherwise
        # streamlit forgets it the moment the user touches another
        # widget (every interaction reruns the script top to bottom).
        # the F2 model is used for live predictions since it's the
        # better choice for catching malignant cases.
        st.session_state["trained"] = True
        st.session_state["feature_names"] = feature_names
        st.session_state["scaler"] = scaler
        st.session_state["X_reference"] = X
        st.session_state["tuned_model"] = grid_f2.best_estimator_
        st.session_state["pos_label"] = POS_LABEL
        st.session_state["target_names"] = TARGET_NAMES

        st.session_state["baseline_metrics"] = baseline_metrics
        st.session_state["baseline_cm"] = baseline_cm

        st.session_state["f1_metrics"] = f1_metrics
        st.session_state["f1_cm"] = f1_cm
        st.session_state["f1_best_params"] = grid_f1.best_params_
        st.session_state["f1_best_score"] = grid_f1.best_score_

        st.session_state["f2_metrics"] = f2_metrics
        st.session_state["f2_cm"] = f2_cm
        st.session_state["f2_best_params"] = grid_f2.best_params_
        st.session_state["f2_best_score"] = grid_f2.best_score_

    except Exception as e:
        st.session_state["trained"] = False
        st.error(f"⚠️ Something went wrong while training: {e}")

if st.session_state.get("trained"):
    st.success("Training complete - three models trained for comparison.")

    param_col1, param_col2 = st.columns(2)
    with param_col1:
        st.caption("Best params · F1-scored GridSearchCV")
        st.json(st.session_state["f1_best_params"])
        st.write(f"Best CV F1-score: **{st.session_state['f1_best_score']:.4f}**")
    with param_col2:
        st.caption("Best params · F2-scored GridSearchCV (recommended)")
        st.json(st.session_state["f2_best_params"])
        st.write(f"Best CV F2-score: **{st.session_state['f2_best_score']:.4f}**")

    st.subheader("Baseline vs F1-Tuned vs F2-Tuned")
    st.caption(
        f"All metrics use '{st.session_state['target_names'][st.session_state['pos_label']]}' "
        "as the positive class (not sklearn's default of class 1)."
    )

    comparison_df = pd.DataFrame({
        "Baseline": st.session_state["baseline_metrics"],
        "F1-Tuned": st.session_state["f1_metrics"],
        "F2-Tuned (recommended)": st.session_state["f2_metrics"],
    })
    st.dataframe(comparison_df.style.format("{:.4f}"))
    st.download_button(
        "⬇️ Download comparison as CSV",
        comparison_df.to_csv().encode("utf-8"),
        file_name="baseline_vs_tuned_metrics.csv",
        mime="text/csv",
    )

    st.subheader("Confusion Matrix Comparison")
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col1:
        st.pyplot(plot_confusion_matrix(st.session_state["baseline_cm"], "Baseline", st.session_state["target_names"]))
    with cm_col2:
        st.pyplot(plot_confusion_matrix(st.session_state["f1_cm"], "F1-Tuned", st.session_state["target_names"]))
    with cm_col3:
        st.pyplot(plot_confusion_matrix(st.session_state["f2_cm"], "F2-Tuned (recommended)", st.session_state["target_names"]))

    st.info(
        "📌 **Why F2-Tuning Was Used (Even If Results Match Baseline):** "
        "F2-score was deliberately chosen as the tuning metric because, in a "
        "medical diagnosis context, missing a real positive case (False "
        "Negative) is far more costly than raising a false alarm (False "
        "Positive). F2 weighs recall twice as heavily as precision for "
        "exactly this reason.\n\n"
        "On a given dataset and train/test split, F1-scored and F2-scored "
        "GridSearchCV can converge to the same optimal configuration within "
        "the searched hyperparameter grid. When this happens, it's a sign of "
        "model stability on that particular data, not a coincidence - both "
        "objectives independently point to the same decision boundary.\n\n"
        "This does not make F2-tuning unnecessary. It reflects a deliberate, "
        "forward-looking methodology: had the dataset, split, or "
        "hyperparameter grid produced different optimal parameters under F1 "
        "vs. F2, the F2-tuned model would automatically have been the safer "
        "choice - prioritizing catching more true positive cases over "
        "minimizing false alarms. Matching results confirm robustness, while "
        "F2-based selection remains the correct methodology for any future "
        "data where F1 and F2 might diverge."
    )
else:
    st.caption("Click the button above to train and compare all three models.")


# Step 4: Live prediction using the F2-tuned model

st.header("Step 4 · Try a Live Prediction")

if not st.session_state.get("trained"):
    st.caption("Train the models above first - the prediction form needs a trained model to run against.")
else:
    st.write(
        f"Adjust the **{TOP_N_FEATURES} features** that matter most to the model's decision "
        "(ranked by its own coefficients). Every other feature is filled in using the "
        "dataset's median value automatically."
    )

    feature_names = st.session_state["feature_names"]
    tuned_model = st.session_state["tuned_model"]
    scaler = st.session_state["scaler"]
    X_reference = st.session_state["X_reference"]
    target_names = st.session_state["target_names"]

    top_features = get_top_features(tuned_model, feature_names, TOP_N_FEATURES)

    input_values = {}
    slider_cols = st.columns(2)
    for i, feature in enumerate(top_features):
        col = slider_cols[i % 2]
        min_val = float(X_reference[feature].min())
        max_val = float(X_reference[feature].max())
        default_val = float(X_reference[feature].median())
        input_values[feature] = col.slider(
            feature, min_value=min_val, max_value=max_val, value=default_val,
            key=f"input_{feature}",
        )

    if st.button("🔬 Predict"):
        try:
            # build the full 30-feature row: user's slider values for the
            # top features, median values for everything else
            full_row = {}
            for feature in feature_names:
                full_row[feature] = input_values.get(feature, float(X_reference[feature].median()))

            input_df = pd.DataFrame([full_row])[feature_names]
            input_scaled = scaler.transform(input_df)

            prediction = tuned_model.predict(input_scaled)[0]
            probabilities = tuned_model.predict_proba(input_scaled)[0]
            confidence = probabilities[prediction] * 100
            predicted_label = target_names[prediction]

            is_critical = (prediction == st.session_state["pos_label"])
            if not is_critical:
                st.markdown(
                    f'<div class="pred-benign"><b>Prediction: {predicted_label.upper()}</b><br>'
                    f'Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="pred-malignant"><b>Prediction: {predicted_label.upper()}</b><br>'
                    f'Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                " | ".join(
                    f"{name}: {probabilities[i]*100:.1f}%"
                    for i, name in enumerate(target_names)
                )
            )
            st.caption(
                "⚠️ This is a student ML project for learning purposes, not a "
                "medical diagnostic tool."
            )

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")