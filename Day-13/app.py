import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from fashion_mnist_classifier import (
    CLASS_NAMES,
    MODEL_PATH,
    load_fashion_mnist,
    preprocess,
    build_model
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="CNN Vision Dashboard", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

GRAPHS_DIR = "Graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

def save_and_show(fig, name):
    fig.savefig(os.path.join(GRAPHS_DIR, f"{name}.png"), bbox_inches="tight", dpi=150)
    st.pyplot(fig)
    plt.close(fig)


st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(135deg, #1e0b36, #6b4c9a, #4a90e2);
        padding: 30px 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    .main-title {
        font-size: 75px !important;  
        font-weight: 900 !important;
        text-align: center;
        color: #ffffff;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #e0e0e0;
        font-size: 26px !important;  
        font-weight: 300;
        margin-top: 0px;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #6b4c9a, #4a90e2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CACHED FUNCTIONS ---
@st.cache_data
def get_data():
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    x_train_n, x_test_n = preprocess(x_train, x_test)
    return x_train, y_train, x_test, y_test, x_train_n, x_test_n

@st.cache_resource
def get_model_if_exists():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

x_train, y_train, x_test, y_test, x_train_n, x_test_n = get_data()

if "model" not in st.session_state:
    st.session_state.model = get_model_if_exists()
if "history" not in st.session_state:
    st.session_state.history = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Data Overview", 
    "⚙️ Model Training Engine", 
    "📊 Performance Metrics", 
    "🎯 Live Vision (Upload)"
])

st.sidebar.markdown("---")
st.sidebar.info("**MLB Internship - Day 13**\n\nBuilt with TensorFlow & Streamlit.")

# --- MAIN HEADER ---
st.markdown(
    """
    <div class="header-box">
        <p class="main-title">👁️ CNN Vision Dashboard</p>
        <p class="sub-title">Advanced Image Classification using Fashion MNIST</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====-------============ PAGE 1: DATA OVERVIEW =================
if page == "🏠 Data Overview":
    st.subheader("Data Engine Status")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Set Size", f"{x_train.shape[0]:,}")
    col2.metric("Testing Set Size", f"{x_test.shape[0]:,}")
    col3.metric("Total Categories", len(CLASS_NAMES))

    st.markdown("---")
    st.subheader("Dataset Glimpse")
    
    n_samples = st.slider("Select grid size", min_value=5, max_value=20, value=10, step=5)
    cols = st.columns(5)
    for i in range(n_samples):
        with cols[i % 5]:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(x_train[i], cmap="magma") 
            ax.set_title(CLASS_NAMES[y_train[i]], fontsize=10, fontweight='bold')
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)


# ================= PAGE 2: MODEL TRAINING ENGINE =================
elif page == "⚙️ Model Training Engine":
    st.subheader("Train Neural Network")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        epochs = st.number_input("Set Epochs", min_value=1, max_value=20, value=10)
        st.write("**Architecture:** 3x Conv2D + BatchNorm + Dense")
        
        status = "🟢 Active" if st.session_state.model is not None else "🔴 Offline"
        st.metric("Engine Status", status)
        
        train_button = st.button("Initialize Training", use_container_width=True, type="primary")
        
        # : A Rollback Button
        st.write("")
        if st.button("🔄 Restore Production Model", use_container_width=True):
            if os.path.exists(MODEL_PATH):
                st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
                st.success("✅ Original Production Model (91.14%) Restored Successfully!")
            else:
                st.error("Production model file not found.")

    with col2:
        with st.expander("Deep Dive: Network Architecture", expanded=True):
            summary_model = build_model()
            summary_lines = []
            summary_model.summary(print_fn=lambda line: summary_lines.append(line))
            st.code("\n".join(summary_lines), language="text")

    if train_button:
        with st.status("Training Neural Network...", expanded=True) as status_box:
            st.write("Initializing compilation...")
            model = build_model()
            
            st.write(f"Executing {epochs} Epochs. Please wait...")
            early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            history = model.fit(
                x_train_n, y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=128,
                callbacks=[early_stop],
                verbose=1 
            )
            
            #  update the session model for live testing, 
            # but DO NOT blindly overwrite the original MODEL_PATH here anymore!
            st.session_state.model = model
            st.session_state.history = history.history
            
            status_box.update(label="Training Complete!", state="complete", expanded=False)

        final_acc = history.history["val_accuracy"][-1] * 100
        st.success(f"🎉 Experimental Network trained! Final Validation Accuracy: **{final_acc:.2f}%**")
        st.info("💡 You are now using the Experimental Model. Click 'Restore Production Model' to switch back to your best model.")

# ================= PAGE 3: PERFORMANCE METRICS =================
elif page == "📊 Performance Metrics":
    if st.session_state.model is None:
        st.warning("⚠️ Model is offline. Please train the model first.")
    else:
        model = st.session_state.model
        
        with st.spinner("Calculating metrics on 10,000 test images..."):
            loss, accuracy = model.evaluate(x_test_n, y_test, verbose=0)
            probs = model.predict(x_test_n, verbose=0)
            predicted_labels = np.argmax(probs, axis=1)
            cm = confusion_matrix(y_test, predicted_labels)

        col1, col2, col3 = st.columns(3)
        col1.metric("Production Accuracy", f"{accuracy * 100:.2f}%")
        col2.metric("System Loss", f"{loss:.4f}")
        col3.metric("Parameters Optimized", "242,442")

        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
            disp.plot(ax=ax, cmap="Purples", xticks_rotation="vertical", colorbar=False)
            save_and_show(fig, "confusion_matrix")
            
        with col_chart2:
            st.subheader("Class-Wise Accuracy")
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.barh(CLASS_NAMES, per_class_acc, color="#6b4c9a")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Accuracy Rate")
            save_and_show(fig, "per_class_accuracy")
            


        st.markdown("---")
        st.subheader("📂 Offline Training Reports")
        st.write("Detailed visual reports generated during the model's main training phase:")

        project_dir = os.path.join("outputs", "project")

        # 1. Show Training Curves
        curves_path = os.path.join(project_dir, "02_training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, caption="Accuracy & Loss Curves Over Epochs", use_container_width=True)

        st.write("")

        # 2. Show Correct vs Incorrect Predictions Vertically
        st.subheader("Model Validation Samples")

        correct_path = os.path.join(project_dir, "05_correctly_classified.png")
        if os.path.exists(correct_path):
            st.image(correct_path, caption="🟢 10 Correctly Classified Images", use_container_width=True)
            st.write("") 

        incorrect_path = os.path.join(project_dir, "06_incorrectly_classified.png")
        if os.path.exists(incorrect_path):
            st.image(incorrect_path, caption="🔴 10 Incorrectly Classified Images (Where the model got confused)", use_container_width=True)

        st.markdown("---")
        
        # 3.Show Classification Report Text
        report_path = os.path.join(project_dir, "classification_report.txt")
        if os.path.exists(report_path):
            with st.expander("📄 View Raw Classification Report (Precision, Recall, F1-Score)"):
                with open(report_path, "r") as f:
                    st.code(f.read(), language="text")            

# ================= PAGE 4: LIVE VISION =================
elif page == "🎯 Live Vision (Upload)":
    st.subheader("Real-Time Image Classification")
    
    if st.session_state.model is None:
        st.error("⚠️ AI Model is not loaded. Go to the Training Engine to build it.")
    else:
        st.info("💡 **Pro-Tip:** If you upload a regular photo (light background), keep the 'Invert Colors' switch ON.")
        invert = st.toggle("Invert Colors", value=True)
        
        uploaded_file = st.file_uploader("Drop an image here", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Image Processing
            image = Image.open(uploaded_file).convert("L")
            img_resized = image.resize((28, 28))
            
            if invert:
                img_resized = ImageOps.invert(img_resized)
                
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_input = img_array.reshape(1, 28, 28, 1)

            # AI Prediction
            model = st.session_state.model
            prediction = model.predict(img_input)[0]
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.markdown("---")
            col_img, col_res = st.columns([1, 2])
            
            with col_img:
                st.write("**Processed Input (What AI sees):**")
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img_array, cmap="gray")
                ax.axis("off")
                st.pyplot(fig)
                
            with col_res:
                st.success(f"### Result: **{predicted_class}**")
                st.write(f"Confidence Level: **{confidence:.2f}%**")
                
                st.write("**Network Softmax Outputs:**")
                
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = float(prediction[i])
                    col_text, col_bar = st.columns([1, 3])
                    with col_text:
                        if class_name == predicted_class:
                            st.write(f"**{class_name}**")
                        else:
                            st.write(class_name)
                    with col_bar:
                        st.progress(prob)