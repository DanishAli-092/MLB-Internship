
#Day 25 - Feature Matching System

import streamlit as st
import cv2 as cv
import tempfile
import os

# Backend imports from  local files
from feature_detection import harris_corner_detection, orb_keypoint_detection
from feature_matching import orb_knn_matching, orb_bruteforce_matching


st.set_page_config(page_title="Vision Match", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
h1 { color: #0F172A; font-weight: 900; font-family: 'Segoe UI', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background-color: #F8FAFC; border-radius: 8px 8px 0 0; padding: 10px 20px; border: 1px solid #E2E8F0; border-bottom: none; }
.stTabs [aria-selected="true"] { background-color: #0EA5E9 !important; color: white !important; font-weight: bold; }
div[data-testid="stMetric"] { background: white; border-left: 5px solid #0EA5E9; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border-radius: 4px; }
.stDownloadButton>button { background-color: white; color: #0EA5E9; border: 1px solid #0EA5E9; border-radius: 8px; width: 100%; }
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file):
    """Saves Streamlit uploaded file to temp path for OpenCV to read."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name

def bgr_to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def show_image_with_download(img_bgr, caption, filename):
    """Helper to show image and provide a download button below it."""
    st.image(bgr_to_rgb(img_bgr), caption=caption, use_container_width=True)
    success, buffer = cv.imencode(".png", img_bgr)
    if success:
        st.download_button(
            label=f"💾 Download {caption}",
            data=buffer.tobytes(),
            file_name=filename,
            mime="image/png",
            key=filename
        )


with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("Upload your image pair here.")
    
    file1 = st.file_uploader("1️⃣ Base Image", type=["png", "jpg", "jpeg"])
    file2 = st.file_uploader("2️⃣ Target Image", type=["png", "jpg", "jpeg"])
    
    st.divider()
    matching_method = st.radio(
        "🧠 Matching Algorithm",
        ["KNN + Ratio Test", "Brute Force"]
    )
    
    run_analysis = st.button("🚀 Execute Vision Match", use_container_width=True)


st.title("Vision Match 🔍")
st.markdown("Advanced Feature Detection & Image Matching System")

if not (file1 and file2):
    st.info("👈 Please upload both images from the sidebar to begin analysis.")
else:
    # --- Immediate Preview of Uploaded Images ---
    st.subheader("🖼️ Uploaded Images Preview")
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.image(file1, caption="1️⃣ Base Image", use_container_width=True)
    with prev_col2:
        st.image(file2, caption="2️⃣ Target Image", use_container_width=True)
    
    st.divider() 

    if run_analysis:
        with st.spinner("Processing Computer Vision Pipeline... ⏳"):
            # Save files temporarily for backend scripts
            path1 = save_uploaded_file(file1)
            path2 = save_uploaded_file(file2)

    
            if "KNN" in matching_method:
                img_matches, kp1, kp2, des1, des2, good_matches = orb_knn_matching(path1, path2)
                matches_filename = "matches_knn.png"
            else:
                img_matches, kp1, kp2, des1, des2, good_matches = orb_bruteforce_matching(path1, path2)
                matches_filename = "matches_bruteforce.png"

            img_kp1, _, _ = orb_keypoint_detection(path1)
            img_kp2, _, _ = orb_keypoint_detection(path2)
            img_harris1, num_corners1 = harris_corner_detection(path1)
            img_harris2, num_corners2 = harris_corner_detection(path2)

            # Save to outputs folder locally
            cv.imwrite(os.path.join(OUTPUT_DIR, matches_filename), img_matches)
            cv.imwrite(os.path.join(OUTPUT_DIR, "keypoints_image1.png"), img_kp1)
            cv.imwrite(os.path.join(OUTPUT_DIR, "keypoints_image2.png"), img_kp2)
            cv.imwrite(os.path.join(OUTPUT_DIR, "harris_image1.png"), img_harris1)
            cv.imwrite(os.path.join(OUTPUT_DIR, "harris_image2.png"), img_harris2)

            # Clean up temp files
            os.unlink(path1)
            os.unlink(path2)

            
            st.subheader("📈 Real-time Statistics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Keypoints (Image 1)", len(kp1))
            m2.metric("Keypoints (Image 2)", len(kp2))
            m3.metric("Verified Good Matches", len(good_matches))
            st.divider()

           
            st.subheader("🔬 Visual Output")
            tab1, tab2, tab3 = st.tabs(["🔗 Feature Matching", "🎯 ORB Keypoints", "📐 Harris Corners"])

            with tab1:
                show_image_with_download(img_matches, "Matched Features", matches_filename)
                st.success(f"✅ All results saved locally to the '{OUTPUT_DIR}' folder.")

            with tab2:
                st.markdown("**ORB (Oriented FAST and Rotated BRIEF)** detects keypoints along with binary descriptors, which makes cross-image matching possible.")
                c1, c2 = st.columns(2)
                with c1:
                    show_image_with_download(img_kp1, "Image 1 - ORB Keypoints", "keypoints_image1.png")
                with c2:
                    show_image_with_download(img_kp2, "Image 2 - ORB Keypoints", "keypoints_image2.png")

            with tab3:
                st.markdown("**Harris Corner Detection** finds raw corner points based on intensity gradients (no descriptors, so it cannot be matched across images).")
                c3, c4 = st.columns(2)
                with c3:
                    show_image_with_download(img_harris1, f"Image 1 - Harris Corners ({int(num_corners1)})", "harris_image1.png")
                with c4:
                    show_image_with_download(img_harris2, f"Image 2 - Harris Corners ({int(num_corners2)})", "harris_image2.png")
                
                st.divider()
                st.markdown("### 📊 Harris vs ORB Comparison")
                comparison_table = {
                    "Aspect": ["Detects", "Descriptors", "Cross-Image Matching", "Rotation Invariant", "Scale Invariant"],
                    "Harris Corner": ["Corners", "No", "Not Supported", "No", "No"],
                    "ORB": ["Keypoints", "Yes (Binary)", "Supported", "Yes", "Yes (via pyramid)"]
                }
                st.table(comparison_table)