"""
Day 16 - Image Processing Toolkit (Streamlit Web App)
Upload an image and perform OpenCV operations interactively.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import date
import io


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Image Processing Toolkit",
    page_icon="🖼️",
    layout="wide",
)


# ---------------- Helper Functions ----------------

def load_image_from_upload(uploaded_file):
    """Convert a Streamlit uploaded file into an OpenCV BGR image."""
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_image

def to_display_format(image):
    """Convert an image (BGR or grayscale) into RGB for st.image display."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_downloadable_bytes(image):
    """Encode an OpenCV image into PNG bytes for download."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode image for download.")
    return io.BytesIO(buffer.tobytes())

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1 + (contrast / 100.0)
    beta = brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def commit_change(new_image):
    """Push the current image to history, then save the new image as current."""
    st.session_state.history.append(st.session_state.processed_image.copy())
    st.session_state.processed_image = new_image


# ---------------- Session State Setup----------------

if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- Sidebar: Upload & Reset ----------------

st.sidebar.markdown(
    """
    <div style="background-color: #1E1F29; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px; border: 1px solid #2D303E; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin-top: 0; margin-bottom: 5px; font-size: 24px; color: #FFFFFF;">
            🖼️ Vision<span style="color: #4CAF50;">Kit</span>
        </h2>
        <p style="color: #A0AEC0; font-size: 13px; margin-bottom: 12px; line-height: 1.4;">
            OpenCV Processing Studio<br>Day 16 Internship Project
        </p>
        <div style="background-color: #2D303E; padding: 5px 10px; border-radius: 6px; display: inline-block;">
            <p style="margin: 0; font-size: 12px; color: #E2E8F0;">
                Engineered by <b style="color: #4CAF50;">Danish Ali</b>
            </p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload an image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    if st.session_state.original_image is None or st.sidebar.button("🔄 Load / Replace Image", use_container_width=True):
        try:
            image = load_image_from_upload(uploaded_file)
            st.session_state.original_image = image
            st.session_state.processed_image = image.copy()
            st.session_state.history = []
            st.sidebar.success("✅ Image loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Failed to load image: {e}")

st.sidebar.markdown("---")
col_undo, col_reset = st.sidebar.columns(2)

with col_undo:
    if st.button("↩️ Undo", use_container_width=True):
        if st.session_state.history:
            st.session_state.processed_image = st.session_state.history.pop()
            st.rerun()
        else:
            st.sidebar.warning("Nothing to undo.")

with col_reset:
    if st.button("🗑️ Reset", use_container_width=True):
        if st.session_state.original_image is not None:
            st.session_state.processed_image = st.session_state.original_image.copy()
            st.session_state.history = []
            st.rerun()
        else:
            st.sidebar.warning("No image loaded yet.")


# ---------------- Main Area ----------------

if st.session_state.processed_image is None:
    st.title("📷 Image Processing Toolkit")
    st.info("👈 Please upload an image from the sidebar to get started.")
    st.stop()

st.title("📷 Image Processing Toolkit")
st.markdown("---")

st.sidebar.markdown("### 🛠️ Processing Tools")
operation = st.sidebar.selectbox(
    "Choose an operation:",
    [
        "None",
        "Grayscale",
        "Resize",
        "Rotate",
        "Flip",
        "Crop",
        "Draw Shape",
        "Add Text",
        "Brightness / Contrast",
        "Compare BGR vs RGB",
    ],
)

# used ONLY for preview/settings.
current_img = st.session_state.processed_image
h, w = current_img.shape[:2]

# Calculate Channels and Size dynamically
channels = current_img.shape[2] if len(current_img.shape) == 3 else 1
image_size_kb = current_img.nbytes / 1024

# ---------------- NEW FEATURE: IMAGE METADATA ----------------
with st.expander("📊 Current Image Metadata", expanded=True):
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Width", f"{w} px")
    met2.metric("Height", f"{h} px")
    met3.metric("Channels", f"{channels}")
    met4.metric("Memory Size", f"{image_size_kb:.1f} KB")

st.sidebar.markdown("---")

try:
    if operation == "Grayscale":
        st.sidebar.info("🎨 Converts the image to black & white.")
        if st.sidebar.button("✅ Apply Grayscale", type="primary", use_container_width=True):
            gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            commit_change(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            st.rerun()

    elif operation == "Resize":
        col1, col2 = st.sidebar.columns(2)
        new_w = col1.number_input("Width (px)", min_value=10, value=w)
        new_h = col2.number_input("Height (px)", min_value=10, value=h)
        if st.sidebar.button("✅ Apply Resize", type="primary", use_container_width=True):
            commit_change(cv2.resize(current_img, (int(new_w), int(new_h))))
            st.rerun()

    elif operation == "Rotate":
        angle = st.sidebar.selectbox("Select Angle", [90, 180, 270])
        if st.sidebar.button("✅ Apply Rotate", type="primary", use_container_width=True):
            rotation_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            commit_change(cv2.rotate(current_img, rotation_map[angle]))
            st.rerun()

    elif operation == "Flip":
        direction = st.sidebar.selectbox("Direction", ["Horizontal", "Vertical", "Both"])
        if st.sidebar.button("✅ Apply Flip", type="primary", use_container_width=True):
            flip_codes = {"Horizontal": 1, "Vertical": 0, "Both": -1}
            commit_change(cv2.flip(current_img, flip_codes[direction]))
            st.rerun()

    elif operation == "Crop":
        if w <= 1 or h <= 1:
            st.sidebar.error("Image too small to crop further. Reset first.")
        else:
            col1, col2 = st.sidebar.columns(2)
            x1 = col1.slider("x1 (Left)", 0, w - 1, 0)
            x2 = col2.slider("x2 (Right)", 1, w, w)
            y1 = col1.slider("y1 (Top)", 0, h - 1, 0)
            y2 = col2.slider("y2 (Bottom)", 1, h, h)
            if st.sidebar.button("✅ Apply Crop", type="primary", use_container_width=True):
                if x1 < x2 and y1 < y2:
                    commit_change(current_img[y1:y2, x1:x2])
                    st.rerun()
                else:
                    st.sidebar.error("Invalid crop range: x1 < x2 and y1 < y2 required.")

    elif operation == "Draw Shape":
        shape = st.sidebar.selectbox("Shape Type", ["Rectangle", "Circle", "Line", "Polygon"])
        
        col1, col2 = st.sidebar.columns(2)
        color = col1.color_picker("Color", "#00FF00")
        thickness = col2.slider("Thickness", 1, 10, 2)
        
        hex_color = color.lstrip("#")
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        bgr_color = (rgb[2], rgb[1], rgb[0])

        if shape == "Rectangle":
            c1, c2 = st.sidebar.columns(2)
            x1 = c1.number_input("x1", 0, w, 50)
            y1 = c2.number_input("y1", 0, h, 50)
            x2 = c1.number_input("x2", 0, w, max(51, w - 50))
            y2 = c2.number_input("y2", 0, h, max(51, h - 50))
            if st.sidebar.button("✅ Apply Rectangle", type="primary", use_container_width=True):
                preview = current_img.copy()
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), bgr_color, thickness)
                commit_change(preview)
                st.rerun()

        elif shape == "Circle":
            c1, c2 = st.sidebar.columns(2)
            cx = c1.number_input("Center X", 0, w, w // 2)
            cy = c2.number_input("Center Y", 0, h, h // 2)
            radius = st.sidebar.number_input("Radius", 1, max(1, min(h, w)), max(1, min(h, w) // 10))
            if st.sidebar.button("✅ Apply Circle", type="primary", use_container_width=True):
                preview = current_img.copy()
                cv2.circle(preview, (int(cx), int(cy)), int(radius), bgr_color, thickness)
                commit_change(preview)
                st.rerun()

        elif shape == "Line":
            c1, c2 = st.sidebar.columns(2)
            x1 = c1.number_input("x1", 0, w, 0)
            y1 = c2.number_input("y1", 0, h, 0)
            x2 = c1.number_input("x2", 0, w, w)
            y2 = c2.number_input("y2", 0, h, h)
            if st.sidebar.button("✅ Apply Line", type="primary", use_container_width=True):
                preview = current_img.copy()
                cv2.line(preview, (int(x1), int(y1)), (int(x2), int(y2)), bgr_color, thickness)
                commit_change(preview)
                st.rerun()

        elif shape == "Polygon":
            st.sidebar.caption("Enter 3 or more points to form the polygon.")
            num_points = st.sidebar.number_input("Number of points", min_value=3, max_value=10, value=4)
            points = []
            for i in range(int(num_points)):
                pc1, pc2 = st.sidebar.columns(2)
                px = pc1.number_input(f"Point {i+1} X", 0, w, min(w - 1, 50 + i * 40), key=f"poly_x_{i}")
                py = pc2.number_input(f"Point {i+1} Y", 0, h, min(h - 1, 50 + i * 30), key=f"poly_y_{i}")
                points.append((int(px), int(py)))
            if st.sidebar.button("✅ Apply Polygon", type="primary", use_container_width=True):
                preview = current_img.copy()
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [pts], isClosed=True, color=bgr_color, thickness=thickness)
                commit_change(preview)
                st.rerun()

    elif operation == "Add Text":
        text = st.sidebar.text_input("Text", f"Danish Ali | {date.today()}")
        c1, c2 = st.sidebar.columns(2)
        x = c1.number_input("X position", 0, w, 20)
        y = c2.number_input("Y position", 0, h, max(20, h - 20))
        
        c3, c4 = st.sidebar.columns(2)
        font_scale = c3.slider("Font Scale", 0.3, 5.0, max(1.0, w / 800))
        color = c4.color_picker("Text Color", "#FF0000")
        
        hex_color = color.lstrip("#")
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        bgr_color = (rgb[2], rgb[1], rgb[0])

        if st.sidebar.button("✅ Apply Text", type="primary", use_container_width=True):
            preview = current_img.copy()
            cv2.putText(preview, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, bgr_color, 2)
            commit_change(preview)
            st.rerun()

    elif operation == "Brightness / Contrast":
        brightness = st.sidebar.slider("Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Contrast", -100, 100, 0)
        if st.sidebar.button("✅ Apply Brightness/Contrast", type="primary", use_container_width=True):
            commit_change(adjust_brightness_contrast(current_img, brightness, contrast))
            st.rerun()

    elif operation == "Compare BGR vs RGB":
        col1, col2 = st.columns(2)
        
        # 1. Correct Image (Converted to RGB)
        correct_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
        col1.image(correct_rgb, caption="1. Correct (Converted to RGB)", use_container_width=True)
        
        # 2. The Buggy Image (Raw BGR passed directly)
        col2.image(current_img, caption="2. Raw OpenCV (BGR) - The Smurf Effect!", use_container_width=True)
        
        st.info(
            "🚨 **The Smurf Effect:** In the second image, the Red and Blue colors have been swapped. "
            "This happens because OpenCV reads the image in BGR format, whereas Streamlit renders it "
            "assuming it is in RGB format."
        )

except Exception as e:
    st.error(f"Operation failed: {e}")


# ---------------- Display: Original vs Processed ----------------

if operation != "Compare BGR vs RGB":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original")
        st.image(to_display_format(st.session_state.original_image), use_container_width=True)

    with col2:
        st.subheader("✨ Processed")
        st.image(to_display_format(st.session_state.processed_image), use_container_width=True)

    st.markdown("---")
    try:
        download_bytes = convert_to_downloadable_bytes(st.session_state.processed_image)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.download_button(
                label="⬇️ Download Processed Image",
                data=download_bytes,
                file_name="processed_image.png",
                mime="image/png",
                type="primary",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Could not prepare download: {e}")


# ---------------- Footer ----------------
st.sidebar.markdown("---")
st.sidebar.caption("Day 16 - ML Internship | OpenCV Fundamentals & Basic Image Processing   Developed By Danish")






