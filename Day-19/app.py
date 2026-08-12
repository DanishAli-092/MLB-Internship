
# Day 19 - ContourVision Streamlit App Lets user upload an image and see contour detection + shape detection results side by side, with area/perimeter drawn on the images and also listed in a table below.

import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

MIN_CONTOUR_AREA = 100

# 1. Blur the color image directly skip grey scale
def preprocess_image(image):
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Split into Blue, Green, and Red channels
    b, g, r = cv2.split(blurred)
    
    # 3. apply canny edge detection to each channel
    
    edges_b = cv2.Canny(b, 50, 150)
    edges_g = cv2.Canny(g, 50, 150)
    edges_r = cv2.Canny(r, 50, 150)
    
    # 4. Merge all edges no color hide in bg
    combined_edges = cv2.bitwise_or(edges_b, edges_g)
    combined_edges = cv2.bitwise_or(combined_edges, edges_r)
    
    # 5. Dilate to thicken thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    
    return thick_edges


def detect_shape(contour, area, perimeter):
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = len(approx)

    if corners == 3:
        shape_name = "Triangle"

    elif corners == 4:
        
        # need to check aspect ratio to tell square apart from rectangle
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.90 <= aspect_ratio <= 1.10:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"

    elif corners == 5:
        shape_name = "Pentagon"

    else:
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        shape_name = "Circle" if circularity > 0.8 else "Polygon"

    return shape_name

   
    # Runs the full pipeline and returns contour image, labeled shape image,and a list of detected shapes with their area/perimeter. Both output images have area/perimeter drawn directly on them.
    
def process_image(image):
    
    thresh = preprocess_image(image)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_output = image.copy()
    shape_output = image.copy()
    results = []

    for contour in contours:
        contour = cv2.convexHull(contour)
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        shape_name = detect_shape(contour, area, perimeter)
        metrics_label = f"A:{int(area)} P:{int(perimeter)}"

        x, y, w, h = cv2.boundingRect(contour)

        # contour output: outline + box + metrics 
        
        cv2.drawContours(contour_output, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(contour_output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(contour_output, metrics_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # shape output: outline + box + shape name +   metrics, both above the box for clerly visulize
        cv2.drawContours(shape_output, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(shape_output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(shape_output, shape_name, (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(shape_output, metrics_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        results.append({
            "Shape": shape_name,
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2)
        })

    return contour_output, shape_output, results


def convert_image_to_bytes(img_array):
    """Converts OpenCV BGR image array to bytes for downloading"""
    is_success, buffer = cv2.imencode(".png", img_array)
    if is_success:
        return buffer.tobytes()
    return None


def main():
    
    st.set_page_config(page_title="ContourVision", page_icon="📐", layout="wide")

    
    st.title("ContourVision 📐")
    st.write("Upload an image containing simple shapes (circle, square, "
             "rectangle, triangle, polygon) to detect and label them.")
    st.markdown("---") 

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Please upload an image to get started.")
        return

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        return

    
    with st.spinner("Processing image..."):
        contour_result, shape_result, shapes_data = process_image(image)

    st.markdown("### Image Processing Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### **1. Original Image**")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.markdown("#### **2. Contours Detected**")
        st.image(cv2.cvtColor(contour_result, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col3:
        st.markdown("#### **3. Labeled Shapes**")
        st.image(cv2.cvtColor(shape_result, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")

    col_data, col_dl = st.columns([2, 1])

    with col_data:
        st.markdown("#### Shape Details")
        if shapes_data:
            st.dataframe(shapes_data, use_container_width=True) 
        else:
            st.warning("No shapes detected in this image.")

    with col_dl:
        st.markdown("#### Export Outputs")
        st.write("Download the processed images here.")
        
        st.download_button(
            label="📥 Download Contours Image",
            data=convert_image_to_bytes(contour_result),
            file_name="contours_output.png",
            mime="image/png",
            use_container_width=True
        )
        
        st.download_button(
            label="📥 Download Labeled Shapes",
            data=convert_image_to_bytes(shape_result),
            file_name="shapes_output.png",
            mime="image/png",
            use_container_width=True
        )

if __name__ == "__main__":
    main()