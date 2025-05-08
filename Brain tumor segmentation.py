import cv2
import numpy as np
import streamlit as st
from skimage import measure

# Streamlit app setup
st.title("Brain Tumor Segmentation")
st.write("This application segments brain tumors from MRI images using OpenCV.")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to segment the tumor
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    segmented_image = image.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)

    # Display the segmented image
    st.image(segmented_image, caption="Segmented Image", use_column_width=True)

    # Calculate tumor area
    tumor_area = 0
    for contour in contours:
        tumor_area += cv2.contourArea(contour)
    st.write(f"Tumor Area: {tumor_area:.2f} pixels")

    # Display labeled regions
    labels = measure.label(thresh, connectivity=2)
    st.write(f"Number of Regions Detected: {labels.max()}")
else:
    st.write("Please upload an MRI image to proceed.")