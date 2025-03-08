# frontend.py
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# API endpoint configuration (adjust URL if deployed)
API_URL = "http://localhost:8000/predict"
GRADCAM_URL = "http://localhost:8000/gradcam"

# Basic auth credentials (should match those in auth.py)
AUTH = ("admin", "password123")

st.title("Image Classification Demo")
st.write("Upload an image to classify it using our EfficientNet-based model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Option to perform classification
    if st.button("Classify Image"):
        # Prepare file for API request
        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post(API_URL, files={"file": uploaded_file}, auth=AUTH)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Class: {result['predicted_class']} (Confidence: {result['confidence']:.2f})")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.write("### Generate Grad-CAM Visualization")
    if st.button("Generate Grad-CAM"):
        try:
            response = requests.post(GRADCAM_URL, files={"file": uploaded_file}, auth=AUTH)
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                # Display Grad-CAM image if available
                gradcam_image = Image.open("gradcam_output.jpg")
                st.image(gradcam_image, caption="Grad-CAM Output", use_column_width=True)
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
