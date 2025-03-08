import streamlit as st
import requests
from PIL import Image
import io

# Title and project description
st.title("Fashion MNIST Image Classifier")
st.markdown("""
### About This Project
**Author:** Aditya Khamitkar  
**Assessment for:** Soul AI by Deccan AI

This project was developed as an assessment to showcase the process of building and deploying an image classification model. The goal is to classify clothing items from the Fashion MNIST dataset into 10 categories such as T-shirt, Trouser, Pullover, Dress, and more.

**How It Works:**  
1. **Image Upload:** You upload an image (PNG, JPG, or JPEG) via the file uploader.  
2. **Preprocessing:** The image is converted to RGB, resized to 96x96 pixels, and normalized.  
3. **Prediction:** The preprocessed image is sent to a FastAPI endpoint that hosts a trained CNN model.  
4. **Result Display:** The predicted class and the confidence score are returned and displayed.

**Fun Facts:**  
- The Fashion MNIST dataset is a modern replacement for the classic MNIST dataset, offering 70,000 grayscale images of clothing items.  
- It covers 10 different classes, making it an excellent benchmark for basic image classification tasks.  
- This project demonstrates key concepts in computer vision and deep learning, such as data preprocessing, model training, and API deployment.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Authentication token (should match the one in auth.py)
token = "mysecuretoken"
headers = {"Authorization": f"Bearer {token}"}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Image"):
        with st.spinner("Predicting..."):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            response = requests.post("http://127.0.0.1:8000/predict", files={"file": img_bytes.getvalue()}, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['class']} ({result['confidence']*100:.2f}%)")
            else:
                st.error("Error in prediction")
