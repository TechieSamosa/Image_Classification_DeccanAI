# app.py
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from app.auth import get_current_username
from src.explainability import get_gradcam_heatmap, save_and_display_gradcam
from src.data_preprocessing import load_and_preprocess_data
import logging

# Set up logger
logger = logging.getLogger("ImageClassificationAPI")
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API for image classification using EfficientNet and Grad-CAM explainability.",
    version="1.0"
)

# Allow CORS for local testing (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (ensure model file exists)
try:
    model = tf.keras.models.load_model("model/final_model.h5")
    # Update the name of the last conv layer based on your model architecture
    LAST_CONV_LAYER = "top_conv"  
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load model: %s", e)
    raise RuntimeError("Model file not found or could not be loaded.")

# Define class names (consistent with training)
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def read_imagefile(file) -> np.ndarray:
    """Read image file and convert it into a numpy array."""
    image_bytes = file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file")
    return image

def preprocess_image(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Resize and normalize image for model inference."""
    image_resized = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype('float32') / 255.0
    return image_normalized

@app.post("/predict", summary="Predict image class", dependencies=[Depends(get_current_username)])
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.
    Basic authentication is required.
    """
    try:
        image = read_imagefile(await file.read())
        preprocessed = preprocess_image(image)
        # Expand dimensions to match model input
        input_tensor = np.expand_dims(preprocessed, axis=0)
        prediction = model.predict(input_tensor)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_index]
        logger.info("Image predicted as %s", predicted_class)
        return JSONResponse(content={"predicted_class": predicted_class, "confidence": float(np.max(prediction))})
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Error processing the image.")

@app.post("/gradcam", summary="Generate Grad-CAM visualization", dependencies=[Depends(get_current_username)])
async def gradcam(file: UploadFile = File(...)):
    """
    Endpoint to generate and return a Grad-CAM visualization for the input image.
    """
    try:
        image = read_imagefile(await file.read())
        original_image = image.copy()
        preprocessed = preprocess_image(image)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        # Get Grad-CAM heatmap
        heatmap = get_gradcam_heatmap(model, preprocessed, last_conv_layer_name=LAST_CONV_LAYER)
        # Save and overlay the Grad-CAM heatmap (output saved as gradcam_output.jpg)
        output_path = "gradcam_output.jpg"
        save_and_display_gradcam(original_image, heatmap, output_path=output_path)
        logger.info("Grad-CAM generated and saved at %s", output_path)
        return JSONResponse(content={"message": "Grad-CAM generated successfully", "output_path": output_path})
    except Exception as e:
        logger.error("Grad-CAM error: %s", e)
        raise HTTPException(status_code=500, detail="Error generating Grad-CAM visualization.")
