from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from auth import get_current_user

# Create the FastAPI application
app = FastAPI()

# Load the trained model using a corrected path (use forward slashes or a raw string)
model = tf.keras.models.load_model(r"Notebook/model/final_model.h5")

# Define the class names for predictions
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
IMG_SIZE = 96

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    try:
        # Read and preprocess the uploaded image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)
        
        # Perform prediction
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
