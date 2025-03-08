
# Image Classification Model - Report

## 1. Introduction

This report details the development, training, evaluation, and deployment of an image classification model for the assessment task of a Computer Vision Engineer for Soul AI by Deccan AI. The objective of this project is to process image data, train a CNN-based model on the Fashion MNIST dataset, and deploy the model via a REST API using FastAPI. A lightweight Streamlit frontend is also provided to facilitate easy interaction with the model.

**Author**: Aditya Khamitkar  
**Email**: [khamitkaraditya@gmail.com](mailto:khamitkaraditya@gmail.com)

---

## 2. Data Preprocessing & Exploratory Data Analysis (EDA)

- **Dataset**: Fashion MNIST  
  - **Training Samples**: 60,000 images of size 28×28  
  - **Test Samples**: 10,000 images of size 28×28  
  - **Classes**: 10 (represented as digits 0–9)

- **Preprocessing Steps**:
  - **Resizing**: Converted images from 28×28 to 96×96.
  - **Channel Conversion**: Converted grayscale images to RGB (3 channels).
  - **Normalization**: Scaled pixel values to the range [0, 1].
  - **Dataset Splitting**: The dataset was split into training, validation, and testing sets.

- **Exploratory Analysis**:  
  Sample images were visualized along with their corresponding class labels to verify the preprocessing steps.

---

## 3. Model Architecture & Training Setup

The model is constructed using TensorFlow/Keras with MobileNetV2 as the base feature extractor. The architecture is as follows:

- **Base Model**: `mobilenetv2_1.00_96`
  - **Output Shape**: (None, 3, 3, 1280)
- **Additional Layers**:
  - **Global Average Pooling**: Reduces spatial dimensions.
  - **Dropout**: Regularizes the model to prevent overfitting.
  - **Dense Layer**: Final classification layer with 10 neurons (one per class) using softmax activation.

**Training Details**:
- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Callbacks**:
  - Early stopping to prevent overfitting.
  - Learning rate scheduling via `ReduceLROnPlateau` to adjust the learning rate when validation performance plateaus.
- **Epochs**: Up to 20 epochs were run.

**Training Logs Snapshot**:
```
Epoch 1/20 - Training Accuracy: 75.36%, Val Accuracy: 85.92%
Epoch 2/20 - Training Accuracy: 84.81%, Val Accuracy: 87.27%
...
Epoch 16/20 - Training Accuracy: 86.52%, Val Accuracy: 87.88%
Test Accuracy: 87.01%
```

---

## 4. Evaluation Results

After training, the model was evaluated on the test set, achieving:

- **Test Accuracy**: 87.01%

### Detailed Classification Report:
```
              precision    recall  f1-score   support

 T-shirt/top       0.79      0.86      0.83      1000
     Trouser       0.99      0.96      0.98      1000
    Pullover       0.82      0.82      0.82      1000
       Dress       0.88      0.83      0.85      1000
        Coat       0.77      0.83      0.80      1000
      Sandal       0.97      0.94      0.95      1000
       Shirt       0.65      0.60      0.62      1000
     Sneaker       0.90      0.95      0.92      1000
         Bag       0.97      0.99      0.98      1000
  Ankle boot       0.96      0.94      0.94      1000

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000
```

These metrics indicate strong performance across most classes, with some variation (e.g., "Shirt" class showed slightly lower performance).

---

## 5. Grad-CAM Visualization

To enhance model interpretability, Grad-CAM was implemented to visualize the regions of the image that contributed most to the model's decision. The technique leverages the output of the last convolutional layer (`block_16_project` from MobileNetV2) to generate a heatmap overlay on the original image.

**Steps**:
- A custom function computes the gradient of the target class with respect to the output of the selected convolutional layer.
- The resulting heatmap is normalized and resized to match the input image dimensions.
- The overlay helps to understand which parts of the image the model focused on during prediction.


---

## 6. Model Deployment

The trained model is deployed as a REST API using FastAPI, and a lightweight frontend is built using Streamlit.

- **FastAPI Backend** (`app.py`):
  - Loads the saved model (`model/final_model.h5`).
  - Implements a `/predict` endpoint to accept an image and return the predicted class with confidence.
  - Incorporates authentication to secure the endpoint.

- **Streamlit Frontend** (`frontend.py`):
  - Provides a user-friendly interface for uploading images.
  - Sends the image to the FastAPI endpoint for prediction.
  - Displays the prediction result to the user.

- **Containerization**:
  - A `Dockerfile` is provided to containerize the FastAPI application, facilitating deployment across various environments.

---

## 7. Future Improvements

Potential enhancements for future work include:
- **Advanced Data Augmentation**: Utilize more aggressive augmentation strategies to further improve model robustness.
- **Transfer Learning**: Experiment with alternative architectures like ResNet or EfficientNet for potential performance gains.
- **Model Explainability**: Integrate additional interpretability tools (e.g., SHAP) alongside Grad-CAM.
- **Cloud Deployment**: Deploy the API on platforms such as AWS, GCP, or Azure for scalability.
- **Enhanced Logging & Error Handling**: Implement robust logging and error management in the API.

---

## 8. Conclusion

This project successfully demonstrates the complete pipeline of an image classification system—from data preprocessing and model training to API deployment and interactive frontend development. With a test accuracy of 87.01% and a detailed Grad-CAM analysis, the model shows promising performance on the Fashion MNIST dataset. This solution is modular and scalable, making it well-suited for further enhancements and real-world deployment.

---

**Aditya Khamitkar**  
[khamitkaraditya@gmail.com](mailto:khamitkaraditya@gmail.com)  
*Assessment for Soul AI by Deccan AI*
