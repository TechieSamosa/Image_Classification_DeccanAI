# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from src.data_preprocessing import load_and_preprocess_data
from src.utils import configure_logging

logger = configure_logging()

def evaluate_model():
    try:
        logger.info("Loading test data...")
        (_, _), (x_test, y_test) = load_and_preprocess_data(target_size=(224, 224))
        
        logger.info("Loading the saved model...")
        model = load_model('model/final_model.h5')
        
        logger.info("Evaluating model on test data...")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        logger.info("Test Loss: %.4f, Test Accuracy: %.4f", loss, accuracy)
        
        # Generate predictions and compute evaluation metrics
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Create and save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        # Ensure evaluation folder exists
        import os
        os.makedirs("evaluation", exist_ok=True)
        plt.savefig("evaluation/confusion_matrix.png")
        plt.show()
        
        # Classification report
        target_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                        'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        report = classification_report(y_test, y_pred_classes, target_names=target_names)
        logger.info("Classification Report:\n%s", report)
        
        # Save classification report to a file
        with open("evaluation/classification_report.txt", "w") as f:
            f.write(report)
    except Exception as e:
        logger.error("Error during evaluation: %s", str(e))
        raise

if __name__ == "__main__":
    evaluate_model()
