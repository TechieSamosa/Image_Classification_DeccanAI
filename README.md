# Image Classification Project

This repository implements an image classification model on the CIFAR-10 dataset using EfficientNetB0 with transfer learning. The project covers the complete pipeline from exploratory data analysis (EDA) and data preprocessing to model training, evaluation, and deployment as a REST API. Bonus features include Grad-CAM explainability, logging and error handling, and an interactive frontend built with Streamlit.

## Project Structure

```
image-classification-project/
├── app/
│   ├── app.py                # FastAPI application for serving predictions and Grad-CAM visualizations
│   ├── auth.py               # Basic authentication for API endpoints
│   ├── frontend.py           # Streamlit frontend for interactive demo
│   └── requirements.txt      # Dependencies for the API and frontend
├── data/
│   ├── raw/                  # Original dataset files (if applicable)
│   └── processed/            # Preprocessed images (optional)
├── evaluation/
│   ├── confusion_matrix.png  # Confusion matrix plot generated during evaluation
│   └── classification_report.txt  # Classification report (precision, recall, F1-score)
├── logs/
│   └── app.log               # Application logs
├── model/
│   ├── best_model.h5         # Best model checkpoint during training
│   └── final_model.h5        # Final saved model for deployment
├── notebooks/
│   └── EDA.ipynb             # Jupyter Notebook for exploratory data analysis
├── src/
│   ├── __init__.py           # Package initializer
│   ├── data_preprocessing.py # Data loading and preprocessing functions
│   ├── evaluate.py           # Model evaluation script
│   ├── explainability.py     # Grad-CAM implementation for model explainability
│   ├── model.py              # Model architecture definition (EfficientNetB0 based)
│   ├── train.py              # Training script with callbacks, logging, and checkpointing
│   └── utils.py              # Utility functions including logging configuration
├── Dockerfile                # Docker configuration file (if containerizing the API)
├── README.md                 # Project overview and setup instructions
└── report.md                 # Detailed report documenting the project methodology
```

## Getting Started

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)
- Docker (optional, if containerizing the API)

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/image-classification-project.git
   cd image-classification-project
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   - Install main project dependencies:
     ```bash
     pip install -r app/requirements.txt
     pip install tensorflow opencv-python matplotlib seaborn scikit-learn pillow python-multipart streamlit
     ```
   
4. **Data Preparation & EDA:**
   - Open the `notebooks/EDA.ipynb` notebook in Jupyter Notebook or JupyterLab to review the exploratory data analysis.

5. **Training the Model:**
   - Run the training script:
     ```bash
     python src/train.py
     ```
   - The model checkpoints and final model will be saved in the `model/` directory.

6. **Evaluating the Model:**
   - Execute the evaluation script:
     ```bash
     python src/evaluate.py
     ```
   - Check the `evaluation/` directory for confusion matrix and classification report.

7. **Running the API:**
   - Start the FastAPI server with Uvicorn:
     ```bash
     uvicorn app.app:app --reload
     ```
   - The API will be accessible at [http://localhost:8000](http://localhost:8000).

8. **Using the Frontend:**
   - Launch the Streamlit app:
     ```bash
     streamlit run app/frontend.py
     ```
   - Use the web interface to upload images for classification and view Grad-CAM visualizations.

9. **(Optional) Docker:**
   - Build the Docker image:
     ```bash
     docker build -t image-classification-api .
     ```
   - Run the Docker container:
     ```bash
     docker run -p 8000:8000 image-classification-api
     ```

## API Authentication

The API endpoints are secured using basic HTTP authentication. Default credentials are:

- **Username:** `Khamitkar`
- **Password:** `Deccan@Ai`

You can adjust these credentials in the `app/auth.py` file.

## Bonus Features

- **Grad-CAM Explainability:** Use the `/gradcam` endpoint to generate visual explanations.
- **Logging & Error Handling:** Logs are saved in the `logs/` directory.
- **Streamlit Frontend:** Interactive demo for image classification and Grad-CAM visualizations.

## Contact

For any questions or issues, please open an issue in this repository or contact me at [e-Mail](mailto:khamitkaraditya@gmail.com).

Enjoy exploring and improving the project!


