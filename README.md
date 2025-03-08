# Image Classification Project
**Author**: Aditya Khamitkar  
**Email**: [khamitkaraditya@gmail.com](mailto:khamitkaraditya@gmail.com)  
**Assessment for**: Soul AI by Deccan AI

---

## Overview
This repository demonstrates an end-to-end image classification pipeline on the **Fashion MNIST** dataset using **TensorFlow** and **FastAPI**. It includes:

1. A **Jupyter Notebook** (`Image_Classification.ipynb`) for data preprocessing, model training, and evaluation.
2. A **FastAPI** application (`app.py`) exposing a `/predict` endpoint for model inference.
3. A **Streamlit** frontend (`frontend.py`) for easy interaction with the model.
4. A **Dockerfile** to containerize the application.
5. A **report.md** detailing the approach, decisions, and implementation.

> **Note**: This project is part of an assessment for **Soul AI** by **Deccan AI**.

---

## Project Structure

```
├── app
│   ├── app.py          # FastAPI backend
│   ├── auth.py         # Authentication (HTTP Basic or Bearer Token)
│   ├── frontend.py     # Streamlit frontend
│   └── requirements.txt # Dependencies
├── Notebook
│   └── Image_Classification.ipynb # Model training and evaluation
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── report.md
```

### Key Components

1. **`Image_Classification.ipynb`**  
   - Loads the Fashion MNIST dataset.  
   - Preprocesses and augments images (resize, normalize).  
   - Builds and trains a CNN model.  
   - Evaluates performance using accuracy, confusion matrix, etc.  
   - Saves the trained model (`.h5` format).

2. **`app.py`**  
   - Loads the saved model.  
   - Implements the `/predict` endpoint with **FastAPI**.  
   - Accepts images via `POST` requests.  
   - Returns prediction results (class + confidence score).

3. **`auth.py`**  
   - Provides **basic authentication** or **token-based** authentication (depending on the version you choose).  
   - Restricts access to the `/predict` endpoint unless correct credentials are provided.

4. **`frontend.py`**  
   - Simple **Streamlit** interface for uploading images.  
   - Forwards the image to the FastAPI endpoint.  
   - Displays the predicted class and confidence score.

5. **`Dockerfile`**  
   - Containerizes the FastAPI app for easy deployment.  
   - Installs dependencies and exposes port `8000`.

6. **`report.md`**  
   - Summarizes data preprocessing, model selection, training, evaluation metrics, and deployment steps.  
   - Describes potential improvements like advanced data augmentation, transfer learning, or cloud deployment.

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/image-classification.git
   cd image-classification
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook** *(Optional for retraining)*  
   ```bash
   jupyter notebook Notebook/Image_Classification.ipynb
   ```
   - This step is only necessary if you want to retrain or modify the model.

4. **Start the FastAPI Server**  
   ```bash
   uvicorn app.app:app --reload
   ```
   - The API will be available at `http://127.0.0.1:8000`.

5. **Run the Streamlit Frontend**  
   ```bash
   streamlit run app/frontend.py
   ```
   - Visit the provided URL (typically `http://localhost:8501`) to upload images and get predictions.

---

## Authentication
Depending on the `auth.py` version you choose, you’ll
- Provide a **Bearer token** in your request header.

Example using Bearer Token:
```bash
curl -X POST -H "Authorization: Bearer mysecuretoken" \
     -F "file=@path_to_image.jpg" \
     http://127.0.0.1:8000/predict
```

---

## Docker Deployment
To run everything in a container:
1. **Build the Docker image**:
   ```bash
   docker build -t image-classification .
   ```
2. **Run the container**:
   ```bash
   docker run -p 8000:8000 image-classification
   ```
3. The FastAPI app will be available at `http://127.0.0.1:8000`.

---

## Contributing
Feel free to open issues or submit pull requests. This is a demonstration project, so any improvements or new ideas are welcome.

---

## License
This project is licensed under the terms of the [MIT License](LICENSE).

---

### Author
**Aditya Khamitkar**  
[khamitkaraditya@gmail.com](mailto:khamitkaraditya@gmail.com)

---

## Special Notes
- This code was written as part of an **Assessment for Soul AI by Deccan AI**.  
- For production use, secure your credentials properly (avoid hardcoding) and consider advanced authentication/authorization schemes.  
- Future improvements could include advanced data augmentation, transfer learning (e.g., MobileNet, EfficientNet), Grad-CAM for interpretability, and cloud deployment (AWS, GCP, Azure).

---
