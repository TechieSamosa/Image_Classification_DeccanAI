# Image Classification Project

**Author**: Aditya Khamitkar  
**Email**: [khamitkaraditya@gmail.com](mailto:khamitkaraditya@gmail.com)  
**Assessment for**: Soul AI by Deccan AI  

---

## Overview
This project demonstrates an end-to-end image classification pipeline using **TensorFlow**, **FastAPI**, and **Streamlit** on the **Fashion MNIST** dataset. It includes:

1. **Jupyter Notebook** (`Image_Classification.ipynb`) for model training and evaluation.
2. **FastAPI backend** (`app.py`) serving a `/predict` endpoint.
3. **Streamlit frontend** (`frontend.py`) for easy interaction.
4. **Authentication module** (`auth.py`) for secure API access.
5. **Docker support** for easy deployment.
6. **Comprehensive documentation** (`report.md`).

This project is part of an assessment for **Soul AI** by **Deccan AI**.

---

## Project Structure

```
├── app
│   ├── app.py          # FastAPI backend
│   ├── auth.py         # Authentication module
│   ├── frontend.py     # Streamlit frontend
│   ├── requirements.txt # Dependencies
├── Notebook
│   └── Image_Classification.ipynb # Model training & evaluation
├── model
│   └── final_model.h5  # Trained model
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── report.md
```

### Key Components

#### **1. `Image_Classification.ipynb`**
- Loads **Fashion MNIST dataset**.
- Preprocesses and augments images.
- Builds and trains a **CNN model**.
- Saves the trained model in `.h5` format.

#### **2. `app.py` (FastAPI Backend)**
- Loads the saved model.
- Implements the `/predict` endpoint.
- Accepts images via `POST` requests.
- Returns prediction results (class + confidence score).

#### **3. `auth.py` (Authentication Module)**
- Implements **Bearer Token authentication**.
- Restricts access to the API.

#### **4. `frontend.py` (Streamlit UI)**
- Provides a simple UI to upload images.
- Calls the `/predict` API.
- Displays the **predicted class & confidence score**.

#### **5. `Dockerfile` (Containerization)**
- Containerizes the FastAPI application.
- Installs dependencies and exposes port `8000`.

#### **6. `report.md` (Documentation)**
- Summarizes **data preprocessing, model selection, training, and deployment**.
- Discusses potential improvements (e.g., transfer learning, cloud deployment).

---

## Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/image-classification.git
cd image-classification
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Jupyter Notebook** *(Optional for retraining)*
```bash
jupyter notebook Notebook/Image_Classification.ipynb
```
- This step is **only necessary** if you want to **retrain the model**.

### **4️⃣ Start the FastAPI Server**
```bash
uvicorn app:app --reload
```
- The API will be available at `http://127.0.0.1:8000`.

### **5️⃣ Run the Streamlit Frontend**
```bash
streamlit run frontend.py
```
- Visit `http://localhost:8501` to upload images & get predictions.

---

## Authentication (Bearer Token)

To make API requests, you **must** provide a valid authentication token.

#### **Example: cURL Request with Token**
```bash
curl -X POST -H "Authorization: Bearer mysecuretoken" \
     -F "file=@path_to_image.jpg" \
     http://127.0.0.1:8000/predict
```

---

## Running with Docker

### **1️⃣ Build the Docker Image**
```bash
docker build -t image-classification .
```

### **2️⃣ Run the Container**
```bash
docker run -p 8000:8000 image-classification
```

- The **FastAPI app** will be available at `http://127.0.0.1:8000`.
- To access the **frontend**, run Streamlit separately.

---

## Debugging & Troubleshooting

| **Issue** | **Possible Cause & Fix** |
|-----------|--------------------------|
| **No response / app hangs** | The FastAPI backend (`uvicorn`) might not be running. Run: `uvicorn app:app --reload` |
| **403: Forbidden** | Ensure `Authorization` header in `frontend.py` has `Bearer mysecuretoken`. |
| **500: Internal Server Error** | Model input might not match. Try printing `processed_image.shape` before `model.predict()`. |
| **404: Not Found** | Check if the API URL is correct in `frontend.py` (`http://127.0.0.1:8000/predict`). |

---

## Future Improvements
- **Use Transfer Learning**: Models like MobileNetV2 or EfficientNet can improve accuracy.
- **Grad-CAM Visualization**: Implement Grad-CAM to interpret model predictions.
- **Deploy on Cloud**: AWS, GCP, or Azure for scalability.
- **Improve Authentication**: Use OAuth2 or JWT instead of a static token.
- **Enhance UI**: Add drag-and-drop, batch uploads, and result history.

---

## Contributing
Feel free to open issues or submit pull requests. This is a demonstration project, so any improvements or new ideas are welcome.

---

## License
This project is licensed under the terms of the [MIT License](LICENSE).

---

### **Author**
**Aditya Khamitkar**  
[khamitkaraditya@gmail.com](mailto:khamitkaraditya@gmail.com)

---

## Special Notes
- This project is part of an **Assessment for Soul AI by Deccan AI**.
- For production use, **secure credentials properly** (avoid hardcoding).
- Future enhancements may include **cloud deployment, improved UI, and deeper model analysis**.

---

