# Use a slim Python 3.9 image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the API requirements file and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install any additional dependencies not in requirements.txt
RUN pip install tensorflow opencv-python matplotlib seaborn scikit-learn pillow python-multipart streamlit

# Copy the rest of your application code into the container
COPY . .

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the API using Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
