# Use the official lightweight Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /auto-forecasting-mlops

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]