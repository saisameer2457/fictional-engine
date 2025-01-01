# Use the official Python image from Docker Hub
FROM python:3.12.0-slim

# Set the working directory to the MNIST folder
WORKDIR /app/MNIST

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY . .

# Set the command to run the application
CMD ["python", "app.py"]