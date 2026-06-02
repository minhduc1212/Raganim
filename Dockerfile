FROM python:3.10

WORKDIR /code

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the FastAPI server
# Note: Hugging Face Spaces and other container platforms require listening on port 7860 and host 0.0.0.0
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
