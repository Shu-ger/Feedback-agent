# -----------------------------
# 1. Use a lightweight Python base image
# -----------------------------
FROM python:3.11-slim
# -----------------------------
# 2. Set working directory inside the container
# -----------------------------
WORKDIR /app
# -----------------------------
# 3. Install system dependencies (optional, needed for some packages)
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------
# 4. Copy requirements and install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# -----------------------------
# 5. Copy the application code
# -----------------------------
COPY . .
# -----------------------------
# 6. Expose the port that FastAPI runs on
# -----------------------------
EXPOSE 8000
# -----------------------------
# 7. Set environment variables (optional)
# -----------------------------
ENV PYTHONUNBUFFERED=1
# -----------------------------
# 8. Run the FastAPI app using uvicorn
# -----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]