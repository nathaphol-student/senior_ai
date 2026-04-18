# Start from the official PyTorch runtime image with CUDA 12.1
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

# Install system dependencies for OpenCV (required for libGL, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install other deps
RUN pip install -r requirements.txt

COPY . .

EXPOSE ${PORT:-8084}

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8084}
