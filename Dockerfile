# Select PyTorch image from nvidia
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app/

# Install code requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt