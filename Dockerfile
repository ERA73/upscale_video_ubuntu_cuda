# Base image with GPU support and Ubuntu
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set non-interactive environment to avoid apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    curl \
    unzip \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get install -y wget

# Alias so that "python" points to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /app

# Copy your scripts and requirements
COPY setup_realesrgan_docker.sh .
COPY frames_count.json .
COPY requirements.txt .
COPY *.py .

# Run additional installation script if needed
RUN bash setup_realesrgan_docker.sh

# Default command (you can change it to your final script)
CMD ["python3", "main.py"]
