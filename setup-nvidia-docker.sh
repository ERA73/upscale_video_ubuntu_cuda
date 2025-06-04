#!/bin/bash

set -e

echo "🚀 Installing NVIDIA Container Toolkit for Docker on Ubuntu"

# Detect distribution
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
echo "📦 Detected distro: $distribution"

# Add GPG key and repository
echo "🔑 Adding GPG key and NVIDIA repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
echo "📦 Installing nvidia-container-toolkit..."
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker runtime
echo "⚙️ Configuring Docker to use NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
echo "🔄 Restarting Docker..."
sudo systemctl restart docker

# Final verification
echo "✅ Verifying installation..."
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

echo "🎉 Installation complete. Docker can now use the NVIDIA GPU."
