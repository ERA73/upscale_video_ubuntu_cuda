#!/bin/bash

set -e

echo "🚀 Installing environment for Real-ESRGAN on local GPU (CUDA 11.8)..."

# Upgrade pip and tools
pip install --upgrade pip setuptools wheel

# Install PyTorch + TorchVision compatible with CUDA 11.8
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

echo "🔄 Updating repositories..."
apt-get update

echo "📦 Installing ffmpeg..."
apt-get install -y ffmpeg

# Clone the official Real-ESRGAN repository
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# Install as local package
pip install -r requirements.txt
python setup.py develop
# Create weights folder and download updated model
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth -P weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -P weights
# check more models at https://github.com/xinntao/Real-ESRGAN/releases/
cd ..


# Create base structure for video processing
mkdir -p video output_video original_frames upscaled_frames jpgfiles temp

echo ""
echo "✅ Installation and setup completed successfully."
echo ""
echo "📂 Suggested folder structure:"
echo "  ├── output_video/               # Place for your original MP4 videos"
echo "  ├── original_frames/            # Extracted frames from the video"
echo "  ├── upscaled_frames/            # Enhanced frames"
echo "  ├── weights/                    # Pretrained models"
echo "  ├── Real-ESRGAN/                # Model source code"
echo "  ├── main.py                     # Script to upscale video"
echo ""
echo "👉 To get started:"
echo "  1. source venv/bin/activate"
echo "  2. Put your video in 'videos/video_720p.mp4'"
echo "  3. Run: python main.py"
