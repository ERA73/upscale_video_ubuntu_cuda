# upscale_video_ubuntu_cuda

## Set Folder Permissions for the Current User

```bash
sudo chown -R $USER:$USER .
chmod -R u+rwX .
```

## Install All Dependencies

```bash
sudo ./setup_realesrgan.sh
```

If you want to use Docker, install the required dependencies as follows:
```bash
sudo ./setup-nvidia-docker.sh
```

## Upscale Your Video

Edit the first lines of the `main.py` file. For a simple execution, you only need to modify the following parameters:
```
ORIGINAL_VIDEO_NAME
SELECTED_MODEL
FRAME_TYPE
SCALE
```

Then, run the upscaling process with:
```bash
python3 main.py
```