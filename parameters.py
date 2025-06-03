import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_RATE = 30
CHUNK_SIZE = 1000
FRAMES_DIR = os.path.join(BASE_DIR, 'frames')
UPSCALED_FRAMES_DIR = os.path.join(BASE_DIR, 'upscaled_frames')
JPG_FRAMES_DIR = os.path.join(BASE_DIR, 'jpgfiles') # optional, for JPG conversion
AUDIO_PATH = os.path.join(BASE_DIR, 'audio_temp.aac')
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, 'output_video', 'video_temp.mp4')
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {percentage:.2f}%]"