import os
import cv2
import json
import torch
import shutil
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from concurrent.futures import ThreadPoolExecutor, as_completed
from parameters import (
    BASE_DIR, FRAME_RATE, CHUNK_SIZE, FRAMES_DIR, UPSCALED_FRAMES_DIR,
    JPG_FRAMES_DIR, AUDIO_PATH, OUTPUT_VIDEO_PATH, BAR_FORMAT
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === YOUR CONFIGURATION ===
ORIGINAL_VIDEO_NAME = 'your video.mp4'  # Change to your video file name
SELECTED_MODEL = 'realesr-general-x4v3'  # Change according to the desired model
# Available models:
#       realesr-animevideov3
#       realesr-general-wdn-x4v3
#       realesr-general-x4v3
#       RealESRGAN_x4plus_anime_6B
FRAME_TYPE = 'jpg'  # Change to 'jpg' or 'png'
SCALE = 2 # Change to 2, 3, up to 4


# === CONFIGURATION ===
VIDEO_PATH = os.path.join(BASE_DIR, ORIGINAL_VIDEO_NAME)
OUTPUT_FULL_VIDEO_PATH = os.path.join(BASE_DIR, f'{ORIGINAL_VIDEO_NAME[:-4]}_new.mp4')
MODEL_PATH = os.path.join(BASE_DIR, 'Real-ESRGAN', 'weights', f'{SELECTED_MODEL}.pth')




def get_model():
    if SELECTED_MODEL == 'realesrgan-x4plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
    elif SELECTED_MODEL == 'realesrgan-x4plus-anime':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=6, num_grow_ch=32, scale=4)
    elif SELECTED_MODEL == 'realesr-animevideov3':
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                               num_feat=64, num_conv=32,
                               upscale=4, act_type='prelu')
    elif SELECTED_MODEL == 'realesr-general-x4v3':
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                               num_feat=64, num_conv=32,
                               upscale=4, act_type='prelu')
    else:
        raise ValueError(f"Unsupported model: {SELECTED_MODEL}")

# === FULL VALIDATION (SANITY CHECK) ===
def sanity_check():
    print("🔍 Validating environment...")

    if not torch.cuda.is_available():
        raise EnvironmentError("🚫 GPU not available. Make sure you have CUDA drivers and GPU support in PyTorch.")
    print("✅ GPU available.")

    for module in ['cv2', 'numpy', 'PIL', 'tqdm']:
        print(f"✅ {module} loaded successfully.")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"🚫 Video file not found: {VIDEO_PATH}")
    print(f"✅ Video found: {VIDEO_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"🚫 Model not found at: {MODEL_PATH}")
    print(f"✅ Model found: {MODEL_PATH}")

    # Test model loading
    try:
        model = get_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=False)
        print("✅ Model architecture compatible.")
    except Exception as e:
        raise RuntimeError(f"🚫 Error loading model: {e}")

# === FUNCTIONS ===

def get_frames_count():
    try:
        with open("frames_count.json", "r") as f:
            frames_data = json.loads("".join(f.readlines()))
        return frames_data["count"]
    except:
        return 0

def set_frames_count(value):
    with open("frames_count.json", "w") as f:
        f.write(json.dumps({"count": value}, indent=4))

def get_video_fps(video_path: str) -> float:
    """
    Gets the frame rate (FPS) of the video using ffprobe.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = json.loads(result.stdout)
        rate_str = output['streams'][0]['r_frame_rate']
        num, denom = map(int, rate_str.split('/'))
        fps = num / denom
        return round(fps, 3)
    except Exception as e:
        print(f"⚠️ Error getting FPS from video: {e}")
        return FRAME_RATE

def get_video_info(video_path: str) -> dict:
    """
    Returns relevant video information: duration, resolution, fps, codecs, bitrate.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_format',
        '-show_streams',
        '-print_format', 'json',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)

        # General format
        format_info = data.get('format', {})
        duration = float(format_info.get('duration', 0))
        bit_rate = int(format_info.get('bit_rate', 0))

        # Streams
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)

        width = int(video_stream['width']) if video_stream else None
        height = int(video_stream['height']) if video_stream else None
        codec_video = video_stream['codec_name'] if video_stream else None

        fps = None
        if video_stream and 'r_frame_rate' in video_stream:
            num, denom = map(int, video_stream['r_frame_rate'].split('/'))
            fps = round(num / denom, 3) if denom != 0 else None

        codec_audio = audio_stream['codec_name'] if audio_stream else None

        return {
            'duration_sec': round(duration, 2),
            'width': width,
            'height': height,
            'fps': fps,
            'video_codec': codec_video,
            'audio_codec': codec_audio,
            'bit_rate': bit_rate
        }
    except Exception as e:
        print(f"❌ Error analyzing video: {e}")
        return {}

def extract_frames(video_path, output_dir):
    print(f"Extracting frames ...")
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = get_frames_count()
    if count:
        for i in range(count):
            success, image = vidcap.read()
    while success:
        if count and count % CHUNK_SIZE == 0:
            upscale_frames(FRAMES_DIR, UPSCALED_FRAMES_DIR, MODEL_PATH)
            set_frames_count(count)
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.{FRAME_TYPE}"), image)
        success, image = vidcap.read()
        count += 1
    if count and count % CHUNK_SIZE > 0:
        upscale_frames(FRAMES_DIR, UPSCALED_FRAMES_DIR, MODEL_PATH)
        set_frames_count(count)
    vidcap.release()
    print(f"📸 {count} frames extracted.")

def process_image(fname, upsampler):
    img_path = os.path.join(FRAMES_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    with torch.no_grad():
        output, _ = upsampler.enhance(img_np, outscale=SCALE)

    out_path = os.path.join(UPSCALED_FRAMES_DIR, fname)
    Image.fromarray(output).save(out_path)

def is_valid_image(path):
    """Checks if an image is valid and not corrupted."""
    full_path = os.path.join(UPSCALED_FRAMES_DIR, path)
    try:
        with Image.open(full_path) as img:
            img.verify()
        return path
    except Exception:
        os.remove(full_path)
        return None
    
def clear_directory(dir_path):
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        try:
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.unlink(full_path)  # Remove file or symlink
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)  # Remove subdirectory
        except Exception as e:
            print(f"Error deleting {full_path}: {e}")

def validate_output_images_parallel(output_dir, max_workers=4, quick_check=True):
    """Checks 1% last images in the directory in parallel with a progress bar."""
    files = sorted([f for f in os.listdir(output_dir) if f.endswith(f'.{FRAME_TYPE}')])
    if not files:
        print("No images found in the output directory.")
        return []
    valid_files = []
    if quick_check and len(files) >= 10:
        print("Performing quick check on the last 1% of images...")
        one_percent = int(len(files) // 100)
        valid_files = files[:-max(1, one_percent)]  # Keep all but the last 1% for validation
        files = files[-max(1, one_percent):] # Only validate the last 1%
        print(f"First file: {files[0]} . . . . Last file: {files[-1]}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(is_valid_image, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🧪 Checking images", bar_format=BAR_FORMAT):
            path = future.result()
            if path:
                valid_files.append(path)

    return valid_files

def upscale_frames(input_dir, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Used: {device}")

    model = get_model()

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        device=device,
        dni_weight=None,
        tile=0,  # or tile=128 if you have low GPU memory
        tile_pad=10,
        pre_pad=0
    )
    input_files = set(f for f in os.listdir(input_dir) if f.endswith(f'.{FRAME_TYPE}'))
    output_files = set(validate_output_images_parallel(output_dir, max_workers=16))
    frame_files = sorted(list(input_files - output_files))
    process = partial(process_image, upsampler=upsampler)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process, fname) for fname in frame_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="⚙️ Processing images", bar_format=BAR_FORMAT):
            pass

    clear_directory(FRAMES_DIR)
    print(f"✅ Enhanced frames saved in {output_dir}")

def convert_single_png_to_jpg(fname, source_dir, quality=90, delete_original=False):
    if not fname.lower().endswith('.png'):
        return
    path_png = os.path.join(source_dir, fname)
    path_jpg = os.path.join(JPG_FRAMES_DIR, fname[:-4] + ".jpg")
    try:
        with Image.open(path_png) as img:
            rgb = img.convert("RGB")
            rgb.save(path_jpg, "JPEG", quality=quality)
        if delete_original:
            os.remove(path_png)
        return True
    except Exception as e:
        print(f"❌ Error converting {fname}: {e}")
        return False

def convert_png_to_jpg_parallel(source_dir, quality=90, delete_original=False, max_workers=8):
    png_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(convert_single_png_to_jpg, f, source_dir, quality, delete_original)
            for f in png_files
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="🚀 Converting PNG → JPG", bar_format=BAR_FORMAT):
            pass  # just for progress bar

def assemble_video(frame_dir, output_path, fps=30):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        raise RuntimeError("No frames found to assemble the video.")

    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fname in tqdm(frame_files, desc="🎞️ Building video", bar_format=BAR_FORMAT):
        frame = cv2.imread(os.path.join(frame_dir, fname))
        out.write(frame)

    out.release()
    clear_directory(UPSCALED_FRAMES_DIR)
    print(f"🎉 Final video saved at: {output_path}")

def extract_audio(input_video_path: str, output_audio_path: str) -> None:
    """
    Extracts audio from the video. If direct copy is not possible, converts to AAC.
    """
    def run_ffmpeg(command: list[str]) -> tuple[bool, str]:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr

    print(f"🔊 Extracting audio from: {input_video_path}")

    # 1. Try direct copy
    copy_cmd = [
        'ffmpeg', '-y',
        '-i', input_video_path,
        '-vn',
        '-acodec', 'copy',
        output_audio_path
    ]
    success, log = run_ffmpeg(copy_cmd)

    if success:
        print("✅ Audio extracted successfully with direct copy.")
        return

    print("⚠️ Direct extraction failed, trying AAC transcoding...")
    print("🪵 Error details:", log.strip())

    # 2. Fallback to transcoding (e.g., for AMR or incompatible codecs)
    transcode_cmd = [
        'ffmpeg', '-y',
        '-i', input_video_path,
        '-vn',
        '-acodec', 'aac',
        output_audio_path
    ]
    success, log = run_ffmpeg(transcode_cmd)

    if success:
        print("✅ Audio extracted successfully with transcoding.")
    else:
        print("❌ Extraction failed even with AAC conversion.")
        print("🪵 Error details:", log.strip())
        raise RuntimeError("Could not extract audio from video.")

def add_audio(video_path, audio_path, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    subprocess.run(command, check=True)

# === MAIN PROCESS ===

if __name__ == '__main__':
    sanity_check()
    VIDEO_METADATA = get_video_info(VIDEO_PATH)
    FRAME_RATE = VIDEO_METADATA.get('fps', FRAME_RATE)
    extract_audio(VIDEO_PATH, AUDIO_PATH)
    extract_frames(VIDEO_PATH, FRAMES_DIR)
    assemble_video(UPSCALED_FRAMES_DIR, OUTPUT_VIDEO_PATH, FRAME_RATE)
    add_audio(OUTPUT_VIDEO_PATH, AUDIO_PATH, OUTPUT_FULL_VIDEO_PATH)
    set_frames_count(0)
    os.unlink(AUDIO_PATH)
