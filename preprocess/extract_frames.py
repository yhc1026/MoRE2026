import os
import sys
import subprocess
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback


def is_valid_video(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path], capture_output=True, text=True, check=True)
        duration = float(result.stdout)
        return duration > 0
    except subprocess.CalledProcessError:
        return False
    except ValueError:
        return False


def extract_frame_cpu(video_path, timestamp, output_file):
    subprocess.run(
        ["ffmpeg",
             '-loglevel',
             'quiet',
             '-i',
            video_path,
             '-ss',
             str(timestamp),
             '-frames:v',
             '1',
             output_file],
        check=True)

def extract_frame(video_path: str, timestamp: float, output_file: str):
    """Extract a frame from a video at a specific timestamp."""
    cmd = [
        "ffmpeg",
        '-hwaccel', 'cuda',  # 只保留这一个 CUDA 参数
        '-loglevel', 'quiet',
        '-i', video_path,
        '-ss', str(timestamp),
        '-frames:v', '1',
        output_file  # 让 ffmpeg 自动选择编码器
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def process_video(video_path, output_folder, num_frames):
    if not is_valid_video(video_path):
        print(f"跳过无效视频: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    # 检查是否已处理
    if os.path.exists(video_output_folder):
        existing_frames = glob.glob(os.path.join(video_output_folder, "frame_*.jpg"))
        if len(existing_frames) == num_frames:
            print(f"Skip video {video_name}: {num_frames} frames already exist")
            return

    # 清理旧文件夹
    if os.path.exists(video_output_folder):
        shutil.rmtree(video_output_folder)
    os.makedirs(video_output_folder, exist_ok=True)

    # 获取视频时长
    try:
        result = subprocess.run(
            ["ffprobe",
             '-v', 'error',
             '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(result.stdout.strip())
    except:
        print(f"无法获取时长: {video_path}")
        return

    if duration <= 0:
        print(f"无效时长: {video_path}")
        return

    # 方案1：使用 fps 过滤器（最稳定）
    try:
        # 计算帧间隔（秒）
        interval = max(0.1, duration) / num_frames

        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',           # GPU解码
            '-loglevel', 'error',         # 只显示错误
            '-i', video_path,
            '-vf', f'fps=1/{interval:.3f}',  # 按时间间隔提取
            '-frames:v', str(num_frames),
            '-c:v', 'mjpeg',              # 指定编码器
            '-q:v', '2',                  # 质量
            '-vsync', '0',                # 禁用同步
            '-y',                         # 覆盖
            os.path.join(video_output_folder, 'frame_%03d.jpg')
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✓ GPU成功: {video_name}")
            return True
        else:
            print(f"⚠ GPU失败，尝试CPU: {video_name}")
    except Exception as e:
        print(f"⚠ GPU异常，尝试CPU: {video_name} - {str(e)[:50]}")


def main():
    num_frames = 16
    input_folder = "data/videos"
    output_folder = "data/frames_origin_{}".format(num_frames)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    video_paths = glob.glob(os.path.join(input_folder, '*.mp4'))

    for video_path in tqdm(video_paths, desc="Processing videos"):
        process_video(video_path, output_folder, num_frames)


if __name__ == "__main__":
    main()

