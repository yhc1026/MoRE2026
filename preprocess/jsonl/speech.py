# import os
# import json
# import whisper
# from tqdm import tqdm
#
# # 配置路径（替换为你的数据集路径）
# dataset_dir = r"D:\code\LAB\MoRE2026\data"
# audio_dir = os.path.join(dataset_dir, "audios")  # 存放每个视频的音频文件（.wav格式）
# output_file = os.path.join(dataset_dir, "speech.jsonl")
# vid_file = r"D:\code\LAB\MoRE2026\data\vids\vids.csv"
# ffmpeg_path = r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
# # 1. 加载视频ID列表
# with open(vid_file, "r") as f:
#     vids = [line.strip() for line in f]
#
# # 2. 加载Whisper模型（论文用预训练版，small模型平衡速度和准确率）
# model = whisper.load_model("small")  # 支持 "base" "small" "medium"，越大越准但越慢
#
# # 3. 批量转录音频并生成JSONL
# with open(output_file, "w", encoding="utf-8") as f_out:
#     for vid in tqdm(vids, desc="转录音频为文本"):
#         audio_path = os.path.join(audio_dir, f"{vid}.wav")  # 音频文件命名：{vid}.wav
#         if not os.path.exists(audio_path):
#             # 音频缺失时补空文本（避免后续特征提取报错）
#             transcript = ""
#         else:
#             # 转录（支持自动识别语言）
#             result = model.transcribe(audio_path, language="en")  # HateMM是英文，中文数据集用 language="zh"
#             transcript = result["text"].strip()  # 提取转录文本
#
#         # 4. 按JSONL格式写入（字段：vid + transcript）
#         json_line = {"vid": vid, "transcript": transcript}
#         f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
import os
import json
import whisper
from tqdm import tqdm
import torch  # 新增导入

# ========== 仅添加这部分：指定FFmpeg路径（核心修复） ==========
# 设置FFmpeg路径，让Whisper能找到可执行文件
ffmpeg_dir = os.path.dirname(r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe")
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
print(f"✅ 已设置 FFmpeg 路径: {ffmpeg_dir}")

# 配置路径（完全保留你的原始配置，未做任何修改）
dataset_dir = r"D:\code\LAB\MoRE2026\data"
audio_dir = os.path.join(dataset_dir, "audios")  # 存放每个视频的音频文件（.wav格式）
output_file = os.path.join(dataset_dir, "speech.jsonl")
vid_file = r"D:\code\LAB\MoRE2026\data\vids\vids.csv"
ffmpeg_path = r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

# 1. 加载视频ID列表（完全保留你的逻辑）
with open(vid_file, "r") as f:
    vids = [line.strip() for line in f]

# 2. 修复：使用 torch.cuda.is_available() 替代 whisper.utils.is_cuda_available()
# 检测 CUDA 可用性
use_cuda = torch.cuda.is_available()
print(f"🖥️  CUDA 可用: {use_cuda}")
print(f"💾  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if use_cuda else "💾  使用 CPU")

# 加载 Whisper 模型
device = "cuda" if use_cuda else "cpu"
model = whisper.load_model("small", device=device)

# 3. 批量转录音频并生成JSONL（完全保留你的输出格式）
with open(output_file, "w", encoding="utf-8") as f_out:
    for vid in tqdm(vids, desc="转录音频为文本"):
        audio_path = os.path.join(audio_dir, f"{vid}.wav")
        if not os.path.exists(audio_path):
            transcript = ""
            print(f"\n⚠️  音频文件缺失: {vid}.wav")  # 可选：打印缺失文件
        else:
            try:
                # 转录时根据设备选择是否使用 fp16 加速
                result = model.transcribe(
                    audio_path,
                    language="en",
                    fp16=(device == "cuda"),  # 仅 CUDA 设备使用 fp16
                    verbose=False  # 关闭详细日志
                )
                transcript = result["text"].strip()

                # 可选：打印转录长度
                if len(transcript) > 0:
                    print(f"\n✅  {vid}: {len(transcript)} 字符")

            except Exception as e:
                print(f"\n❌  处理 {vid} 时出错：{str(e)}")
                transcript = ""

        # 4. 完全保留你的JSONL输出格式
        json_line = {"vid": vid, "transcript": transcript}
        f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

print(f"\n✅ 转录完成！结果已保存到：{output_file}")
print(f"📊 处理了 {len(vids)} 个视频ID")
print(f"📌 输出格式：JSONL（每行{{'vid': '视频ID', 'transcript': '转录文本'}}）")