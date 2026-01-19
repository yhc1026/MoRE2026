import os
import json
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


def generate_video_captions(input_folder, output_jsonl, model_path):
    # 加载BLIP2模型
    print("加载BLIP2模型...")
    if torch.cuda.is_available():
        print("cuda")
    else:
        return 0
    device = "cuda"

    torch_dtype = torch.float16
    processor = Blip2Processor.from_pretrained(model_path)
    # model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",  # 自动分配到GPU
        load_in_8bit=False,  # 如果内存很小可以尝试8bit量化
        load_in_4bit=False  # 或者4bit量化
    )

    # model.to(device)
    # model.eval()

    # 获取所有视频文件夹
    video_folders = []
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path):
            video_folders.append(item_path)

    print(f"找到 {len(video_folders)} 个视频文件夹")

    # 创建输出文件
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'r', encoding='utf-8') as f:
        # 读取所有行
        lines = f.readlines()

        # 获取最后一行（跳过可能的空行）
        last_line = lines[-1].strip()
        data = json.loads(last_line)
        last_vid = data.get('vid')

    with open(output_path, 'a', encoding='utf-8') as f:
        # 遍历每个视频文件夹
        for video_idx, video_folder in enumerate(video_folders, 1):
            video_name = Path(video_folder).name
            if last_vid != "":
                if video_name <= last_vid:
                    print("跳过视频"+ video_name)
                    continue
            print(f"处理视频 {video_idx}/{len(video_folders)}: {video_name}")

            # 获取所有帧图片
            frame_files = []
            for ext in ['jpg']:
                frame_files.extend(Path(video_folder).glob(f'*.{ext}'))

            # 按文件名排序，确保顺序正确
            frame_files = sorted(frame_files, key=lambda x: x.name)

            if not frame_files:
                print(f"警告: 文件夹 {video_name} 中没有找到图片")
                continue

            print(f"  找到 {len(frame_files)} 帧图片")

            # 为每帧生成描述
            frame_captions = []

            for frame_idx, frame_path in enumerate(frame_files, 1):
                try:
                    # 加载图片
                    image = Image.open(frame_path).convert('RGB')

                    # 生成描述
                    with torch.no_grad():
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        generated_ids = model.generate(**inputs, max_new_tokens=50)
                        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    frame_captions.append(" "+caption)

                    if frame_idx % 10 == 0:
                        print(f"    已处理 {frame_idx}/{len(frame_files)} 帧")

                except Exception as e:
                    print(f"    处理 {frame_path.name} 时出错: {e}")

            # 构建视频级别的JSON记录
            # video_record = {
            #     "video_id": video_name,
            #     "total_frames": len(frame_captions),
            #     "frame_captions": frame_captions,
            #     "combined_caption": " ".join([fc["caption"] for fc in frame_captions])
            # }
            combined_text = " ".join(frame_captions)
            video_record = {
                "vid": video_name,  # 注意：键名是"vid"不是"video_id"
                "text": combined_text  # 注意：键名是"text"不是"combined_caption"
            }

            # 写入JSONL文件
            f.write(json.dumps(video_record, ensure_ascii=False) + '\n')
            f.flush()  # 确保立即写入

            print(f"  完成 {video_name} 的描述生成\n")

    print(f"所有视频处理完成！结果已保存至: {output_jsonl}")


if __name__ == '__main__':
    input_folder = r"D:\code\LAB\MoRE2026\data\frames_32"
    output_jsonl = r"D:\code\LAB\MoRE2026\data\caption.jsonl"
    model_path = r"D:\models\blip\blip2-opt-2.7b"
    generate_video_captions(input_folder, output_jsonl, model_path)