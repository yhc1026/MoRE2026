import os
import json
import easyocr

# ========== 配置 ==========
dataset_dir = r"D:\code\LAB\MoRE2026\data"
frames_root_dir = r"D:\code\LAB\MoRE2026\data\frames_32"  # 视频帧根目录
output_jsonl = r"D:\code\LAB\MoRE2026\data\ocr.jsonl"  # 输出文件

# ========== 初始化OCR ==========
reader = easyocr.Reader(['en'], gpu=True)  # 自动下载模型


# ========== 处理函数 ==========
def extract_ocr_from_video(video_folder):
    """从视频文件夹提取OCR文本"""
    ocr_texts = []

    # 遍历视频文件夹中的所有帧文件
    for file in os.listdir(video_folder):
        if file.startswith('frame_') and file.endswith(('.jpg', '.png')):
            frame_path = os.path.join(video_folder, file)

            try:
                # 提取文本
                results = reader.readtext(frame_path, detail=0, paragraph=True)
                for text in results:
                    if text.strip():
                        ocr_texts.append(text.strip())
            except:
                continue

    # 去重合并
    unique_texts = []
    seen = set()
    for text in ocr_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return " ".join(unique_texts)


# ========== 主程序 ==========
print("开始提取OCR文本...")

with open(output_jsonl, 'w', encoding='utf-8') as f_out:
    # 遍历frames根目录下的所有视频文件夹
    for video_name in os.listdir(frames_root_dir):
        video_path = os.path.join(frames_root_dir, video_name)

        if os.path.isdir(video_path):  # 确保是文件夹
            print(f"处理: {video_name}")

            # 提取OCR文本
            ocr_text = extract_ocr_from_video(video_path)

            # 写入JSONL
            json_data = {"vid": video_name, "ocr": ocr_text}
            f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")

print(f"完成！结果保存至: {output_jsonl}")