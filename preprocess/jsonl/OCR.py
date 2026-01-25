# import os
# import json
# import easyocr
#
# # ========== 配置 ==========
# dataset_dir = "data"
# frames_root_dir = "data/frames_16"  # 视频帧根目录
# output_jsonl = "data/OCR.jsonl"  # 输出文件
#
# # ========== 初始化OCR ==========
# reader = easyocr.Reader(['en'], gpu=True)  # 自动下载模型
#
#
# # ========== 处理函数 ==========
# def extract_ocr_from_video(video_folder):
#     """从视频文件夹提取OCR文本"""
#     ocr_texts = []
#
#     # 遍历视频文件夹中的所有帧文件
#     for file in os.listdir(video_folder):
#         if file.startswith('frame_') and file.endswith(('.jpg', '.png')):
#             frame_path = os.path.join(video_folder, file)
#
#             try:
#                 # 提取文本
#                 results = reader.readtext(frame_path, detail=0, paragraph=True)
#                 for text in results:
#                     if text.strip():
#                         ocr_texts.append(text.strip())
#             except:
#                 continue
#
#     # 去重合并
#     unique_texts = []
#     seen = set()
#     for text in ocr_texts:
#         if text not in seen:
#             seen.add(text)
#             unique_texts.append(text)
#
#     return " ".join(unique_texts)
#
#
# # ========== 主程序 ==========
# print("开始提取OCR文本...")
#
# with open(output_jsonl, 'w', encoding='utf-8') as f_out:
#     # 遍历frames根目录下的所有视频文件夹
#     for video_name in os.listdir(frames_root_dir):
#         video_path = os.path.join(frames_root_dir, video_name)
#
#         if os.path.isdir(video_path):  # 确保是文件夹
#             print(f"处理: {video_name}")
#
#             # 提取OCR文本
#             ocr_text = extract_ocr_from_video(video_path)
#
#             # 写入JSONL
#             json_data = {"vid": video_name, "ocr": ocr_text}
#             f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
#
# print(f"完成！结果保存至: {output_jsonl}")

# import json
# import easyocr
#
# # ========== 配置 ==========
# dataset_dir = r"D:\code\LAB\MoRE2026\data"
# frames_root_dir = r"D:\code\LAB\MoRE2026\data\frames_32"  # 视频帧根目录
# output_jsonl = r"D:\code\LAB\MoRE2026\data\ocr.jsonl"  # 输出文件
#
# # ========== 初始化OCR ==========
# reader = easyocr.Reader(['en'], gpu=True)  # 自动下载模型
#
#
# # ========== 处理函数 ==========
# def extract_ocr_from_video(video_folder):
#     """从视频文件夹提取OCR文本"""
#     ocr_texts = []
#
#     # 遍历视频文件夹中的所有帧文件
#     for file in os.listdir(video_folder):
#         if file.startswith('frame_') and file.endswith(('.jpg', '.png')):
#             frame_path = os.path.join(video_folder, file)
#
#             try:
#                 # 提取文本
#                 results = reader.readtext(frame_path, detail=0, paragraph=True)
#                 for text in results:
#                     if text.strip():
#                         ocr_texts.append(text.strip())
#             except:
#                 continue
#
#     # 去重合并
#     unique_texts = []
#     seen = set()
#     for text in ocr_texts:
#         if text not in seen:
#             seen.add(text)
#             unique_texts.append(text)
#
#     return " ".join(unique_texts)
#
#
# # ========== 主程序 ==========
# print("开始提取OCR文本...")
#
# with open(output_jsonl, 'w', encoding='utf-8') as f_out:
#     # 遍历frames根目录下的所有视频文件夹
#     for video_name in os.listdir(frames_root_dir):
#         video_path = os.path.join(frames_root_dir, video_name)
#
#         if os.path.isdir(video_path):  # 确保是文件夹
#             print(f"处理: {video_name}")
#
#             # 提取OCR文本
#             ocr_text = extract_ocr_from_video(video_path)
#
#             # 写入JSONL
#             json_data = {"vid": video_name, "ocr": ocr_text}
#             f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
#
# print(f"完成！结果保存至: {output_jsonl}")

import os
import json
import numpy as np
from decord import VideoReader, cpu
import easyocr
from tqdm import tqdm
import torch

# ========== 配置 ==========
video_dir = "data/videos"
output_jsonl = "data/OCR.jsonl"

# ========== 本地模型路径配置 ==========
# 请修改为你本地的 EasyOCR 模型路径
local_model_path = "/root/autodl-tmp/MoRE/MoRE2026-Cloud/models/easyocr_models"  # 修改为你的实际路径

# 检查目录是否存在
if not os.path.exists(local_model_path):
    print(f"⚠ 警告: 本地模型路径不存在: {local_model_path}")
    print("请修改 local_model_path 为正确的路径")
    # 尝试创建目录
    os.makedirs(local_model_path, exist_ok=True)
    print(f"已创建目录: {local_model_path}")

# 检查 CUDA 可用性
cuda_available = torch.cuda.is_available()
print("=" * 60)
print("🎬 视频 OCR 文本提取 (使用本地模型)")
print("=" * 60)
print(f"本地模型路径: {local_model_path}")
print(f"PyTorch CUDA 可用: {cuda_available}")
print("=" * 60)

# 初始化EasyOCR（使用本地模型）
try:
    # 使用本地模型路径，禁止下载
    reader = easyocr.Reader(
        lang_list=['en'],  # 只使用英语
        gpu=cuda_available,  # 根据CUDA可用性决定
        model_storage_directory=local_model_path,
        download_enabled=False,  # 禁止下载
        detector=True,
        recognizer=True,
        verbose=False  # 减少输出
    )
    print(f"✅ EasyOCR 初始化成功 ({'GPU' if cuda_available else 'CPU'}模式，使用本地模型)")
except Exception as e:
    print(f"❌ EasyOCR 初始化失败: {e}")
    print("尝试简化初始化...")
    try:
        reader = easyocr.Reader(['en'], gpu=cuda_available)
        print("✅ EasyOCR 简化初始化成功")
    except Exception as e2:
        print(f"❌ EasyOCR 完全失败: {e2}")
        raise

print("=" * 60)


def extract_ocr_from_video(video_path, target_frames=16):
    """提取视频中的OCR文本"""
    try:
        video_name = os.path.basename(video_path)
        print(f"📹 处理: {video_name}")

        # 始终使用 CPU 读取视频（更稳定）
        ctx = cpu(0)

        # 读取视频
        vr = VideoReader(video_path, ctx=ctx)
        total_frames = len(vr)

        # 优化采样策略：减少采样帧数以提高速度
        if total_frames < 10:  # 视频太短
            target_frames = min(total_frames, 4)
        elif total_frames < 30:
            target_frames = 8
        else:
            target_frames = 12  # 减少采样帧数

        # 选择采样帧
        if total_frames < target_frames:
            indices = list(range(total_frames))
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

        print(f"  总帧数: {total_frames}, 采样: {len(indices)}帧")

        ocr_texts = []
        processed_frames = 0

        # 逐帧处理
        for idx in indices:
            try:
                # 读取帧
                frame = vr[idx].asnumpy()
                # BGR to RGB
                frame_rgb = frame[:, :, ::-1]

                # OCR识别
                results = reader.readtext(
                    frame_rgb,
                    detail=0,  # 只返回文本
                    paragraph=True,  # 段落模式
                    width_ths=0.7,
                    text_threshold=0.3,
                    batch_size=4  # 批量处理
                )

                # 处理结果
                for text in results:
                    text_clean = text.strip()
                    if text_clean and len(text_clean) > 2:
                        ocr_texts.append(text_clean)

                processed_frames += 1

            except Exception as frame_error:
                print(f"  帧 {idx} 处理失败: {str(frame_error)[:50]}")
                continue

        print(f"  成功处理: {processed_frames}/{len(indices)} 帧")

        # 去重和合并
        if not ocr_texts:
            print(f"  未识别到文本")
            return ""

        # 去重（保持顺序）
        unique_texts = []
        seen = set()
        for text in ocr_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        print(f"  识别到 {len(unique_texts)} 条文本")

        # 合并文本，限制长度
        result = " ".join(unique_texts[:50])  # 限制最大文本数量
        if len(result) > 500:  # 限制字符数
            result = result[:500] + "..."

        return result

    except Exception as e:
        print(f"❌ 处理视频 {os.path.basename(video_path)} 时出错: {str(e)[:100]}")
        return ""


# ========== 主程序 ==========
def main():
    print(f"视频目录: {video_dir}")
    print(f"输出文件: {output_jsonl}")
    print("=" * 60)

    # 支持的视频格式
    video_extensions = ('.mp4')

    # 收集视频文件
    video_files = []
    for f in os.listdir(video_dir):
        if f.lower().endswith(video_extensions):
            video_files.append(f)

    if not video_files:
        print(f"❌ 错误: 在 {video_dir} 中未找到视频文件")
        return

    print(f"✅ 找到 {len(video_files)} 个视频文件")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # 处理视频
    processed_count = 0
    failed_count = 0

    # 进度条
    pbar = tqdm(video_files, desc="整体进度", unit="视频")

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for video_file in pbar:
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]

            # 更新进度条描述
            pbar.set_description(f"处理 {video_name[:20]}...")

            try:
                # 提取OCR文本
                ocr_text = extract_ocr_from_video(video_path)

                # 写入JSONL
                json_data = {"vid": video_name, "ocr": ocr_text}
                f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                f_out.flush()

                processed_count += 1

                # 显示处理结果预览
                if ocr_text:
                    pbar.set_postfix({
                        "状态": "成功",
                        "文本长度": len(ocr_text),
                        "进度": f"{processed_count}/{len(video_files)}"
                    })
                else:
                    pbar.set_postfix({
                        "状态": "无文本",
                        "进度": f"{processed_count}/{len(video_files)}"
                    })

            except Exception as e:
                pbar.set_postfix({"状态": "失败"})
                print(f"\n⚠ 处理 {video_file} 失败: {str(e)[:100]}")
                # 写入空记录
                json_data = {"vid": video_name, "ocr": ""}
                f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                failed_count += 1

    # 统计信息
    print("\n" + "=" * 60)
    print("🎉 处理完成!")
    print(f"✅ 成功: {processed_count} 个")
    print(f"❌ 失败: {failed_count} 个")
    print(f"📊 总计: {processed_count + failed_count} 个")
    print(f"📁 输出文件: {output_jsonl}")

    # 显示示例
    print("\n📋 前3条记录示例:")
    try:
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):
                data = json.loads(line.strip())
                ocr_preview = data['ocr'][:80] + "..." if len(data['ocr']) > 80 else data['ocr']
                print(f"  {i + 1}. {data['vid']}: {ocr_preview}")
    except Exception as e:
        print(f"读取示例失败: {e}")


if __name__ == "__main__":
    main()


# # 加入模糊匹配机制，词典
#
# import os
# import json
# import torch
# import pandas as pd
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModel, AutoTokenizer
#
# # ---------- Paths & model ----------
# dataset_dir = 'data'
# output_file = os.path.join(dataset_dir, 'fea/fea_transcript_bert-base-uncased.pt')
# model_id = "/root/autodl-tmp/MoRE/MoRE2026-Cloud/models/bert/bert-base-uncased"
#
# model = AutoModel.from_pretrained(model_id, device_map='cuda')
# processor = AutoTokenizer.from_pretrained(model_id)
#
# # ---------- Sensitive words (one per line, case-sensitive) ----------
# sensitive_file = os.path.join(dataset_dir, 'sensitive_words.txt')
# with open(sensitive_file, 'r', encoding='utf-8') as f:
#     sensitive_words = [line.strip() for line in f if line.strip()]
#
# # ---------- Levenshtein distance & fuzzy match ----------
# def levenshtein(a: str, b: str) -> int:
#     la, lb = len(a), len(b)
#     if la == 0: return lb
#     if lb == 0: return la
#     prev_row = list(range(lb + 1))
#     for i in range(1, la + 1):
#         cur_row = [i] + [0] * lb
#         for j in range(1, lb + 1):
#             cost = 0 if a[i-1] == b[j-1] else 1
#             cur_row[j] = min(prev_row[j] + 1, cur_row[j-1] + 1, prev_row[j-1] + cost)
#         prev_row = cur_row
#     return prev_row[lb]
#
# def fuzzy_in_text(sensitive: str, text: str, max_dist: int = 1) -> bool:
#     if not sensitive or not text:
#         return False
#     ls = len(sensitive)
#     min_l = max(1, ls - 1)
#     max_l = ls + 1
#     n = len(text)
#     for L in range(min_l, max_l + 1):
#         if L > n:
#             continue
#         for i in range(0, n - L + 1):
#             sub = text[i:i+L]
#             if levenshtein(sensitive, sub) <= max_dist:
#                 return True
#     return False
#
# # ---------- OCR processing: produce jsonl with {"vid":..., "ocr": "" or "word1 word2"} ----------
# ocr_file = os.path.join(dataset_dir, 'ocr.jsonl')
# ocr_out_file = os.path.join(dataset_dir, 'ocr_sensitive.jsonl')
#
# if os.path.exists(ocr_file):
#     out_lines = []
#     with open(ocr_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except Exception:
#                 continue
#             vid = obj.get('vid')
#             # adjust these extractions if your ocr.jsonl structure differs
#             text_field = obj.get('ocr') if 'ocr' in obj else obj.get('text', '')
#             if isinstance(text_field, list):
#                 text = ' '.join(str(x) for x in text_field)
#             elif isinstance(text_field, dict):
#                 text = ' '.join(str(v) for v in text_field.values())
#             else:
#                 text = str(text_field)
#
#             matched = []
#             for sw in sensitive_words:
#                 try:
#                     if fuzzy_in_text(sw, text, max_dist=1):
#                         matched.append(sw)
#                 except Exception:
#                     continue
#             ocr_val = "" if not matched else " ".join(matched)
#             out_lines.append({"vid": vid, "ocr": ocr_val})
#
#     with open(ocr_out_file, 'w', encoding='utf-8') as fout:
#         for rec in out_lines:
#             fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
# else:
#     # write nothing or create empty file with vids if desired
#     print(f"OCR file not found at {ocr_file}, skipping OCR-sensitive generation.")
#
# # ---------- Feature extraction for transcripts + captions (similar to your original) ----------
# class MyDataset(Dataset):
#     def __init__(self):
#         vid_file = "data/vids/vids.csv"
#         with open(vid_file, 'r', encoding='utf-8') as f:
#             self.vids = [line.strip() for line in f if line.strip()]
#         self.trans_df = pd.read_json(os.path.join(dataset_dir, 'speech.jsonl'), lines=True)
#         self.caption_df = pd.read_json(os.path.join(dataset_dir, 'caption.jsonl'), lines=True)
#
#     def __len__(self):
#         return len(self.vids)
#
#     def __getitem__(self, index):
#         vid = self.vids[index]
#         trans = ''
#         try:
#             trans_row = self.trans_df[self.trans_df['vid'] == vid]
#             if len(trans_row) > 0:
#                 trans = trans_row['transcript'].values[0]
#                 if isinstance(trans, dict) and 'transcript' in trans:
#                     trans = trans['transcript']
#                 if pd.isna(trans) or (isinstance(trans, str) and trans.strip() == ''):
#                     trans = ''
#         except Exception:
#             trans = ''
#
#         caption = ''
#         try:
#             cap_row = self.caption_df[self.caption_df['vid'] == vid]
#             if len(cap_row) > 0:
#                 caption = cap_row['text'].values[0]
#                 if isinstance(caption, dict) and 'text' in caption:
#                     caption = caption['text']
#                 if pd.isna(caption) or (isinstance(caption, str) and caption.strip() == ''):
#                     caption = ''
#         except Exception:
#             caption = ''
#
#         text = f'{caption}\n{trans}'
#         return vid, text
#
# def customed_collate_fn(batch):
#     vids, texts = zip(*batch)
#     inputs = processor(list(texts), padding='max_length', truncation=True, return_tensors='pt', max_length=512)
#     return vids, inputs
#
# save_dict = {}
# dataloader = DataLoader(MyDataset(), batch_size=1, collate_fn=customed_collate_fn, num_workers=0, shuffle=True)
#
# model.eval()
# for batch in tqdm(dataloader):
#     with torch.no_grad():
#         vids, inputs = batch
#         inputs = {k: v.to('cuda') for k, v in inputs.items()}
#         outputs = model(**inputs)
#         last_hidden = outputs['last_hidden_state'][:, 0, :]
#         pooler_output = last_hidden.detach().cpu()
#         for i, vid in enumerate(vids):
#             save_dict[vid] = pooler_output[i]
#
# torch.save(save_dict, output_file)
# print(f"Saved features to {output_file}")
