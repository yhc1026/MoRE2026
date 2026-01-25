# import os
# import re
#
# import cv2
# import easyocr
# import numpy as np
# import pandas as pd
# import pytesseract
# from autocorrect import Speller
# from Levenshtein import ratio
# from skimage.metrics import structural_similarity as ssim
# from tqdm import tqdm
#
# spell = Speller(lang="en")
# reader = easyocr.Reader(["en"], gpu=True)
#
#
# def extract_frames(video_path, fps=1):
#     frames = []
#     video = cv2.VideoCapture(video_path)
#     video_fps = video.get(cv2.CAP_PROP_FPS)
#     interval = int(video_fps / fps)
#
#     frame_count = 0
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         if frame_count % interval == 0:
#             frames.append(frame)
#         frame_count += 1
#
#     video.release()
#     return frames
#
#
# def frame_similarity(frame1, frame2):
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     score, _ = ssim(gray1, gray2, full=True)
#     return score
#
#
# def remove_similar_frames(frames, threshold=0.95):
#     if not frames:
#         return []
#
#     unique_frames = [frames[0]]
#     for i in range(1, len(frames)):
#         if frame_similarity(frames[i], frames[i - 1]) < threshold:
#             unique_frames.append(frames[i])
#     return unique_frames
#
#
# def ocr_frames(frames):
#     texts = []
#     for frame in frames:
#         text = reader.readtext(frame, detail=0)
#         text = " ".join(text)
#         # if text is not blank str
#         if text:
#             texts.append(text.strip())
#     return texts
#
#
# def remove_duplicate_texts(texts, threshold=0.7):
#     if not texts:
#         return []
#
#     unique_texts = [texts[0]]
#     for i in range(1, len(texts)):
#         if ratio(texts[i], texts[i - 1]) < threshold:
#             unique_texts.append(texts[i])
#     return unique_texts
#
#
# def clean_and_correct_text(text):
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     text = re.sub(r"\s+", " ", text)
#
#     return text.strip()
#
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         if word.isupper():
#             corrected_words.append(word)
#         else:
#             corrected_words.append(spell(word))
#
#     corrected_text = " ".join(corrected_words)
#
#     return corrected_text.strip()
#
#
# def extract_text_from_video(video_path):
#     frames = extract_frames(video_path)
#     # print(f'Extracted {len(frames)} frames')
#
#     unique_frames = remove_similar_frames(frames)
#     # print(f'Removed {len(frames) - len(unique_frames)} similar frames')
#
#     texts = ocr_frames(unique_frames)
#     # print text
#     # for text in texts:
#     #     print(f'{text}')
#
#     cleaned_texts = [clean_and_correct_text(text) for text in texts]
#
#     # for text in cleaned_texts:
#     #     print(f'{text}')
#
#     unique_texts = remove_duplicate_texts(cleaned_texts)
#
#     return unique_texts
#
#
# src_dir = r"D:\code\LAB\MoREBaseline\MoRE\data\HateMM\videos\non_hate_videos"
# dst_file = r"D:\code\LAB\MoREBaseline\MoRE\data\HateMM\OCRs\OCR.jsonl"
#
# if not os.path.exists(dst_file):
#     dst_df = pd.DataFrame(columns=["vid", "ocr"])
#     dst_df.to_json(dst_file, orient="records", lines=True)
# else:
#     dst_df = pd.read_json(dst_file, lines=True)
#
# cur_ids = dst_df["vid"].values if len(dst_df) > 0 else []
#
# for file in tqdm(os.listdir(src_dir)):
#     audio_file = os.path.join(src_dir, file)
#     audio_id = file.replace(".mp4", "")
#
#     if audio_id in cur_ids:
#         continue
#
#     ocr = ""
#     result = extract_text_from_video(audio_file)
#     for text in result:
#         if len(text) > 3:
#             ocr += text + "\n"
#
#     # caption = image_to_caption(video_frames)
#     tmp_df = pd.DataFrame([{"vid": audio_id, "ocr": ocr}])
#     dst_df = pd.concat([dst_df, tmp_df], ignore_index=True)
#     dst_df.to_json(dst_file, orient="records", lines=True, force_ascii=False)

import os
import re
import cv2
import easyocr
import numpy as np
import pandas as pd
from Levenshtein import ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import json
from functools import lru_cache
import hashlib


class VideoTextExtractor:
    def __init__(self, gpu=True, batch_size=4, num_workers=None):
        """
        初始化文本提取器

        Args:
            gpu: 是否使用GPU
            batch_size: 批量处理帧的数量
            num_workers: 并行工作线程数
        """
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        self.batch_size = batch_size

        # 自动设置工作线程数
        if num_workers is None:
            self.num_workers = min(multiprocessing.cpu_count(), 8)
        else:
            self.num_workers = num_workers

        print(f"初始化完成: GPU={gpu}, 批大小={batch_size}, 工作线程={self.num_workers}")

    @staticmethod
    def extract_frames_efficient(video_path, target_fps=1):
        """
        高效提取视频帧

        Args:
            video_path: 视频文件路径
            target_fps: 目标帧率（每秒提取多少帧）
        """
        frames = []
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            return frames

        video_fps = video.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # 默认值

        # 计算间隔
        interval = max(1, int(video_fps / target_fps))

        # 使用预分配内存
        frame_count = 0
        success = True

        while success:
            success, frame = video.read()
            if not success:
                break

            if frame_count % interval == 0:
                frames.append(frame)

            frame_count += 1

            # 限制最多处理3000帧（防止超长视频）
            if frame_count > 3000:
                break

        video.release()

        # 如果帧太多，均匀采样
        if len(frames) > 100:
            step = len(frames) // 50
            frames = frames[::step][:50]

        return frames

    @staticmethod
    def compute_frame_hash(frame):
        """计算帧的哈希值用于去重"""
        # 缩小图像以加速哈希计算
        small = cv2.resize(frame, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray.tobytes()).hexdigest()

    def remove_similar_frames_fast(self, frames, similarity_threshold=0.9):
        """
        快速去除相似帧

        Args:
            frames: 帧列表
            similarity_threshold: 相似度阈值
        """
        if not frames or len(frames) < 2:
            return frames

        # 方法1：使用哈希去重（快速）
        hashes = {}
        unique_frames_hash = []

        for frame in frames:
            frame_hash = self.compute_frame_hash(frame)
            if frame_hash not in hashes:
                hashes[frame_hash] = True
                unique_frames_hash.append(frame)

        # 如果哈希去重后仍然太多，使用SSIM进一步去重
        if len(unique_frames_hash) > 30:
            return self.remove_similar_frames_ssim(unique_frames_hash, similarity_threshold)

        return unique_frames_hash

    @staticmethod
    def remove_similar_frames_ssim(frames, threshold=0.9):
        """使用SSIM去除相似帧"""
        if not frames:
            return frames

        unique_frames = [frames[0]]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # 使用下采样加速SSIM计算
            if prev_gray.shape != (64, 64):
                prev_gray_small = cv2.resize(prev_gray, (64, 64))
                gray_small = cv2.resize(gray, (64, 64))
            else:
                prev_gray_small = prev_gray
                gray_small = gray

            score = ssim(prev_gray_small, gray_small)

            if score < threshold:
                unique_frames.append(frames[i])
                prev_gray = gray

        return unique_frames

    def batch_ocr_frames(self, frames):
        """
        批量OCR处理帧

        Args:
            frames: 帧列表
        """
        if not frames:
            return []

        texts = []

        # 批量处理
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]

            # 并行处理批次中的帧
            with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                future_to_frame = {
                    executor.submit(self._single_frame_ocr, frame): idx
                    for idx, frame in enumerate(batch)
                }

                batch_results = []
                for future in as_completed(future_to_frame):
                    try:
                        result = future.result(timeout=10)
                        if result:
                            batch_results.append(result)
                    except Exception as e:
                        print(f"OCR处理错误: {e}")
                        continue

                texts.extend(batch_results)

        return texts

    def _single_frame_ocr(self, frame):
        """单帧OCR处理"""
        try:
            results = self.reader.readtext(frame, detail=0, paragraph=True)
            if results:
                text = " ".join(results).strip()
                if len(text) > 2:  # 过滤太短的文本
                    return text
        except Exception as e:
            print(f"帧OCR错误: {e}")
        return None

    @staticmethod
    def clean_text_batch(texts):
        """批量清理文本"""
        cleaned = []
        for text in texts:
            if not text:
                continue

            # 移除特殊字符，保留字母、数字和空格
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            # 合并多个空格
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > 2:
                cleaned.append(text)

        return cleaned

    @staticmethod
    def remove_duplicate_texts_fast(texts, threshold=0.7):
        """快速去除重复文本"""
        if not texts:
            return []

        unique_texts = []
        seen_hashes = set()

        for text in texts:
            if not text:
                continue

            # 计算文本的简单哈希（前50个字符）
            text_hash = text[:50].lower()

            # 检查相似度
            is_duplicate = False
            for seen_text in unique_texts[-10:]:  # 只检查最近的10个
                if ratio(text.lower(), seen_text.lower()) > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate and text_hash not in seen_hashes:
                unique_texts.append(text)
                seen_hashes.add(text_hash)

        return unique_texts

    def extract_text_from_video(self, video_path):
        """
        从视频中提取文本的主函数

        Args:
            video_path: 视频文件路径
        """
        try:
            # 1. 提取帧
            frames = self.extract_frames_efficient(video_path, target_fps=1)
            if not frames:
                return []

            # 2. 去除相似帧
            unique_frames = self.remove_similar_frames_fast(frames)

            # 3. 批量OCR
            texts = self.batch_ocr_frames(unique_frames)

            # 4. 清理文本
            cleaned_texts = self.clean_text_batch(texts)

            # 5. 去重
            final_texts = self.remove_duplicate_texts_fast(cleaned_texts)

            return final_texts

        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {e}")
            return []


def process_single_video(args):
    """处理单个视频（用于并行处理）"""
    video_path, extractor, dst_df, dst_file = args
    file_name = os.path.basename(video_path)
    video_id = os.path.splitext(file_name)[0]

    # 检查是否已处理
    if not dst_df.empty and video_id in dst_df["vid"].values:
        return video_id, "已跳过", None

    try:
        # 提取文本
        texts = extractor.extract_text_from_video(video_path)

        # 合并文本
        ocr_text = "\n".join([t for t in texts if len(t) > 3])

        return video_id, "成功", ocr_text

    except Exception as e:
        print(f"处理视频 {video_id} 失败: {e}")
        return video_id, "失败", None


def main():
    # 配置参数
    src_dir = r"D:\code\LAB\MoRE2026\data\videos"
    dst_file = r"D:\code\LAB\MoRE2026\data\OCR.jsonl"

    # 创建输出目录
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    # 初始化提取器（使用GPU加速）
    extractor = VideoTextExtractor(gpu=True, batch_size=8, num_workers=2)

    # 加载或创建数据框
    if os.path.exists(dst_file):
        try:
            dst_df = pd.read_json(dst_file, lines=True)
            print(f"已加载 {len(dst_df)} 条现有记录")
        except:
            dst_df = pd.DataFrame(columns=["vid", "ocr"])
    else:
        dst_df = pd.DataFrame(columns=["vid", "ocr"])

    # 获取待处理的视频文件
    video_files = []
    for file in os.listdir(src_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(src_dir, file)
            video_files.append(video_path)

    print(f"找到 {len(video_files)} 个视频文件")

    # 准备待处理任务（过滤已处理的）
    tasks = []
    processed_ids = set(dst_df["vid"].values) if not dst_df.empty else set()

    for video_path in video_files:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if video_id not in processed_ids:
            tasks.append((video_path, extractor, dst_df, dst_file))

    print(f"需要处理 {len(tasks)} 个新视频")

    if not tasks:
        print("所有视频已处理完成！")
        return

    # 并行处理视频
    results = []
    with ProcessPoolExecutor(max_workers=extractor.num_workers) as executor:
        futures = {executor.submit(process_single_video, task): task[0]
                   for task in tasks}

        with tqdm(total=len(tasks), desc="处理视频") as pbar:
            for future in as_completed(futures):
                try:
                    video_id, status, ocr_text = future.result(timeout=300)  # 5分钟超时
                    results.append((video_id, status, ocr_text))

                    # 更新进度条
                    pbar.set_postfix({
                        "成功": len([r for r in results if r[1] == "成功"]),
                        "失败": len([r for r in results if r[1] == "失败"]),
                        "跳过": len([r for r in results if r[1] == "已跳过"])
                    })
                    pbar.update(1)

                except Exception as e:
                    print(f"任务执行失败: {e}")
                    pbar.update(1)

    # 保存成功的结果
    new_records = []
    for video_id, status, ocr_text in results:
        if status == "成功" and ocr_text:
            new_records.append({"vid": video_id, "ocr": ocr_text})

    if new_records:
        new_df = pd.DataFrame(new_records)
        dst_df = pd.concat([dst_df, new_df], ignore_index=True)

        # 保存到文件
        dst_df.to_json(dst_file, orient="records", lines=True, force_ascii=False)
        print(f"\n成功处理 {len(new_records)} 个视频，已保存到 {dst_file}")

    # 输出统计信息
    success_count = len([r for r in results if r[1] == "成功"])
    fail_count = len([r for r in results if r[1] == "失败"])
    skip_count = len([r for r in results if r[1] == "已跳过"])

    print("\n" + "=" * 50)
    print(f"处理完成统计:")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {fail_count}")
    print(f"  ⏭️  跳过: {skip_count}")
    print(f"  📊 总计: {len(results)}")


if __name__ == "__main__":
    main()
