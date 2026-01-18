# import os
#
# import numpy as np
# import torch
# from PIL import Image
# from regex import F
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# from transformers import AutoModel, AutoProcessor
#
# dataset_dir = r"D:\code\LAB\MoREBaseline\MoRE\data\HateMM"
# frames_path = r"\frames"
# output_file = r"D:\code\LAB\MoREBaseline\MoRE\data\HateMM\fea"
# model_id = r"D:\models\vit-base-patch16-224"
# model = AutoModel.from_pretrained(model_id).to("cuda")
# processor = AutoProcessor.from_pretrained(model_id)
#
#
# class MyDataset(Dataset):
#     def __init__(self):
#         vid_file = r"D:\code\LAB\MoREBaseline\MoRE\data\HateMM\vids\vids.csv"
#         with open(vid_file, "r") as f:
#             self.vids = [line.strip() for line in f]
#
#     def __len__(self):
#         return len(self.vids)
#
#     def __getitem__(self, idx):
#         vid = self.vids[idx]
#         frames = []
#         for i in range(16):
#             frame_path = os.path.join(dataset_dir, frames_path, f"{vid}", f"frame_{i:03d}.jpg")
#             if os.path.exists(frame_path):
#                 frame = Image.open(frame_path).convert("RGB")
#                 frames.append(frame)
#             else:
#                 # If the frame does not exist, use a black image as a placeholder
#                 # raise ValueError(f"Frame {frame_path} not found")
#                 frames.append(Image.new("RGB", (224, 224), color="black"))
#         return vid, frames
#
#
# def collate_fn(batch):
#     vids, all_frames = zip(*batch)
#     all_frames = [frame for frames in all_frames for frame in frames]
#     processed_frames = processor(images=all_frames, return_tensors="pt", padding=True)
#     return vids, processed_frames
#
#
# dataset = MyDataset()
# dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)
# features = {}
#
# with torch.no_grad():
#     for vids, processed_frames in tqdm(dataloader):
#         bs = len(vids)
#         inputs = {k: v.to(model.device) for k, v in processed_frames.items()}
#         outputs = model(**inputs)
#         hidden_stats = outputs.last_hidden_state.view(bs, 16, -1, outputs.last_hidden_state.size(-1))
#         for i, vid in enumerate(vids):
#             features[vid] = (
#                 hidden_stats[i][
#                     :,
#                     0,
#                 ]
#                 .detach()
#                 .cpu()
#             )
#             # print(f"Video {vid}: Feature shape {features[vid].shape}")
#
#
# torch.save(features, output_file)


import os

import numpy as np
import torch
from PIL import Image
from regex import F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# 保留你原有的所有常量定义
dataset_dir = r"D:\code\LAB\MoRE2026\data"
frames_path = r"D:\code\LAB\MoRE2026\data\frames_32"
output_file = r"D:\code\LAB\MoRE2026\data\fea\fea_frames_16_google-vit-base-16-224.pt"
model_id = r"D:\models\vit-base-patch16-224"

# 保留你原有的 Dataset 类定义
class MyDataset(Dataset):
    def __init__(self):
        vid_file = r"D:\code\LAB\MoRE2026\data\vids\vids.csv"
        with open(vid_file, "r") as f:
            self.vids = [line.strip() for line in f]

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        frames = []
        for i in range(16):
            frame_path = os.path.join(dataset_dir, frames_path, f"{vid}", f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert("RGB")
                frames.append(frame)
            else:
                # If the frame does not exist, use a black image as a placeholder
                # raise ValueError(f"Frame {frame_path} not found")
                frames.append(Image.new("RGB", (224, 224), color="black"))
        return vid, frames

# 保留你原有的 collate_fn 定义
def collate_fn(batch):
    vids, all_frames = zip(*batch)
    all_frames = [frame for frames in all_frames for frame in frames]
    processed_frames = processor(images=all_frames, return_tensors="pt", padding=True)
    return vids, processed_frames

# ========== 关键修改1：将主逻辑包裹在 if __name__ == '__main__' 中 ==========
if __name__ == '__main__':
    # 加载模型和处理器（保留你之前改好的 to("cuda") 写法）
    model = AutoModel.from_pretrained(model_id).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)

    # ========== 关键修改2：Windows 下将 num_workers 改为 0（禁用多进程） ==========
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    features = {}

    with torch.no_grad():
        for vids, processed_frames in tqdm(dataloader):
            bs = len(vids)
            inputs = {k: v.to(model.device) for k, v in processed_frames.items()}
            outputs = model(** inputs)
            hidden_stats = outputs.last_hidden_state.view(bs, 16, -1, outputs.last_hidden_state.size(-1))
            for i, vid in enumerate(vids):
                features[vid] = (
                    hidden_stats[i][
                        :,
                        0,
                    ]
                    .detach()
                    .cpu()
                )
                # print(f"Video {vid}: Feature shape {features[vid].shape}")

    torch.save(features, output_file)