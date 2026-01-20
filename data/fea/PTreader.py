import torch

# 加载整个模型（包含结构和参数）
model = torch.load(r"D:\code\LAB\MoRE2026\data\fea\fea_frames_16_google-vit-base-16-224.pt")
print(model)