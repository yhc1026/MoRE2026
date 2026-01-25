#
# import os
# import torch
# import pandas as pd
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModel, AutoTokenizer
#
# def extract_bert_features(dataset_dir, model_id, output_file):
#     class MyDataset(Dataset):
#         def __init__(self, dataset_dir):
#             vid_file = "data/vids/vids.csv"
#             with open(vid_file, 'r') as f:
#                 self.vids = [line.strip() for line in f]
#             text_file = "data/ocr.jsonl"
#             self.text_df = pd.read_json(text_file, lines=True)
#
#         def __len__(self):
#             return len(self.vids)
#
#         def __getitem__(self, index):
#             vid = self.vids[index]
#             # text = self.text_df[self.text_df['vid'] == vid]['ocr'].values[0]
#             # return vid, text
#
#             vid_data = self.text_df[self.text_df['vid'] == vid]
#             if len(vid_data) == 0:
#                 text = ""
#             else:
#                 text = vid_data['ocr'].values[0]
#             return vid, text
#
#     def customed_collate_fn(batch):
#         vids, texts = zip(*batch)
#         inputs = processor(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
#         return vids, inputs
#
#     model = AutoModel.from_pretrained(model_id, device_map='cuda')
#     processor = AutoTokenizer.from_pretrained(model_id)
#
#     save_dict = {}
#
#     dataloader = DataLoader(MyDataset(dataset_dir), batch_size=1, collate_fn=customed_collate_fn, num_workers=0, shuffle=False)
#     model.eval()
#     for batch in tqdm(dataloader):
#         with torch.no_grad():
#             vids, inputs = batch
#             inputs = inputs.to('cuda')
#             pooler_output = model(**inputs)['last_hidden_state'][:,0,:]
#             pooler_output = pooler_output.detach().cpu()
#             for i, vid in enumerate(vids):
#                 save_dict[vid] = pooler_output[i]
#
#     torch.save(save_dict, output_file)
# # test
# # 使用示例:
# # extract_bert_features('data/MultiHateClip/zh', 'google-bert/bert-base-chinese', 'data/MultiHateClip/zh/fea/fea_ocr_bert-base-chinese.pt')
# # extract_bert_features('data/MultiHateClip/en', 'google-bert/bert-base-uncased', 'data/MultiHateClip/en/fea/fea_ocr_bert-base-uncased.pt')
# extract_bert_features("data", "/root/autodl-tmp/MoRE/MoRE2026-Cloud/models/bert/bert-base-uncased", "data/fea/fea_ocr_bert-base-uncased.pt")

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


def extract_bert_features(dataset_dir, model_id, output_file):
    class MyDataset(Dataset):
        def __init__(self, dataset_dir):
            # 读取视频ID
            vid_file = "data/vids/vids.csv"
            with open(vid_file, 'r') as f:
                self.vids = [line.strip() for line in f]

            # 读取OCR数据
            text_file = "data/OCR.jsonl"
            self.text_df = pd.read_json(text_file, lines=True)

        def __len__(self):
            return len(self.vids)

        def __getitem__(self, index):
            vid = self.vids[index]

            # 查找OCR文本
            vid_data = self.text_df[self.text_df['vid'] == vid]

            if len(vid_data) == 0:
                text = ""  # 没有OCR记录
            else:
                text = vid_data['ocr'].values[0]

                # 处理空OCR内容
                if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                    text = ""

            return vid, text

    def customed_collate_fn(batch):
        vids, texts = zip(*batch)

        # 处理空文本：用占位符
        processed_texts = []
        for text in texts:
            if text == "":
                processed_texts.append("[EMPTY_OCR]")
            else:
                processed_texts.append(text)

        # 编码文本
        inputs = processor(
            processed_texts,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=512
        )

        return vids, inputs

    # 加载模型
    model = AutoModel.from_pretrained(model_id, device_map='cuda')
    processor = AutoTokenizer.from_pretrained(model_id)

    # 创建数据加载器
    dataloader = DataLoader(
        MyDataset(dataset_dir),
        batch_size=1,
        collate_fn=customed_collate_fn,
        num_workers=0,
        shuffle=False
    )

    # 提取特征
    save_dict = {}
    model.eval()

    for batch in tqdm(dataloader):
        with torch.no_grad():
            vids, inputs = batch

            # 移动到设备
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # 获取特征
            pooler_output = model(**inputs)['last_hidden_state'][:, 0, :]
            pooler_output = pooler_output.detach().cpu()

            for i, vid in enumerate(vids):
                save_dict[vid] = pooler_output[i]

    # 保存特征
    torch.save(save_dict, output_file)


# 使用
extract_bert_features(
    "data",
    "/root/autodl-tmp/MoRE/MoRE2026-Cloud/models/bert/bert-base-uncased",
    "data/fea/fea_ocr_bert-base-uncased.pt"
)