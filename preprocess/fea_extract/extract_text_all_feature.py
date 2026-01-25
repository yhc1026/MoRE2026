# # 全模态
# from transformers import BertModel, BertTokenizer,AutoModel, AutoTokenizer
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# import os
# import numpy as np
# import torch
#
#
# dataset_dir = 'data'
# output_file = os.path.join(dataset_dir, 'fea/fea_transcript_bert-base-uncased.pt')
# model_id = "/root/autodl-tmp/MoRE/MoRE2026-Cloud/models/bert/bert-base-uncased"
# model = AutoModel.from_pretrained(model_id, device_map='cuda')
# processor = AutoTokenizer.from_pretrained(model_id)
#
#
# class MyDataset(Dataset):
#     def __init__(self):
#         vid_file = "data/vids/vids.csv"
#         # each line of vid_file a vid
#         with open(vid_file, 'r') as f:
#             self.vids = [line.strip() for line in f]
#         ocr_file = os.path.join(dataset_dir, 'ocr.jsonl')
#         trans_file = os.path.join(dataset_dir, 'speech.jsonl')
#         caption_file = os.path.join(dataset_dir, 'caption.jsonl')
#         #self.ocr_df = pd.read_json(ocr_file, lines=True)
#         self.trans_df = pd.read_json(trans_file, lines=True)
#         self.caption_df = pd.read_json(caption_file, lines=True)
#
#     def __len__(self):
#         return len(self.vids)
#
#     def __getitem__(self, index):
#         vid = self.vids[index]
#         ocr = self.ocr_df[self.ocr_df['vid'] == vid]['ocr'].values[0]
#         if len(ocr) > 0:
#             ocr = ocr['ocr'].values[0]
#             if pd.isna(ocr) or (isinstance(ocr, str) and ocr.strip() == ''):
#                 ocr = ''
#         else:
#             ocr = ''
#
#         trans = self.trans_df[self.trans_df['vid'] == vid]['transcript'].values[0]
#         if len(trans) > 0:
#             trans = trans['transcript'].values[0]
#             if pd.isna(trans) or (isinstance(trans, str) and trans.strip() == ''):
#                 trans = ''
#         else:
#             trans = ''
#
#         caption = self.caption_df[self.caption_df['vid'] == vid]['text'].values[0]
#         if len(caption) > 0:
#             caption = caption['text'].values[0]
#             if pd.isna(caption) or (isinstance(caption, str) and caption.strip() == ''):
#                 caption = ''
#         else:
#             caption = ''
#
#         text = f'{caption}\n{trans}\n{ocr}'
#         return vid, text
#
# def customed_collate_fn(batch):
#     # preprocess
#     # merge to one list
#     vids, texts = zip(*batch)
#     inputs = processor(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
#     return vids, inputs
#
# save_dict = {}
#
# dataloader = DataLoader(MyDataset(), batch_size=1, collate_fn=customed_collate_fn, num_workers=0, shuffle=True)
# model.eval()
# for batch in tqdm(dataloader):
#     with torch.no_grad():
#         vids, inputs = batch
#         inputs = inputs.to('cuda')
#         pooler_output = model(**inputs)['last_hidden_state'][:,0,:]
#         pooler_output = pooler_output.detach().cpu()
#         # process outputs
#         for i, vid in enumerate(vids):
#             save_dict[vid] = pooler_output[i]
# torch.save(save_dict, output_file)


# 全模态
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import json

dataset_dir = 'data'
output_file = os.path.join(dataset_dir, 'fea/fea_transcript_bert-base-uncased.pt')
model_id = "/root/autodl-tmp/MoRE/MoRE2026-Cloud/models/bert/bert-base-uncased"
model = AutoModel.from_pretrained(model_id, device_map='cuda')
processor = AutoTokenizer.from_pretrained(model_id)


class MyDataset(Dataset):
    def __init__(self):
        vid_file = "data/vids/vids.csv"
        # each line of vid_file a vid
        with open(vid_file, 'r') as f:
            self.vids = [line.strip() for line in f]
        ocr_file = os.path.join(dataset_dir, 'OCR.jsonl')
        trans_file = os.path.join(dataset_dir, 'speech.jsonl')
        caption_file = os.path.join(dataset_dir, 'caption.jsonl')

        # 处理OCR文件，修复空值
        ocr_data = []
        with open(ocr_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 修复ocr字段的空字符串
                        if 'ocr' in data and data['ocr'] == "":
                            data['ocr'] = ""
                        ocr_data.append(data)
                    except json.JSONDecodeError:
                        # 如果解析失败，创建空记录
                        ocr_data.append({'vid': '', 'ocr': ''})
        self.ocr_df = pd.DataFrame(ocr_data)

        # 处理transcript文件，修复空值
        trans_data = []
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 修复transcript字段的空字符串
                        if 'transcript' in data and data['transcript'] == "":
                            data['transcript'] = ""
                        trans_data.append(data)
                    except json.JSONDecodeError:
                        # 如果解析失败，创建空记录
                        trans_data.append({'vid': '', 'transcript': ''})
        self.trans_df = pd.DataFrame(trans_data)

        # 处理caption文件，修复空值
        caption_data = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 修复text字段的空字符串
                        if 'text' in data and data['text'] == "":
                            data['text'] = ""
                        caption_data.append(data)
                    except json.JSONDecodeError:
                        # 如果解析失败，创建空记录
                        caption_data.append({'vid': '', 'text': ''})
        self.caption_df = pd.DataFrame(caption_data)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]

        # 处理OCR
        ocr_rows = self.ocr_df[self.ocr_df['vid'] == vid]
        if len(ocr_rows) > 0:
            ocr = ocr_rows.iloc[0]['ocr']
            if pd.isna(ocr) or (isinstance(ocr, str) and ocr.strip() == ''):
                ocr = ''
        else:
            ocr = ''

        # 处理Transcript
        trans_rows = self.trans_df[self.trans_df['vid'] == vid]
        if len(trans_rows) > 0:
            trans = trans_rows.iloc[0]['transcript']
            if pd.isna(trans) or (isinstance(trans, str) and trans.strip() == ''):
                trans = ''
        else:
            trans = ''

        # 处理Caption
        caption_rows = self.caption_df[self.caption_df['vid'] == vid]
        if len(caption_rows) > 0:
            caption = caption_rows.iloc[0]['text']
            if pd.isna(caption) or (isinstance(caption, str) and caption.strip() == ''):
                caption = ''
        else:
            caption = ''

        text = f'{caption}\n{trans}\n{ocr}'
        return vid, text


def customed_collate_fn(batch):
    # preprocess
    # merge to one list
    vids, texts = zip(*batch)
    inputs = processor(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    return vids, inputs


save_dict = {}

dataloader = DataLoader(MyDataset(), batch_size=1, collate_fn=customed_collate_fn, num_workers=0, shuffle=True)
model.eval()
for batch in tqdm(dataloader):
    with torch.no_grad():
        vids, inputs = batch
        inputs = inputs.to('cuda')
        pooler_output = model(**inputs)['last_hidden_state'][:, 0, :]
        pooler_output = pooler_output.detach().cpu()
        # process outputs
        for i, vid in enumerate(vids):
            save_dict[vid] = pooler_output[i]
torch.save(save_dict, output_file)