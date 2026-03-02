import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from einops import rearrange
from .submodule import LearnablePositionalEncoding, check_shape, AttentivePooling
from .crossAttention import CrossModalAttention

class ModalityFFN(nn.Module):
    def __init__(self, hid_dim):
        super(ModalityFFN, self).__init__()
        self.modailty_ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))

    def forward(self, x, pos_x, neg_x):
        x = self.modailty_ffn(x)
        pos_x = self.modailty_ffn(pos_x)
        neg_x = self.modailty_ffn(neg_x)
        return x, pos_x, neg_x

class ModalityExpert(nn.Module):
    def __init__(self, hid_dim, dropout, num_head, alpha, ablation):
        super(ModalityExpert, self).__init__()
        self.ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))
        self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.alpha = alpha
        self.ablation = ablation

    def forward(self, query, pos, neg):
        print(f"query: {query.shape}, \n pos: {pos.shape}, \n neg: {neg.shape}")
        if query.dim() == 2:
            query = query.unsqueeze(1)
        query = self.ffn(query)
        pos = self.ffn(pos)
        neg = self.ffn(neg)
        if pos.dim() == 4:
            batch_size = pos.size(0)
            pos = pos.reshape(batch_size, -1, pos.size(-1))
            neg = neg.reshape(batch_size, -1, neg.size(-1))
        pos_attn, _ = self.pos_attn(query, pos, pos)
        neg_attn, _ = self.neg_attn(query, neg, neg)
        ret = self.alpha * pos_attn + (1 - self.alpha) * neg_attn + query
        return ret

from typing import Optional

class _TwoLayerAdapter(nn.Module):
    def __init__(self, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LazyLinear(out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        out = self.net(x)
        if x.size(-1) == out.size(-1):
            out = out + x
        return self.norm(out)

class _FeedForwardSimple(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.LazyLinear(dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.LazyLinear(dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        y = self.fc2(self.dropout(self.act(self.fc1(x))))
        return self.norm(x + self.dropout(y))

class TextExpert(nn.Module):
    def __init__(self, hid_dim, dropout, num_head, alpha):
        super().__init__()
        self.hid_dim = hid_dim
        self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout)
        self.shared_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hid_dim)
        self.ff = _FeedForwardSimple(hid_dim, dropout)
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def _ensure_seq(self, x):
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, query, pos, neg,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        query = self._ensure_seq(query)
        pos = self._ensure_seq(pos)
        neg = self._ensure_seq(neg)
        query = self.dim_adapter(query)
        pos = self.dim_adapter(pos)
        neg = self.dim_adapter(neg)
        pos_out, _ = self.shared_attn(query, pos, pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        neg_out, _ = self.shared_attn(query, neg, neg, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        mixed = self.alpha * pos_out + (1.0 - self.alpha) * neg_out
        x = self.attn_norm(self.dropout(mixed) + query)
        x = self.ff(x)
        return x

class VisionExpert(nn.Module):
    def __init__(self, hid_dim, dropout, num_head, alpha):
        super().__init__()
        self.hid_dim = hid_dim
        self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout)
        self.frame_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hid_dim)
        self.ff = _FeedForwardSimple(hid_dim, dropout)
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def _flatten_if_4d(self, x):
        if x.dim() == 4:
            b, n, t, c = x.size()
            return x.view(b, n * t, c)
        return x

    def forward(self, query, pos, neg,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        query = self.dim_adapter(query)
        q_att, _ = self.frame_attn(query, query, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        query = self.attn_norm(self.dropout(q_att) + query)
        pos = self._flatten_if_4d(pos)
        neg = self._flatten_if_4d(neg)
        pos = self.dim_adapter(pos)
        neg = self.dim_adapter(neg)
        pos_out, _ = self.frame_attn(query, pos, pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        neg_out, _ = self.frame_attn(query, neg, neg, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        mixed = self.alpha * pos_out + (1.0 - self.alpha) * neg_out
        x = self.attn_norm(self.dropout(mixed) + query)
        x = self.ff(x)
        return x

class AudioExpert(nn.Module):
    def __init__(self, hid_dim, dropout, num_head, alpha):
        super().__init__()
        self.hid_dim = hid_dim
        self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hid_dim)
        self.ff = _FeedForwardSimple(hid_dim, dropout)
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

    def _ensure_seq(self, x):
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, query, pos, neg,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        query = self._ensure_seq(query)
        query = self.dim_adapter(query)
        q_att, _ = self.temporal_attn(query, query, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        query = self.attn_norm(self.dropout(q_att) + query)
        pos = self._ensure_seq(pos)
        neg = self._ensure_seq(neg)
        pos = self.dim_adapter(pos)
        neg = self.dim_adapter(neg)
        pos_out, _ = self.temporal_attn(query, pos, pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        neg_out, _ = self.temporal_attn(query, neg, neg, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        mixed = self.alpha * pos_out + (1.0 - self.alpha) * neg_out
        x = self.attn_norm(self.dropout(mixed) + query)
        x = self.ff(x)
        return x

class MoRE(nn.Module):
    def __init__(self, text_encoder, fea_dim=768, dropout=0.2, num_head=8, alpha=0.5, delta=0.25, num_epoch=20, ablation='No', loss='No', **kargs):
        super(MoRE, self).__init__()

        # CrossAttention 开关（默认 True：保持当前行为/性能）
        self.use_cross_attention = bool(kargs.get('use_cross_attention', True))
        self.alpha_interact = float(kargs.get('alpha_interact', 0.05))

        self.bert = AutoModel.from_pretrained(text_encoder).requires_grad_(False)
        if hasattr(self.bert, 'text_model'):
            self.bert = self.bert.text_model

        self.text_ffn = nn.LazyLinear(fea_dim)
        self.vision_ffn = nn.LazyLinear(fea_dim)
        self.audio_ffn = nn.LazyLinear(fea_dim)

        self.text_expert = TextExpert(fea_dim, dropout, num_head, alpha)
        self.vision_expert = VisionExpert(fea_dim, dropout, num_head, alpha)
        self.audio_expert = AudioExpert(fea_dim, dropout, num_head, alpha)

        self.positional_encoding = LearnablePositionalEncoding(768, 16)

        self.cross_attn_vision = CrossModalAttention(fea_dim, num_heads=num_head, dropout=dropout)
        self.cross_attn_audio = CrossModalAttention(fea_dim, num_heads=num_head, dropout=dropout)
        self.cross_attn_text = CrossModalAttention(fea_dim, num_heads=num_head, dropout=dropout)

        self.text_pre_router = nn.LazyLinear(fea_dim)
        self.vision_pre_router = nn.LazyLinear(fea_dim)
        self.audio_pre_router = nn.LazyLinear(fea_dim)

        self.router_attn = CrossModalAttention(fea_dim, num_heads=num_head, dropout=dropout)

        self.router = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.ReLU(),
            nn.Linear(fea_dim, 1),
        )
        # 不使用 CrossAttention 时的 router（旧版 concat-router）
        self.router_no_attn = nn.Sequential(
            nn.LazyLinear(fea_dim),
            nn.ReLU(),
            nn.LazyLinear(3),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(nn.LazyLinear(200), nn.ReLU(), nn.Dropout(dropout), nn.Linear(200, 2))

        self.text_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
        self.vision_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
        self.audio_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))

        self.text_pooler = AttentivePooling(fea_dim)
        self.vision_pooler = AttentivePooling(fea_dim)

        self.delta = delta
        self.total_epoch = num_epoch
        self.ablation = ablation
        self.loss = loss

    def forward(self, **inputs):
        text_fea = inputs['text_fea']
        audio_fea = inputs['audio_fea']
        vision_fea = inputs['vision_fea']
        text_sim_pos_fea = inputs['text_sim_pos_fea']
        text_sim_neg_fea = inputs['text_sim_neg_fea']
        frame_sim_pos_fea = inputs['vision_sim_pos_fea']
        frame_sim_neg_fea = inputs['vision_sim_neg_fea']
        mfcc_sim_pos_fea = inputs['audio_sim_pos_fea']
        mfcc_sim_neg_fea = inputs['audio_sim_neg_fea']

        vision_fea = self.positional_encoding(vision_fea)
        frame_sim_pos_fea = self.positional_encoding(frame_sim_pos_fea)
        frame_sim_neg_fea = self.positional_encoding(frame_sim_neg_fea)

        text_fea_aug = self.text_expert(text_fea, text_sim_pos_fea, text_sim_neg_fea)
        vision_fea_aug = self.vision_expert(vision_fea, frame_sim_pos_fea, frame_sim_neg_fea)
        audio_fea_aug = self.audio_expert(audio_fea, mfcc_sim_pos_fea, mfcc_sim_neg_fea)

        if self.use_cross_attention:
            visual_context = self.cross_attn_vision(
                query=vision_fea_aug,
                key=text_fea_aug,
                value=text_fea_aug
            )
            vision_fea_aug = vision_fea_aug + self.alpha_interact * visual_context
        #
        # audio_context = self.cross_attn_audio(
        #     query=audio_fea_aug,
        #     key=vision_fea_aug,
        #     value=vision_fea_aug
        # )
        # audio_fea_aug = audio_fea_aug + 0.01 * audio_context
        #
        # text_context = self.cross_attn_text(
        #     query=text_fea_aug,
        #     key=vision_fea_aug,
        #     value=vision_fea_aug
        # )
        # text_fea_aug = text_fea_aug + 0.01 * text_context

        vision_fea = vision_fea.mean(dim=1, keepdim=True)
        if vision_fea.dim() == 3:
            vision_fea = vision_fea.squeeze(1)

        if self.use_cross_attention:
            text_router = self.text_pre_router(text_fea)
            vision_router = self.vision_pre_router(vision_fea)
            audio_router = self.audio_pre_router(audio_fea)

            modal_tokens = torch.stack([text_router, vision_router, audio_router], dim=1)
            router_ctx = self.router_attn(modal_tokens, modal_tokens, modal_tokens)
            logits = self.router(router_ctx).squeeze(-1)
            weight = F.softmax(logits, dim=-1)
        else:
            def _to_vec(x: torch.Tensor) -> torch.Tensor:
                if x.dim() == 4:
                    return x.mean(dim=(1, 2))
                if x.dim() == 3:
                    return x.mean(dim=1)
                return x

            router_fea = torch.cat([_to_vec(text_fea), _to_vec(vision_fea), _to_vec(audio_fea)], dim=-1)
            weight = self.router_no_attn(router_fea)

        text_fea_aug = text_fea_aug.mean(dim=1)
        vision_fea_aug = self.vision_pooler(vision_fea_aug)
        audio_fea_aug = audio_fea_aug.mean(dim=1)

        text_pred = self.text_preditor(text_fea_aug)
        vision_pred = self.vision_preditor(vision_fea_aug)
        audio_pred = self.audio_preditor(audio_fea_aug)

        fea = (text_fea_aug * weight[:, 0].unsqueeze(1) + vision_fea_aug * weight[:, 1].unsqueeze(1) + audio_fea_aug * weight[:, 2].unsqueeze(1))
        output = self.classifier(fea)

        return {
            'pred': output,
            'text_pred': text_pred,
            'vision_pred': vision_pred,
            'audio_pred': audio_pred,
            'weight': weight,
        }

    def calculate_loss(self, **inputs):
        delta = self.delta
        total_epoch = self.total_epoch

        pred = inputs['pred']
        label = inputs['label']
        text_pred = inputs['text_pred']
        vision_pred = inputs['vision_pred']
        audio_pred = inputs['audio_pred']
        cur_epoch = inputs['epoch']

        f_epo = (float(cur_epoch) / float(total_epoch)) ** 2

        l_mix = F.cross_entropy(pred, label)
        if text_pred is not None:
            text_loss = F.cross_entropy(text_pred, label)
        else:
            text_loss = 0.0
        if vision_pred is not None:
            vision_loss = F.cross_entropy(vision_pred, label)
        else:
            vision_loss = 0.0
        if audio_pred is not None:
            audio_loss = F.cross_entropy(audio_pred, label)
        else:
            audio_loss = 0.0

        l_exp = (text_loss + vision_loss + audio_loss) / 3

        l_join = min(1-f_epo, delta) * l_exp + max(f_epo, 1-delta) * l_mix

        return l_join, l_mix



# class _TwoLayerAdapter(nn.Module):
#     def __init__(self, out_dim, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LazyLinear(out_dim),    # 使用LazyLinear
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LazyLinear(out_dim),    # 使用LazyLinear
#         )
#         self.norm = nn.LayerNorm(out_dim)
#
#     def forward(self, x):
#         # 确保残差连接的维度匹配
#         out = self.net(x)
#         # 检查是否可以加残差（需要维度匹配）
#         if x.size(-1) == out.size(-1):
#             out = out + x
#         return self.norm(out)
#
# class _FeedForward(nn.Module):
#     def __init__(self, dim, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LazyLinear(dim * 4),    # 使用LazyLinear
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LazyLinear(dim),        # 使用LazyLinear
#             nn.Dropout(dropout),
#         )
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         return self.norm(x + self.net(x))
#
# class TextExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout=dropout)  # 不指定输入维度
#         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.attn_norm = nn.LayerNorm(hid_dim)
#         self.ff = _FeedForward(hid_dim, dropout=dropout)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, query, pos, neg):
#         if query.dim() == 2:
#             query = query.unsqueeze(1)
#
#         # LazyLinear会在第一次运行时自动确定输入维度
#         query = self.dim_adapter(query)
#         pos = self.dim_adapter(pos)
#         neg = self.dim_adapter(neg)
#
#         # attention blocks with residual & norm
#         pos_attn, _ = self.pos_attn(query, pos, pos)
#         neg_attn, _ = self.neg_attn(query, neg, neg)
#
#         mixed = self.alpha * pos_attn + (1 - self.alpha) * neg_attn
#         out = self.attn_norm(self.dropout(mixed) + query)
#         out = self.ff(out)
#
#         return out
#
# class VisionExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout=dropout)
#         self.pos_encoding = LearnablePositionalEncoding(hid_dim, max_len=32)
#         self.frame_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#
#         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#
#         self.attn_norm = nn.LayerNorm(hid_dim)
#         self.ff = _FeedForward(hid_dim, dropout=dropout)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, query, pos, neg):
#         # adapter (LazyLinear自动适配维度)
#         query = self.dim_adapter(query)
#         pos = self.dim_adapter(pos)
#         neg = self.dim_adapter(neg)
#
#         # positional encoding + frame-level self-attention
#         query = self.pos_encoding(query)
#         query_att, _ = self.frame_attention(query, query, query)
#         query = self.attn_norm(self.dropout(query_att) + query)
#
#         batch = query.size(0)
#         # support both (B, N, T, C) or (B, L, C) for pos/neg
#         if pos.dim() == 4:
#             b, num_pos, t, c = pos.size()
#             pos = pos.view(b, num_pos * t, c)
#         if neg.dim() == 4:
#             b, num_neg, t, c = neg.size()
#             neg = neg.view(b, num_neg * t, c)
#
#         pos_attn, _ = self.pos_attn(query, pos, pos)
#         neg_attn, _ = self.neg_attn(query, neg, neg)
#
#         mixed = self.alpha * pos_attn + (1 - self.alpha) * neg_attn
#         out = self.attn_norm(self.dropout(mixed) + query)
#         out = self.ff(out)
#         print(f"Vision out: {out.shape}")
#         return out
#
# class AudioExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha):
#         super().__init__()
#         self.hid_dim = hid_dim
#         # 总是使用TwoLayerAdapter，让LazyLinear自动处理维度
#         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout=dropout)
#         self.pos_encoding = LearnablePositionalEncoding(hid_dim, max_len=500)
#         self.temporal_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#
#         self.attn_norm = nn.LayerNorm(hid_dim)
#         self.ff = _FeedForward(hid_dim, dropout=dropout)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, query, pos, neg):
#         # adapter (LazyLinear自动适配维度)
#         query = self.dim_adapter(query)
#         pos = self.dim_adapter(pos)
#         neg = self.dim_adapter(neg)
#
#         if query.dim() == 2:
#             query = query.unsqueeze(1)
#
#         query = self.pos_encoding(query)
#         query_att, _ = self.temporal_attention(query, query, query)
#         query = self.attn_norm(self.dropout(query_att) + query)
#
#         pos_attn, _ = self.pos_attn(query, pos, pos)
#         neg_attn, _ = self.neg_attn(query, neg, neg)
#
#         mixed = self.alpha * pos_attn + (1 - self.alpha) * neg_attn
#         out = self.attn_norm(self.dropout(mixed) + query)
#         out = self.ff(out)
#         print(f"Audio out: {out.shape}")
#         return out


#没有加入loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from transformers import AutoModel
# from einops import rearrange
#
#
# from .submodule import LearnablePositionalEncoding, check_shape, AttentivePooling
#
# class ModalityFFN(nn.Module):
#     def __init__(self, hid_dim):
#         super(ModalityFFN, self).__init__()
#         self.modailty_ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))
#
#     def forward(self, x, pos_x, neg_x):
#         x = self.modailty_ffn(x)
#         pos_x = self.modailty_ffn(pos_x)
#         neg_x = self.modailty_ffn(neg_x)
#
#         return x, pos_x, neg_x
#
# class ModalityExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha, ablation):
#         super(ModalityExpert, self).__init__()
#         self.ffn = nn.Sequential(nn.LazyLinear(hid_dim), nn.ReLU(), nn.LazyLinear(hid_dim))
#         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.alpha = alpha
#         self.ablation = ablation
#
#     def forward(self, query, pos, neg):
#         print(f"query: {query.shape}, \n pos: {pos.shape}, \n neg: {neg.shape}")
#         if query.dim() == 2:
#             query = query.unsqueeze(1)
#         query = self.ffn(query)
#         pos = self.ffn(pos)
#         neg = self.ffn(neg)
#         if pos.dim() == 4:
#             batch_size = pos.size(0)
#             pos = pos.reshape(batch_size, -1, pos.size(-1))
#             neg = neg.reshape(batch_size, -1, neg.size(-1))
#         pos_attn, _ = self.pos_attn(query, pos, pos)
#         neg_attn, _ = self.neg_attn(query, neg, neg)
#         ret = self.alpha * pos_attn + (1 - self.alpha) * neg_attn + query
#         return ret
#
# from typing import Optional
#
# class _TwoLayerAdapter(nn.Module):
#     def __init__(self, out_dim, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LazyLinear(out_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LazyLinear(out_dim),
#         )
#         self.norm = nn.LayerNorm(out_dim)
#     def forward(self, x):
#         out = self.net(x)
#         if x.size(-1) == out.size(-1):
#             out = out + x
#         return self.norm(out)
#
# class _FeedForwardSimple(nn.Module):
#     def __init__(self, dim, dropout):
#         super().__init__()
#         self.fc1 = nn.LazyLinear(dim * 4)
#         self.act = nn.GELU()
#         self.fc2 = nn.LazyLinear(dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(dim)
#     def forward(self, x):
#         y = self.fc2(self.dropout(self.act(self.fc1(x))))
#         return self.norm(x + self.dropout(y))
#
# class TextExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout)
#         # 共享 attention，分别以 pos/neg 作为 key/value 输入
#         self.shared_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.attn_norm = nn.LayerNorm(hid_dim)
#         self.ff = _FeedForwardSimple(hid_dim, dropout)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#     def _ensure_seq(self, x):
#         return x.unsqueeze(1) if x.dim() == 2 else x
#
#     def forward(self, query, pos, neg,
#                 attn_mask: Optional[torch.Tensor] = None,
#                 key_padding_mask: Optional[torch.Tensor] = None):
#         query = self._ensure_seq(query)
#         pos = self._ensure_seq(pos)
#         neg = self._ensure_seq(neg)
#
#         # 统一适配一次
#         query = self.dim_adapter(query)
#         pos = self.dim_adapter(pos)
#         neg = self.dim_adapter(neg)
#
#         # 使用共享 attention：分别对 pos/neg 计算 attention 输出，再线性混合
#         pos_out, _ = self.shared_attn(query, pos, pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#         neg_out, _ = self.shared_attn(query, neg, neg, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#
#         mixed = self.alpha * pos_out + (1.0 - self.alpha) * neg_out
#         x = self.attn_norm(self.dropout(mixed) + query)
#         x = self.ff(x)
#         return x
#
# class VisionExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout)
#         self.frame_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.attn_norm = nn.LayerNorm(hid_dim)
#         self.ff = _FeedForwardSimple(hid_dim, dropout)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#     def _flatten_if_4d(self, x):
#         if x.dim() == 4:
#             b, n, t, c = x.size()
#             return x.view(b, n * t, c)
#         return x
#
#     def forward(self, query, pos, neg,
#                 attn_mask: Optional[torch.Tensor] = None,
#                 key_padding_mask: Optional[torch.Tensor] = None):
#         # query: (B, L, C) 或 (B, T, C)
#         query = self.dim_adapter(query)
#         # optional frame-level self-attention
#         q_att, _ = self.frame_attn(query, query, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#         query = self.attn_norm(self.dropout(q_att) + query)
#
#         pos = self._flatten_if_4d(pos); neg = self._flatten_if_4d(neg)
#         pos = self.dim_adapter(pos); neg = self.dim_adapter(neg)
#
#         pos_out, _ = self.frame_attn(query, pos, pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#         neg_out, _ = self.frame_attn(query, neg, neg, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#
#         mixed = self.alpha * pos_out + (1.0 - self.alpha) * neg_out
#         x = self.attn_norm(self.dropout(mixed) + query)
#         x = self.ff(x)
#         return x
#
# class AudioExpert(nn.Module):
#     def __init__(self, hid_dim, dropout, num_head, alpha):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout)
#         self.temporal_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
#         self.attn_norm = nn.LayerNorm(hid_dim)
#         self.ff = _FeedForwardSimple(hid_dim, dropout)
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#
#     def _ensure_seq(self, x):
#         return x.unsqueeze(1) if x.dim() == 2 else x
#
#     def forward(self, query, pos, neg,
#                 attn_mask: Optional[torch.Tensor] = None,
#                 key_padding_mask: Optional[torch.Tensor] = None):
#         query = self._ensure_seq(query)
#         query = self.dim_adapter(query)
#         q_att, _ = self.temporal_attn(query, query, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#         query = self.attn_norm(self.dropout(q_att) + query)
#
#         pos = self._ensure_seq(pos); neg = self._ensure_seq(neg)
#         pos = self.dim_adapter(pos); neg = self.dim_adapter(neg)
#
#         pos_out, _ = self.temporal_attn(query, pos, pos, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#         neg_out, _ = self.temporal_attn(query, neg, neg, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
#
#         mixed = self.alpha * pos_out + (1.0 - self.alpha) * neg_out
#         x = self.attn_norm(self.dropout(mixed) + query)
#         x = self.ff(x)
#         return x
#
# class MoRE(nn.Module):
#     def __init__(self, text_encoder, fea_dim=768, dropout=0.2, num_head=8, alpha=0.5, delta=0.25, num_epoch=20, ablation='No', loss='No', **kargs):
#         super(MoRE, self).__init__()
#
#         self.bert = AutoModel.from_pretrained(text_encoder).requires_grad_(False)
#         if hasattr(self.bert, 'text_model'):
#             self.bert = self.bert.text_model
#
#         self.text_ffn = nn.LazyLinear(fea_dim)
#         self.vision_ffn = nn.LazyLinear(fea_dim)
#         self.audio_ffn = nn.LazyLinear(fea_dim)
#
#         # self.text_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
#         # self.vision_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
#         # self.audio_expert = ModalityExpert(fea_dim, dropout, num_head, alpha, ablation)
#
#         self.text_expert = TextExpert(fea_dim, dropout, num_head, alpha)
#         self.vision_expert = VisionExpert(fea_dim, dropout, num_head, alpha)
#         self.audio_expert = AudioExpert(fea_dim, dropout, num_head, alpha)
#         #
#         self.positional_encoding = LearnablePositionalEncoding(768, 16)
#
#         self.text_pre_router = nn.LazyLinear(fea_dim)
#         self.vision_pre_router = nn.LazyLinear(fea_dim)
#         self.audio_pre_router = nn.LazyLinear(fea_dim)
#
#         self.router = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(3), nn.Softmax(dim=-1))
#
#         self.classifier = nn.Sequential(nn.LazyLinear(200), nn.ReLU(), nn.Dropout(dropout), nn.Linear(200, 2))
#
#         self.text_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
#         self.vision_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
#         self.audio_preditor = nn.Sequential(nn.LazyLinear(fea_dim), nn.ReLU(), nn.LazyLinear(2))
#
#         self.text_pooler = AttentivePooling(fea_dim)
#         self.vision_pooler = AttentivePooling(fea_dim)
#
#         self.delta = delta
#         self.total_epoch = num_epoch
#         self.ablation = ablation
#         self.loss = loss
#
#     def forward(self,  **inputs):
#
#         text_fea = inputs['text_fea']
#         audio_fea = inputs['audio_fea']
#         vision_fea = inputs['vision_fea']
#         text_sim_pos_fea = inputs['text_sim_pos_fea']
#         text_sim_neg_fea = inputs['text_sim_neg_fea']
#         frame_sim_pos_fea = inputs['vision_sim_pos_fea']
#         frame_sim_neg_fea = inputs['vision_sim_neg_fea']
#         mfcc_sim_pos_fea = inputs['audio_sim_pos_fea']
#         mfcc_sim_neg_fea = inputs['audio_sim_neg_fea']
#
#         vision_fea = self.positional_encoding(vision_fea)
#         frame_sim_pos_fea = self.positional_encoding(frame_sim_pos_fea)
#         frame_sim_neg_fea = self.positional_encoding(frame_sim_neg_fea)
#
#         text_fea_aug = self.text_expert(text_fea, text_sim_pos_fea, text_sim_neg_fea)
#         vision_fea_aug = self.vision_expert(vision_fea, frame_sim_pos_fea, frame_sim_neg_fea)
#         audio_fea_aug = self.audio_expert(audio_fea, mfcc_sim_pos_fea, mfcc_sim_neg_fea)
#
#         vision_fea = vision_fea.mean(dim=1, keepdim=True)
#         if vision_fea.dim() == 3:
#             vision_fea = vision_fea.squeeze(1)
#         router_fea = torch.cat([text_fea, vision_fea, audio_fea], dim=-1)
#         weight = self.router(router_fea).squeeze(1)
#
#         # text_fea_aug = self.text_pooler(text_fea_aug)
#         text_fea_aug = text_fea_aug.mean(dim=1)
#         vision_fea_aug = self.vision_pooler(vision_fea_aug)
#         audio_fea_aug = audio_fea_aug.mean(dim=1)
#
#         text_pred = self.text_preditor(text_fea_aug)
#         vision_pred = self.vision_preditor(vision_fea_aug)
#         audio_pred = self.audio_preditor(audio_fea_aug)
#
#
#         #消融实验
#         # if self.ablation == 'w/o-router':
#         #     fea = (text_fea_aug + vision_fea_aug + audio_fea_aug) / 3
#         # else:
#         #     fea = (text_fea_aug * weight[:, 0].unsqueeze(1) + vision_fea_aug * weight[:, 1].unsqueeze(1) + audio_fea_aug * weight[:, 2].unsqueeze(1))
#
#         fea = (text_fea_aug * weight[:, 0].unsqueeze(1) + vision_fea_aug * weight[:, 1].unsqueeze(1) + audio_fea_aug *weight[:, 2].unsqueeze(1))
#         output = self.classifier(fea)
#
#         return {
#             'pred': output,
#             'text_pred': text_pred,
#             'vision_pred': vision_pred,
#             'audio_pred': audio_pred,
#             'weight': weight,
#         }
#
#     def calculate_loss(self, **inputs):
#         delta = self.delta
#         total_epoch = self.total_epoch
#
#         pred = inputs['pred']
#         label = inputs['label']
#         text_pred = inputs['text_pred']
#         vision_pred = inputs['vision_pred']
#         audio_pred = inputs['audio_pred']
#         cur_epoch = inputs['epoch']
#
#         f_epo = (float(cur_epoch) / float(total_epoch)) ** 2
#
#         l_mix = F.cross_entropy(pred, label)
#         if text_pred is not None:
#             text_loss = F.cross_entropy(text_pred, label)
#         else:
#             text_loss = 0.0
#         if vision_pred is not None:
#             vision_loss = F.cross_entropy(vision_pred, label)
#         else:
#             vision_loss = 0.0
#         if audio_pred is not None:
#             audio_loss = F.cross_entropy(audio_pred, label)
#         else:
#             audio_loss = 0.0
#
#         l_exp = (text_loss + vision_loss + audio_loss) / 3
#
#         l_join = min(1-f_epo, delta) * l_exp + max(f_epo, 1-delta) * l_mix
#
#         return l_join, l_mix
#
#
# # class _TwoLayerAdapter(nn.Module):
# #     def __init__(self, out_dim, dropout):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.LazyLinear(out_dim),    # 使用LazyLinear
# #             nn.GELU(),
# #             nn.Dropout(dropout),
# #             nn.LazyLinear(out_dim),    # 使用LazyLinear
# #         )
# #         self.norm = nn.LayerNorm(out_dim)
# #
# #     def forward(self, x):
# #         # 确保残差连接的维度匹配
# #         out = self.net(x)
# #         # 检查是否可以加残差（需要维度匹配）
# #         if x.size(-1) == out.size(-1):
# #             out = out + x
# #         return self.norm(out)
# #
# # class _FeedForward(nn.Module):
# #     def __init__(self, dim, dropout):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.LazyLinear(dim * 4),    # 使用LazyLinear
# #             nn.GELU(),
# #             nn.Dropout(dropout),
# #             nn.LazyLinear(dim),        # 使用LazyLinear
# #             nn.Dropout(dropout),
# #         )
# #         self.norm = nn.LayerNorm(dim)
# #
# #     def forward(self, x):
# #         return self.norm(x + self.net(x))
# #
# # class TextExpert(nn.Module):
# #     def __init__(self, hid_dim, dropout, num_head, alpha):
# #         super().__init__()
# #         self.hid_dim = hid_dim
# #         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout=dropout)  # 不指定输入维度
# #         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #         self.attn_norm = nn.LayerNorm(hid_dim)
# #         self.ff = _FeedForward(hid_dim, dropout=dropout)
# #         self.alpha = alpha
# #         self.dropout = nn.Dropout(dropout)
# #
# #     def forward(self, query, pos, neg):
# #         if query.dim() == 2:
# #             query = query.unsqueeze(1)
# #
# #         # LazyLinear会在第一次运行时自动确定输入维度
# #         query = self.dim_adapter(query)
# #         pos = self.dim_adapter(pos)
# #         neg = self.dim_adapter(neg)
# #
# #         # attention blocks with residual & norm
# #         pos_attn, _ = self.pos_attn(query, pos, pos)
# #         neg_attn, _ = self.neg_attn(query, neg, neg)
# #
# #         mixed = self.alpha * pos_attn + (1 - self.alpha) * neg_attn
# #         out = self.attn_norm(self.dropout(mixed) + query)
# #         out = self.ff(out)
# #
# #         return out
# #
# # class VisionExpert(nn.Module):
# #     def __init__(self, hid_dim, dropout, num_head, alpha):
# #         super().__init__()
# #         self.hid_dim = hid_dim
# #         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout=dropout)
# #         self.pos_encoding = LearnablePositionalEncoding(hid_dim, max_len=32)
# #         self.frame_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #
# #         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #
# #         self.attn_norm = nn.LayerNorm(hid_dim)
# #         self.ff = _FeedForward(hid_dim, dropout=dropout)
# #         self.alpha = alpha
# #         self.dropout = nn.Dropout(dropout)
# #
# #     def forward(self, query, pos, neg):
# #         # adapter (LazyLinear自动适配维度)
# #         query = self.dim_adapter(query)
# #         pos = self.dim_adapter(pos)
# #         neg = self.dim_adapter(neg)
# #
# #         # positional encoding + frame-level self-attention
# #         query = self.pos_encoding(query)
# #         query_att, _ = self.frame_attention(query, query, query)
# #         query = self.attn_norm(self.dropout(query_att) + query)
# #
# #         batch = query.size(0)
# #         # support both (B, N, T, C) or (B, L, C) for pos/neg
# #         if pos.dim() == 4:
# #             b, num_pos, t, c = pos.size()
# #             pos = pos.view(b, num_pos * t, c)
# #         if neg.dim() == 4:
# #             b, num_neg, t, c = neg.size()
# #             neg = neg.view(b, num_neg * t, c)
# #
# #         pos_attn, _ = self.pos_attn(query, pos, pos)
# #         neg_attn, _ = self.neg_attn(query, neg, neg)
# #
# #         mixed = self.alpha * pos_attn + (1 - self.alpha) * neg_attn
# #         out = self.attn_norm(self.dropout(mixed) + query)
# #         out = self.ff(out)
# #         print(f"Vision out: {out.shape}")
# #         return out
# #
# # class AudioExpert(nn.Module):
# #     def __init__(self, hid_dim, dropout, num_head, alpha):
# #         super().__init__()
# #         self.hid_dim = hid_dim
# #         # 总是使用TwoLayerAdapter，让LazyLinear自动处理维度
# #         self.dim_adapter = _TwoLayerAdapter(hid_dim, dropout=dropout)
# #         self.pos_encoding = LearnablePositionalEncoding(hid_dim, max_len=500)
# #         self.temporal_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #         self.pos_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #         self.neg_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=num_head, dropout=dropout, batch_first=True)
# #
# #         self.attn_norm = nn.LayerNorm(hid_dim)
# #         self.ff = _FeedForward(hid_dim, dropout=dropout)
# #         self.alpha = alpha
# #         self.dropout = nn.Dropout(dropout)
# #
# #     def forward(self, query, pos, neg):
# #         # adapter (LazyLinear自动适配维度)
# #         query = self.dim_adapter(query)
# #         pos = self.dim_adapter(pos)
# #         neg = self.dim_adapter(neg)
# #
# #         if query.dim() == 2:
# #             query = query.unsqueeze(1)
# #
# #         query = self.pos_encoding(query)
# #         query_att, _ = self.temporal_attention(query, query, query)
# #         query = self.attn_norm(self.dropout(query_att) + query)
# #
# #         pos_attn, _ = self.pos_attn(query, pos, pos)
# #         neg_attn, _ = self.neg_attn(query, neg, neg)
# #
# #         mixed = self.alpha * pos_attn + (1 - self.alpha) * neg_attn
# #         out = self.attn_norm(self.dropout(mixed) + query)
# #         out = self.ff(out)
# #         print(f"Audio out: {out.shape}")
# #         return out