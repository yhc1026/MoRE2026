import os
import json

# ---------- 路径配置 ----------
dataset_dir = 'data'
ocr_file = os.path.join(dataset_dir, 'OCR_origin.jsonl')
ocr_out_file = os.path.join(dataset_dir, 'OCR.jsonl')
sensitive_file = os.path.join(dataset_dir, 'sensitive_words.txt')

# ---------- 加载敏感词 ----------
with open(sensitive_file, 'r', encoding='utf-8') as f:
    sensitive_words = [line.strip() for line in f if line.strip()]


# ---------- 编辑距离计算 ----------
def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev_row = list(range(lb + 1))
    for i in range(1, la + 1):
        cur_row = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur_row[j] = min(prev_row[j] + 1, cur_row[j - 1] + 1, prev_row[j - 1] + cost)
        prev_row = cur_row
    return prev_row[lb]


# ---------- 模糊匹配 ----------
def fuzzy_in_text(sensitive: str, text: str, max_dist: int = 1) -> bool:
    if not sensitive or not text:
        return False
    ls = len(sensitive)
    min_l = max(1, ls - 1)
    max_l = ls + 1
    n = len(text)
    for L in range(min_l, max_l + 1):
        if L > n:
            continue
        for i in range(0, n - L + 1):
            sub = text[i:i + L]
            if levenshtein(sensitive, sub) <= max_dist:
                return True
    return False


# ---------- OCR处理 ----------
out_lines = []
with open(ocr_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        vid = obj.get('vid')
        text_field = obj.get('ocr') if 'ocr' in obj else obj.get('text', '')

        if isinstance(text_field, list):
            text = ' '.join(str(x) for x in text_field)
        elif isinstance(text_field, dict):
            text = ' '.join(str(v) for v in text_field.values())
        else:
            text = str(text_field)

        matched = []
        for sw in sensitive_words:
            try:
                if fuzzy_in_text(sw, text, max_dist=1):
                    matched.append(sw)
            except Exception:
                continue

        ocr_val = "" if not matched else " ".join(matched)
        out_lines.append({"vid": vid, "ocr": ocr_val})

# ---------- 保存结果 ----------
with open(ocr_out_file, 'w', encoding='utf-8') as fout:
    for rec in out_lines:
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')