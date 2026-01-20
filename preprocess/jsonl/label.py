import json

with open(r"D:\code\LAB\MoRE2026\data\vids\vids.csv", "r") as f:
    vids = [line.strip() for line in f if line.strip()]

# 生成label.jsonl
with open(r"D:\code\LAB\MoRE2026\data\label.jsonl", "w") as f:
    for vid in vids:
        if "non" in vid:
            label = 0
        else:
            label = 1
        f.write(json.dumps({"vid": vid, "label": label}) + "\n")

print(f"已生成 label.jsonl，包含 {len(vids)} 条记录")