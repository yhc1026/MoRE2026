import csv

# 生成视频名称列表
video_names = []

# hate_video_1 到 hate_video_431
for i in range(1, 432):
    video_names.append(f"hate_video_{i}")

# non_hate_video_1 到 non_hate_video_652
for i in range(1, 653):
    video_names.append(f"non_hate_video_{i}")

# 写入 CSV 文件，每行一个视频名称
with open('vids.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # 每行写入一个视频名称
    for video_name in video_names:
        writer.writerow([video_name])  # 注意：writerow 需要传入列表

print(f"✅ 已生成 CSV 文件：vids.csv")
print(f"📊 总行数：{len(video_names)} 行")
print(f"📋 内容预览（前5行和后5行）：")