# 随机生成训练集，测试集，验证集

import random

# 读取vids.csv文件
with open(r"D:\code\LAB\MoRE2026\data\vids\vids.csv", "r") as f:
    vids = [line.strip() for line in f if line.strip()]

print(f"总视频数量: {len(vids)}")

# 随机打乱视频列表
random.shuffle(vids)

# 计算划分数量
total = len(vids)
test_count = int(total * 0.1)    # 10% for test
valid_count = int(total * 0.3)   # 30% for valid
train_count = total - test_count - valid_count  # 剩下的60% for train

print(f"训练集数量: {train_count}")
print(f"验证集数量: {valid_count}")
print(f"测试集数量: {test_count}")

# 划分数据集
test_vids = vids[:test_count]
valid_vids = vids[test_count:test_count + valid_count]
train_vids = vids[test_count + valid_count:]

# 保存到CSV文件
output_dir = r"D:\code\LAB\MoRE2026\data\vids"

# 保存训练集
with open(f"{output_dir}/train.csv", "w") as f:
    f.write("\n".join(train_vids))

# 保存验证集
with open(f"{output_dir}/valid.csv", "w") as f:
    f.write("\n".join(valid_vids))

# 保存测试集
with open(f"{output_dir}/test.csv", "w") as f:
    f.write("\n".join(test_vids))

print(f"\n文件已保存到: {output_dir}")
print(f"train.csv: {len(train_vids)} 个视频")
print(f"valid.csv: {len(valid_vids)} 个视频")
print(f"test.csv: {len(test_vids)} 个视频")

# 简单检查
all_assigned = train_vids + valid_vids + test_vids
print(f"检查: 分配了 {len(set(all_assigned))} 个唯一视频")
print(f"所有视频都被分配: {len(set(all_assigned)) == len(vids)}")