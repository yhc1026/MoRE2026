# fix_model_config.py
import os
import json


def fix_blip2_config(model_path):
    """
    修复BLIP2模型的配置文件
    """
    print(f"修复模型配置: {model_path}")

    # 1. 修复 processor_config.json
    processor_config = os.path.join(model_path, "processor_config.json")
    if os.path.exists(processor_config):
        with open(processor_config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 移除不正确的参数
        keys_to_remove = ['num_query_tokens', 'qformer_config']
        for key in keys_to_remove:
            if key in config:
                print(f"  移除 {key}")
                del config[key]

        # 确保有正确的配置
        if 'processor_class' not in config:
            config['processor_class'] = 'Blip2Processor'

        # 保存
        with open(processor_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("  ✓ 修复 processor_config.json")

    # 2. 修复 config.json（如果需要）
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 检查是否是BLIP2配置
        if 'architectures' in config and 'Blip2ForConditionalGeneration' in config['architectures']:
            print("  ✓ 模型配置文件正常")
        else:
            print("  ⚠ 模型配置文件可能需要更新")

    print("修复完成！")


if __name__ == '__main__':
    model_path = r"D:\models\blip\blip2-opt-2.7b"
    fix_blip2_config(model_path)