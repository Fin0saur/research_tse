import json
import os

# 1. 你的训练集元数据路径 (以你代码中加载的音频字典为例)
# 如果你使用的是 wesep 生成的 data/clean/train/samples.jsonl，请替换为相应的读取逻辑
json_path = "/home/yxy05/code/research_tse/examples/audio/librimix/data/clean/train-100/cues/audio.json"
output_path = "/home/yxy05/code/research_tse/examples/audio/librimix/data/clean/train-100/spk2id.json"


def main():
    print(f"正在扫描训练集提取说话人: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        spk_audio_dict = json.load(f)

    # 2. 提取所有唯一的说话人 ID，并进行排序 (保证每次生成的映射一致)
    unique_speakers = sorted(list(spk_audio_dict.keys()))

    num_speakers = len(unique_speakers)
    print(f"✅ 共找到 {num_speakers} 个独立的说话人！")

    # 3. 构建映射字典
    spk2id = {spk: idx for idx, spk in enumerate(unique_speakers)}

    # 4. 保存为 JSON 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(spk2id, f, indent=4)

    print(f"🎉 字典已成功保存至: {output_path}")
    print(f"示例前 5 个映射: {list(spk2id.items())[:5]}")


if __name__ == "__main__":
    main()
