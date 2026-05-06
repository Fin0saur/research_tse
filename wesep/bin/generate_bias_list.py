"""
生成 1扩20 的偏差数据集列表 (纯文本处理，极速运行)
读取原本的 raw.list，结合 audio.json，生成包含 20 个不同 cue 的 bias_enroll20.jsonl
"""

import os
import json
import random
import argparse

def generate_bias(input_list, audio_json, output_jsonl, num_cues=20, seed=42):
    random.seed(seed)

    # 1. 加载声纹字典 (包含每个说话人的干净语音)
    print(f"🔍 加载声纹字典: {audio_json}")
    with open(audio_json, 'r', encoding='utf-8') as f:
        spk_audio_dict = json.load(f)

    expanded_data = []
    skipped_count = 0

    # 2. 逐行读取 Wesep 的原始 .list 文件 (其实就是 JSONL)
    print(f"⚙️ 正在读取并扩充: {input_list} (目标: 1扩{num_cues})")
    with open(input_list, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())

            # 获取当前混合音频的所有说话人
            spk_ids = data.get('spk', [])
            if isinstance(spk_ids, str):
                spk_ids = [spk_ids]

            # 随机抽 20 个 enrollment 组合 (spk1 一个, spk2 一个)
            num_spk = len(spk_ids)

            # 为每个说话人抽取 num_cues 个候选路径
            spk_paths = []
            for spk_id in spk_ids:
                spk_id = str(spk_id)
                available_utts = spk_audio_dict.get(spk_id, [])
                if not available_utts:
                    skipped_count += 1
                    break
                utt_paths = [item["path"] for item in available_utts]
                if len(utt_paths) >= num_cues:
                    selected = random.sample(utt_paths, num_cues)
                else:
                    selected = random.choices(utt_paths, k=num_cues)
                spk_paths.append(selected)
            else:
                # 所有说话人都找到了路径，生成 20 份数据
                for idx in range(num_cues):
                    new_data = data.copy()
                    new_data['key'] = f"{data['key']}_cue{idx:02d}"
                    # 设置 audio_spk1, audio_spk2, ...
                    for spk_idx, paths in enumerate(spk_paths, start=1):
                        new_data[f'audio_spk{spk_idx}'] = paths[idx]
                    expanded_data.append(new_data)
                continue

    # 3. 落盘保存
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for d in expanded_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"✅ 构建完成！")
    print(f"   - 跳过无效样本: {skipped_count} 个")
    print(f"   - 最终生成偏差样本总数: {len(expanded_data)} 行")
    print(f"   - 已保存至: {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始的 raw.list 路径")
    parser.add_argument("--audio_dict", required=True, help="audio.json 路径")
    parser.add_argument("--output", required=True, help="输出的 bias_enroll20.jsonl 路径")
    parser.add_argument("--num_cues", type=int, default=20, help="每个混合的 cue 扩充数量")
    args = parser.parse_args()

    generate_bias(args.input, args.audio_dict, args.output, num_cues=args.num_cues)