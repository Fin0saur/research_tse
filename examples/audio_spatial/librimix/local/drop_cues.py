#!/usr/bin/env python3
# local/drop_cues.py
import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_json', type=str, required=True, help="Input full JSON file")
    parser.add_argument('--out_json', type=str, required=True, help="Output dropped JSON file")
    parser.add_argument('--keep_ratio', type=float, default=0.6, help="Ratio of keys to keep (0.0 to 1.0)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.in_json, 'r') as f:
        data = json.load(f)

    # 获取所有的 Key (spk_id 或 mix_spk_id)
    keys = list(data.keys())
    # 按照 seed 打乱，保证每次运行结果一致
    keys.sort() # 先排序保证跨平台一致性
    random.shuffle(keys)

    # 计算需要保留的数量
    keep_count = int(len(keys) * args.keep_ratio)
    keep_keys = set(keys[:keep_count])

    # 过滤数据，只保留选中的 keys
    filtered_data = {k: v for k, v in data.items() if k in keep_keys}

    with open(args.out_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"[{args.in_json}] Total Keys: {len(keys)} -> Kept: {len(filtered_data)} (Ratio: {args.keep_ratio}, Seed: {args.seed})")

if __name__ == "__main__":
    main()