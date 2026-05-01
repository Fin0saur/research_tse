#!/usr/bin/env python3
import argparse
import json
import glob
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=
        "Directly generate fixed_enroll.json with a single forced enroll")
    parser.add_argument("--samples_jsonl", type=str, required=True)
    parser.add_argument("--enroll_dir", type=str, required=True)
    parser.add_argument("--outfile",
                        type=str,
                        required=True,
                        help="Path to output fixed_enroll.json")
    args = parser.parse_args()

    # 1. 抓取那个唯一的固定 Enroll 音频
    enroll_files = glob.glob(os.path.join(args.enroll_dir, "*.wav"))
    if len(enroll_files) != 1:
        raise ValueError(
            f"[Error] Expected exactly 1 wav file in {args.enroll_dir}")

    fixed_enroll_path = os.path.abspath(enroll_files[0])
    enroll_utt_id = os.path.splitext(os.path.basename(fixed_enroll_path))[0]
    print(
        f"[Info] Target locked: All enrollments will point to -> {fixed_enroll_path}"
    )

    fixed_enroll_json = {}

    # 2. 读取 JSONL，强行构建 JSON 字典
    with open(args.samples_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            mix_key = obj["key"]
            spk_ids = obj["spk"]

            for spk in spk_ids:
                # 构建 Wesep 要求的 "MixID::SpkID" 键
                dict_key = f"{mix_key}::{spk}"
                # 强行塞入唯一固定的 enroll 信息
                fixed_enroll_json[dict_key] = [{
                    "utt_id": enroll_utt_id,
                    "path": fixed_enroll_path
                }]

    # 3. 直接写出 final JSON
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixed_enroll_json, f, indent=2)

    print(
        f"[OK] Generated strictly formatted JSON for {len(fixed_enroll_json)} speaker entries to: {args.outfile}"
    )


if __name__ == "__main__":
    main()
