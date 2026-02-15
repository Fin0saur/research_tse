import json
import argparse
import os
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--spatial_root",
                        required=True,
                        help="Directory containing spatial npy files, "
                        "named as {mix_id}.npy")
    return parser.parse_args()


def load_samples(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    args = parse_args()

    samples = load_samples(args.samples_jsonl)
    spatial_root = args.spatial_root

    spatial_index = {}

    for s in tqdm(samples):
        mix_id = s["key"]
        spk_ids = s["spk"]  # 顺序必须与 spk1 / spk2 对齐

        spatial_path = os.path.join(spatial_root, f"{mix_id}.npy")
        if not os.path.exists(spatial_path):
            raise FileNotFoundError(f"Spatial npy not found: {spatial_path}")

        spatial_raw = np.load(spatial_path, allow_pickle=True).item()

        # 构建完整 sources（mix-level，全量）
        sources = []
        for i, spk in enumerate(spk_ids):
            idx = i + 1
            sources.append({
                "spk":
                spk,
                "azimuth":
                float(spatial_raw[f"azimuth_spk{idx}"]),
                "elevation":
                float(spatial_raw[f"elevation_spk{idx}"]),
                "position":
                spatial_raw[f"pos_spk{idx}"].tolist(),
            })

        mic = {"center": spatial_raw["pos_mic_center"].tolist()}

        meta = {}
        if "snr_db" in spatial_raw:
            meta["snr_db"] = float(spatial_raw["snr_db"])
        if "sir_db" in spatial_raw:
            meta["sir_db"] = float(spatial_raw["sir_db"])

        # 为每个 target speaker 生成一个 entry
        for tgt_spk in spk_ids:
            mix_spk_id = f"{mix_id}::{tgt_spk}"
            spatial_index[mix_spk_id] = {
                "mix_id": mix_id,
                "target_spk": tgt_spk,
                "sources": sources,
                "mic": mic,
                "meta": meta,
            }

    with open(args.outfile, "w") as f:
        json.dump(spatial_index, f, indent=2)

    print(f"Saved spatial cues to {args.outfile}")


if __name__ == "__main__":
    main()
