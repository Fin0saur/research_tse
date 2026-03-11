#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

def load_speech_json(path):
    """
    读取资源库: spk_id -> list of {utt_id, path}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate deterministic mixture2enrollment.txt")
    parser.add_argument("--samples_jsonl", type=str, required=True, help="Path to evaluation samples.jsonl")
    parser.add_argument("--speech_json", type=str, required=True, help="Path to resources/speech.json")
    parser.add_argument("--outfile", type=str, required=True, help="Output path for mixture2enrollment.txt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")
    args = parser.parse_args()

    random.seed(args.seed)

    spk2items = load_speech_json(args.speech_json)
    out_lines = []
    
    missing_enrolls = 0

    with open(args.samples_jsonl, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            mix_key = obj["key"]
            spk_ids = obj["spk"]
            src_map = obj["src"]

            for spk in spk_ids:
                mix_wav_path = src_map[spk][0]
                
                available_utts = spk2items.get(spk, [])
                
                valid_enrolls = [item for item in available_utts if item["path"] != mix_wav_path]
                if not valid_enrolls:
                    valid_enrolls = available_utts
                    missing_enrolls += 1
                    print(f"[Warning] Speaker {spk} has no other utterances. Reusing mix audio for enrollment in {mix_key}.")
                
                valid_enrolls = sorted(valid_enrolls, key=lambda x: x["utt_id"])
                chosen_enroll = random.choice(valid_enrolls)

                target_field = spk 
                enroll_relpath = chosen_enroll["path"] 

                out_lines.append(f"{mix_key}\t{target_field}\t{enroll_relpath}\n")
                
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)
        
    print(f"[OK] Generated {len(out_lines)} enrollment pairs to: {args.outfile}")
    if missing_enrolls > 0:
        print(f"[!] {missing_enrolls} pairs suffered from data leakage due to lack of alternative utterances.")

if __name__ == "__main__":
    main()