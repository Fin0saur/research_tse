"""
统计 train/val/test 三个数据集上的 C0/C1 质量分布

方案A (二分类):
  C0 (Success): Dynamic_SISNRi >= 1
  C1 (Failure): Dynamic_SISNRi < 1
"""

import os
import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate


def analyze_split(labels_csv, split_name):
    """分析单个数据集的质量分布"""
    df = pd.read_csv(labels_csv)

    total = len(df)
    c0_mask = df['Dynamic_SISNRi'] >= 1.0
    c0_count = c0_mask.sum()
    c1_count = (~c0_mask).sum()

    c0_pct = c0_count / total * 100
    c1_pct = c1_count / total * 100

    # SISNRi 统计
    mean_sisnr = df['Dynamic_SISNRi'].mean()
    median_sisnr = df['Dynamic_SISNRi'].median()
    min_sisnr = df['Dynamic_SISNRi'].min()
    max_sisnr = df['Dynamic_SISNRi'].max()

    oracle_mean = df['Oracle_SISNRi'].mean() if 'Oracle_SISNRi' in df.columns else np.nan
    delta_mean = df['Delta_SISNRi'].mean() if 'Delta_SISNRi' in df.columns else np.nan

    return {
        'split': split_name,
        'total': total,
        'C0_success': c0_count,
        'C1_failure': c1_count,
        'C0_pct': c0_pct,
        'C1_pct': c1_pct,
        'mean_sisnr': mean_sisnr,
        'median_sisnr': median_sisnr,
        'min_sisnr': min_sisnr,
        'max_sisnr': max_sisnr,
        'oracle_mean': oracle_mean,
        'delta_mean': delta_mean,
    }


def main():
    parser = argparse.ArgumentParser(description='统计数据集质量分布')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录，包含 train_features/, val_features/, test_features/')
    args = parser.parse_args()

    splits = {
        'train': os.path.join(args.exp_dir, 'train_features', 'labels.csv'),
        'val': os.path.join(args.exp_dir, 'val_features', 'labels.csv'),
        'test': os.path.join(args.exp_dir, 'test_features', 'labels.csv'),
    }

    results = []
    for name, csv_path in splits.items():
        if os.path.exists(csv_path):
            r = analyze_split(csv_path, name)
            results.append(r)
            print(f"\n{'='*50}")
            print(f"  {name.upper()} SET")
            print(f"{'='*50}")
            print(f"  Total samples : {r['total']}")
            print(f"  C0 (Success)  : {r['C0_success']:6d} ({r['C0_pct']:.1f}%)")
            print(f"  C1 (Failure)  : {r['C1_failure']:6d} ({r['C1_pct']:.1f}%)")
            print(f"  Dynamic SISNRi : mean={r['mean_sisnr']:.2f}, median={r['median_sisnr']:.2f}")
            print(f"                  : min={r['min_sisnr']:.2f}, max={r['max_sisnr']:.2f}")
            if not np.isnan(r['oracle_mean']):
                print(f"  Oracle SISNRi : mean={r['oracle_mean']:.2f}")
            if not np.isnan(r['delta_mean']):
                print(f"  Delta SISNRi  : mean={r['delta_mean']:.2f}")
        else:
            print(f"\n⚠️  {csv_path} 不存在，跳过")

    if results:
        print(f"\n{'='*50}")
        print("  SUMMARY TABLE")
        print(f"{'='*50}")

        table_data = []
        for r in results:
            table_data.append([
                r['split'],
                r['total'],
                f"{r['C0_success']} ({r['C0_pct']:.1f}%)",
                f"{r['C1_failure']} ({r['C1_pct']:.1f}%)",
                f"{r['mean_sisnr']:.2f}",
                f"{r['median_sisnr']:.2f}",
            ])

        headers = ['Split', 'Total', 'C0 (Success)', 'C1 (Failure)', 'Mean SISNRi', 'Median SISNRi']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))


if __name__ == '__main__':
    main()
