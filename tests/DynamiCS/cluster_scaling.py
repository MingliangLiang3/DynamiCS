import argparse
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-sample resampling probabilities via power-law cluster scaling."
    )
    parser.add_argument("--meta", required=True,
                        help="Path to folder containing metadata/*.parquet files")
    parser.add_argument("--indices", required=True,
                        help="Path to folder containing *_indices.npy files")
    parser.add_argument("--cluster-counts", required=True,
                        help="Path to cluster counts CSV (output of read_indices_cluster.py)")
    parser.add_argument("--output", required=True,
                        help="Path for the output CSV with per-sample probabilities")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Power-law exponent for cluster resampling (default: 0.2)")
    parser.add_argument("--target-total", type=int, required=True,
                        help="Target total number of samples after resampling")
    return parser.parse_args()


def resample_clusters(counts, alpha, target_total):
    counts = np.array(counts, dtype=float)
    transformed = counts ** alpha
    resampled = transformed / transformed.sum() * target_total
    return np.round(resampled).astype(int)


def main():
    args = parse_args()

    meta_files = sorted(glob(f"{args.meta}/*.parquet"))
    indices_files = sorted(glob(f"{args.indices}/*.npy"))

    meta_files = sorted(meta_files, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    indices_files = sorted(indices_files, key=lambda x: int(x.split("_")[-2]))

    cluster_counts_df = pd.read_csv(args.cluster_counts, sep="\t")
    counts = cluster_counts_df['count'].tolist()

    resampled_counts = resample_clusters(counts, alpha=args.alpha, target_total=args.target_total)
    cluster_counts_df['resampled_count'] = resampled_counts
    cluster_counts_df['probability'] = (
        cluster_counts_df['resampled_count'] / cluster_counts_df['count']
    ).round(6)

    cluster_probabilities = cluster_counts_df.set_index('cluster_index')['probability'].to_dict()

    assert len(meta_files) == len(indices_files), \
        "Number of metadata files and indices files must be the same"

    all_meta = []
    for meta_file, indices_file in tqdm(zip(meta_files, indices_files), total=len(meta_files)):
        meta_df = pd.read_parquet(meta_file)
        indices = np.load(indices_file, allow_pickle=True)
        meta_df = meta_df.reset_index(drop=True)
        meta_df['cluster_index'] = indices.flatten()  # flatten from (n,1) to (n,)
        meta_df['probability'] = meta_df['cluster_index'].map(cluster_probabilities)
        all_meta.append(meta_df)

    combined_meta = pd.concat(all_meta, ignore_index=True)
    print("Combined metadata shape:", combined_meta.shape)
    combined_meta.to_csv(args.output, index=False, sep='\t')
    print(f"Saved sampling probabilities to {args.output}")


if __name__ == "__main__":
    main()
