import argparse
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate cluster assignment counts across index files.")
    parser.add_argument("--input", required=True,
                        help="Path to folder containing *_indices.npy files")
    parser.add_argument("--output", required=True,
                        help="Path for the output cluster counts CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    indices_files = sorted(glob(f"{args.input}/*.npy"))
    indices_files = sorted(indices_files, key=lambda x: int(x.split("_")[-2]))
    print("indices files:", indices_files)

    indices_list = []
    for indices_file in tqdm(indices_files):
        img_indices = np.load(indices_file, allow_pickle=True)
        print(img_indices.shape)
        indices_list.append(img_indices)

    indices = np.concatenate(indices_list, axis=0)
    print("All indices shape:", indices.shape)

    df_indices = pd.DataFrame(indices, columns=["cluster_index"])
    cluster_index_counts = df_indices['cluster_index'].value_counts().sort_index()
    print(cluster_index_counts)

    cluster_index_counts.to_csv(args.output, sep='\t', header=['count'])
    print(f"Saved cluster counts to {args.output}")


if __name__ == "__main__":
    main()
