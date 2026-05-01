import argparse
import os
from glob import glob

import numpy as np
import faiss
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Assign each image embedding to its nearest cluster.")
    parser.add_argument("--embeddings", required=True,
                        help="Path to folder containing img_emb/*.npy files")
    parser.add_argument("--index", required=True,
                        help="Path to the FAISS index file")
    parser.add_argument("--output", required=True,
                        help="Output folder for cluster index .npy files")
    parser.add_argument("--batch-size", type=int, default=1_000_000,
                        help="Search batch size (default: 1000000)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob(f"{args.embeddings}/*.npy"))
    index = faiss.read_index(args.index)
    print(f"Loaded index with {index.ntotal} centers, dimension {index.d}.")
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT, "Expected IP index for spherical k-means."

    for fn in tqdm(files):
        X = np.load(fn, mmap_mode="r")
        n, d = X.shape
        out_idx = np.empty((n, 1), dtype=np.int64)

        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            xb = np.asarray(X[s:e], dtype="float32", order="C").copy()
            faiss.normalize_L2(xb)
            _, I = index.search(xb, k=1)
            out_idx[s:e, 0] = I[:, 0]

        out_file = os.path.join(args.output, os.path.basename(fn).replace(".npy", "_indices.npy"))
        np.save(out_file, out_idx)


if __name__ == "__main__":
    main()
