import argparse
from glob import glob

import numpy as np
import faiss
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FAISS spherical k-means index on DINOv2 embeddings.")
    parser.add_argument("--input", required=True,
                        help="Path to folder containing img_emb/*.npy files")
    parser.add_argument("--output", required=True,
                        help="Path for the output FAISS index file")
    parser.add_argument("--clusters", type=int, default=50000,
                        help="Number of clusters K (default: 50000)")
    parser.add_argument("--niter", type=int, default=10,
                        help="Number of k-means iterations (default: 10)")
    parser.add_argument("--per-file", type=int, default=250000,
                        help="Max samples to draw per embedding file (default: 250000)")
    parser.add_argument("--dim", type=int, default=768,
                        help="Embedding dimension (default: 768 for DINOv2-base)")
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(glob(f"{args.input}/*.npy"))
    total_samples = len(files) * args.per_file
    train = np.empty((total_samples, args.dim), dtype='float32')

    ptr = 0
    for fn in tqdm(files, desc="Loading embeddings"):
        x = np.load(fn)
        n = len(x)
        take = min(args.per_file, n)
        if take < n:
            idx = np.random.choice(n, take, replace=False)
            x_sample = x[idx]
        else:
            x_sample = x[:take]
        train[ptr:ptr + take] = x_sample.astype('float32')
        ptr += take

    train = train[:ptr]
    print("Final train shape:", train.shape)

    faiss.normalize_L2(train)

    kmeans = faiss.Kmeans(
        args.dim, args.clusters,
        niter=args.niter,
        verbose=True,
        spherical=True,
        max_points_per_centroid=1000,
        gpu=True,
    )
    kmeans.train(train)

    centroids = kmeans.centroids.copy()
    index = faiss.IndexFlatIP(args.dim)
    faiss.normalize_L2(centroids)
    index.add(centroids)

    faiss.write_index(index, args.output)
    print(f"Saved index to {args.output}")


if __name__ == "__main__":
    main()
