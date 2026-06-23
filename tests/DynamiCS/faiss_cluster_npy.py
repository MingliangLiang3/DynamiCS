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
    parser.add_argument("--max-train", type=int, default=0,
                        help="Global cap on training samples (0 = no cap). "
                             "K-means needs no more than ~256*K points; for K=50000 "
                             "that is ~12.8M. More than this just wastes I/O and RAM.")
    parser.add_argument("--dim", type=int, default=768,
                        help="Embedding dimension (default: 768 for DINOv2-base)")
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(glob(f"{args.input}/*.npy"))

    # Spread the global budget evenly across files so the sample is
    # representative of the whole corpus, not just the first few files.
    per_file = args.per_file
    if args.max_train > 0:
        per_file = min(per_file, max(1, args.max_train // len(files)))

    total_samples = len(files) * per_file
    if args.max_train > 0:
        total_samples = min(total_samples, args.max_train)
    train = np.empty((total_samples, args.dim), dtype='float32')

    ptr = 0
    for fn in tqdm(files, desc="Loading embeddings"):
        if ptr >= total_samples:
            break
        # Memory-map so we never read the whole (multi-GB) file into RAM;
        # only the sampled rows are actually pulled from disk.
        x = np.load(fn, mmap_mode='r')
        n = len(x)
        take = min(per_file, n, total_samples - ptr)
        if take < n:
            idx = np.random.choice(n, take, replace=False)
            idx.sort()  # sorted indices -> sequential-ish reads, much faster on GPFS
            x_sample = np.asarray(x[idx], dtype='float32')
        else:
            x_sample = np.asarray(x[:take], dtype='float32')
        train[ptr:ptr + take] = x_sample
        ptr += take
        del x  # release the mmap before opening the next file

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
