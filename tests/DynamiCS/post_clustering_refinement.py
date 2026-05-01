import argparse

import faiss
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate cluster centers from a FAISS index."
    )
    parser.add_argument("--input", required=True,
                        help="Path to the input FAISS index file")
    parser.add_argument("--output", required=True,
                        help="Path for the refined output FAISS index file")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Cosine similarity threshold for merging clusters (default: 0.70)")
    return parser.parse_args()


def main():
    args = parse_args()

    src_index = faiss.read_index(args.input)
    n = src_index.ntotal
    if n == 0:
        raise ValueError("Source index is empty.")

    centers = src_index.reconstruct_n(0, n)
    centers = np.ascontiguousarray(centers.astype("float32"))
    d = centers.shape[1]
    faiss.normalize_L2(centers)

    dst_index = faiss.IndexFlatIP(d)
    kept = []

    print(f"Using threshold {args.threshold} for merging clusters.")
    for i in tqdm(range(n), desc="Merging clusters"):
        vec = centers[i:i + 1]
        if dst_index.ntotal == 0:
            dst_index.add(vec)
            kept.append(i)
            continue

        D, _ = dst_index.search(vec, 1)
        if D[0, 0] >= args.threshold:
            continue

        dst_index.add(vec)
        kept.append(i)

    faiss.write_index(dst_index, args.output)
    print(f"Kept {len(kept)} / {n} centers ({len(kept)/n:.1%})")
    print(f"Saved refined index to {args.output}")


if __name__ == "__main__":
    main()
