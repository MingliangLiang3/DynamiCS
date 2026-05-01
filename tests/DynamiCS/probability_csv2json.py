import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert per-sample probability CSV to a {image_stem: probability} JSON lookup."
    )
    parser.add_argument("--input", required=True,
                        help="Input tab-separated CSV with columns: image_path, probability")
    parser.add_argument("--output", required=True,
                        help="Path for the output JSON file")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="Chunk size for reading CSV (default: 100000)")
    return parser.parse_args()


def main():
    args = parse_args()

    reader = pd.read_csv(
        args.input,
        chunksize=args.chunk_size,
        sep="\t",
        usecols=["image_path", "probability"],
        on_bad_lines="skip",
        lineterminator='\n',
    )

    big_dict = {}
    for i, chunk in enumerate(reader, start=1):
        keys = chunk["image_path"].astype(str).apply(lambda p: Path(p).stem)
        vals = chunk["probability"].astype(float)
        big_dict.update(dict(zip(keys, vals)))
        print(f"Processed chunk {i}, total keys: {len(big_dict)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(big_dict, f, ensure_ascii=False, indent=2)

    print("Done:", args.output)


if __name__ == "__main__":
    main()
