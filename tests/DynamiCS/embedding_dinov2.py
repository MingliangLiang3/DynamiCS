import os
import gc
import argparse
import io
import time

from braceexpand import braceexpand
import torch
import webdataset as wds
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings from a WebDataset.")
    parser.add_argument("--input", required=True,
                        help="Input shard pattern, e.g. '/data/shards/{00000..00100}.tar'")
    parser.add_argument("--output", required=True,
                        help="Output folder for img_emb/ and metadata/")
    parser.add_argument("--model", default="facebook/dinov2-base",
                        help="HuggingFace model name (default: facebook/dinov2-base)")
    parser.add_argument("--cache-dir", default="cache",
                        help="Cache directory for model and dataset (default: cache)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--write-every", type=int, default=500,
                        help="Flush embeddings to disk every N batches (default: 500)")
    parser.add_argument("--samples-per-shard", type=int, default=10000,
                        help="Approximate samples per shard, used only for tqdm estimate (default: 10000)")
    return parser.parse_args()


def main():
    args = parse_args()

    img_emb_folder = os.path.join(args.output, "img_emb")
    metadata_folder = os.path.join(args.output, "metadata")
    os.makedirs(img_emb_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

    # ---- Speed flags ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model & Preprocessing ----
    model = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir)
    model.eval()
    model.to(device, dtype=torch.bfloat16)

    processor = AutoImageProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)

    # ---- WebDataset Pipeline ----
    input_shards = list(braceexpand(args.input))

    dataset = wds.WebDataset(
        input_shards,
        shardshuffle=False,
        cache_dir=args.cache_dir,
        cache_size=10**10,
        handler=wds.handlers.warn_and_continue,
    )

    def preprocess_image(item, image_key="jpg", caption_key="txt"):
        try:
            image = Image.open(io.BytesIO(item[image_key])).convert("RGB")
            if image.size[0] < 10 or image.size[1] < 10:
                image = image.resize((224, 224))
            image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            caption = item[caption_key].decode("utf-8")
            return {
                "image_filename": item["__local_path__"] + '/' + item["__key__"] + '.jpg',
                "image_tensor": image_tensor,
                "caption": caption,
            }
        except Exception as e:
            print(f"Skipping corrupted image: {e}")
            return None

    dataset = dataset.map(preprocess_image, handler=wds.handlers.warn_and_continue)
    dataset = dataset.select(lambda x: x is not None)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,
    )

    def write_batch(image_embeddings, image_names, caption_list, batch_num_str):
        img_emb_mat = np.concatenate(image_embeddings)
        np.save(f"{img_emb_folder}/img_emb_{batch_num_str}.npy", img_emb_mat)
        pd.DataFrame({
            "image_path": image_names,
            "caption": caption_list,
        }).to_parquet(f"{metadata_folder}/metadata_{batch_num_str}.parquet", engine="pyarrow")

    image_embedding = []
    image_names = []
    caption_list = []
    idx = -1

    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    total_estimate = int(len(input_shards) * args.samples_per_shard / args.batch_size)

    with torch.inference_mode():
        for idx, item in enumerate(tqdm(dataloader, desc="Processing Images", total=total_estimate)):
            pixel_values = item["image_tensor"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=pixel_values)
                image_features = outputs.last_hidden_state[:, 0, :]
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

            image_embedding.append(image_features.float().cpu().numpy())
            image_names.extend(item["image_filename"])
            caption_list.extend(item["caption"])

            if (idx + 1) % args.write_every == 0:
                batch_num_str = str(idx // args.write_every).zfill(3)
                write_batch(image_embedding, image_names, caption_list, batch_num_str)
                image_embedding.clear()
                image_names.clear()
                caption_list.clear()
                gc.collect()

    if image_embedding:
        batch_num_str = str(idx // args.write_every + 1).zfill(3)
        write_batch(image_embedding, image_names, caption_list, batch_num_str)
        gc.collect()

    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    main()
