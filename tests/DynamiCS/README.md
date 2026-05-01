# DynamiCS Pipeline

Back to the [main README](../../README.md).

This page contains the full end-to-end DynamiCS workflow, from building sampling probabilities to using them during OpenCLIP pre-training and fine-tuning.

Before running the pipeline, install the training dependencies from the main README and make sure `orjson`, `pyarrow`, and a FAISS build are available in your environment.

The full workflow has two stages:

1. Build a per-sample probability file from the training data.
2. Use that file during OpenCLIP pre-training and fine-tuning.

## 1. Extract DINOv2 Embeddings

```bash
python tests/DynamiCS/embedding_dinov2.py \
  --input "/path/to/shards/{00000000..00001023}.tar" \
  --output /path/to/dinov2_embedding_vit_b16 \
  --model facebook/dinov2-base \
  --cache-dir /path/to/cache \
  --batch-size 2048 \
  --num-workers 8
```

This creates:

- `img_emb/*.npy`: normalized image embeddings
- `metadata/*.parquet`: image paths and captions aligned with the embeddings

## 2. Train the FAISS Cluster Index

```bash
python tests/DynamiCS/faiss_cluster_npy.py \
  --input /path/to/dinov2_embedding_vit_b16/img_emb \
  --output /path/to/faiss_50k.index \
  --clusters 50000 \
  --niter 10 \
  --dim 768
```

## 3. Refine Redundant Cluster Centers

```bash
python tests/DynamiCS/post_clustering_refinement.py \
  --input /path/to/faiss_50k.index \
  --output /path/to/faiss_50k_refined.index \
  --threshold 0.70
```

The paper uses a cosine-similarity threshold of `0.7`.

## 4. Assign Samples to Clusters

```bash
python tests/DynamiCS/search_cluster_index.py \
  --embeddings /path/to/dinov2_embedding_vit_b16/img_emb \
  --index /path/to/faiss_50k_refined.index \
  --output /path/to/cluster_indices
```

## 5. Count Cluster Sizes

```bash
python tests/DynamiCS/read_indices_cluster.py \
  --input /path/to/cluster_indices \
  --output /path/to/cluster_counts.tsv
```

## 6. Compute Cluster-Scaled Sampling Probabilities

```bash
python tests/DynamiCS/cluster_scaling.py \
  --meta /path/to/dinov2_embedding_vit_b16/metadata \
  --indices /path/to/cluster_indices \
  --cluster-counts /path/to/cluster_counts.tsv \
  --output /path/to/sample_probabilities.tsv \
  --alpha 0.2 \
  --target-total 106665000
```

`--target-total` controls the expected number of retained samples per epoch. For a 50% sampling budget, set it to half of the accessible dataset size.

## 7. Convert TSV to JSON

```bash
python tests/DynamiCS/probability_csv2json.py \
  --input /path/to/sample_probabilities.tsv \
  --output /path/to/datacomp_dfn_sampling_probs.json
```

The resulting JSON is a lookup of the form:

```json
{
  "sample_000001": 0.9051
}
```

This is the file consumed by `--sampling-probabilities-file`. In the actual training setup, the key is the sample filename (equivalently the WebDataset `__key__`) and the value is the sampling probability.

We additionally release a SHA256-keyed companion file on Hugging Face:
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoint-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/MingliangLiang3/DynamiCS-ViT-B-16-DataComp-DFN)

In that released file, the key is the sample SHA256 hash code and the value is the sampling probability. To use it for training, you should first map each SHA256 hash to your local sample filename and rewrite the JSON into the filename-keyed format above. The training code expects filename-based keys, not SHA256 keys.

## Training With DynamiCS

DynamiCS becomes active when `--sampling-probabilities-file` is passed to the OpenCLIP training code. The data loader uses the sample key (`__key__`) from each WebDataset example and stochastically keeps or drops it according to the probability stored in the JSON file.

### Pre-training Example

```bash
export PYTHONPATH="$PYTHONPATH:$PWD/src"

torchrun --nproc_per_node=4 -m open_clip_train.main \
  --train-data "/path/to/shards/{00000000..00020068}.tar" \
  --train-num-samples 213330000 \
  --sampling-probabilities-file /path/to/datacomp_dfn_sampling_probs.json \
  --imagenet-val /path/to/imagenet/val \
  --model ViT-B-16 \
  --batch-size 3500 \
  --force-image-size 112 \
  --lr 8.75e-4 \
  --beta1 0.9 \
  --beta2 0.95 \
  --wd 0.2 \
  --warmup 1600 \
  --aug-cfg scale='(0.40, 1.0)' \
  --epochs 6 \
  --workers 12 \
  --precision amp_bf16 \
  --local-loss \
  --gather-with-grad \
  --force-custom-text \
  --ddp-static-graph \
  --name dynamics/pretrain/vit_b16
```

### Fine-tuning Example

```bash
export PYTHONPATH="$PYTHONPATH:$PWD/src"

torchrun --nproc_per_node=4 -m open_clip_train.main \
  --train-data "/path/to/shards/{00000000..00020068}.tar" \
  --train-num-samples 128000000 \
  --sampling-probabilities-file /path/to/datacomp_dfn_sampling_probs.json \
  --imagenet-val /path/to/imagenet/val \
  --model ViT-B-16 \
  --pretrained /path/to/checkpoints/epoch_6.pt \
  --batch-size 1000 \
  --lr 1.25e-5 \
  --beta1 0.9 \
  --beta2 0.95 \
  --wd 0.2 \
  --lr-warmup-epochs 0.1 \
  --epochs 1 \
  --workers 6 \
  --precision amp_bf16 \
  --local-loss \
  --gather-with-grad \
  --force-custom-text \
  --ddp-static-graph \
  --name dynamics/finetune/vit_b16
```

For cluster training on Slurm, see the provided scripts:

- [`clip_run_experiment_train_vit_datacomp_dfn200m_node.sh`](../../clip_run_experiment_train_vit_datacomp_dfn200m_node.sh)
- [`clip_run_experiment_fine_tune_vit_datacomp_dfn200m_array.sh`](../../clip_run_experiment_fine_tune_vit_datacomp_dfn200m_array.sh)
