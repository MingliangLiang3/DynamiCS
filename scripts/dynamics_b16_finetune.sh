#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=gpu_h100
#SBATCH --array=0-0
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=open_clip
#SBATCH --time=24:00:00
#SBATCH --output=./logs/train/my-experiment-%J.out
#SBATCH --error=./logs/train/my-experiment-%J.err
#SBATCH --mail-user=xxx
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/xxx/anaconda3/etc/profile.d/conda.sh
conda activate open_clip_venv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME="eno"
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=22802


INDEX=$(($SLURM_ARRAY_TASK_ID))
PR_ARRAY=(
pretrain_datacomp_large_dfn_200m_train_ViT-B-16_RECLIP_112x112_dynamic_0.5_1.28B
)

pre_trained=${PR_ARRAY["$INDEX"]}
echo "pre_trained="$pre_trained

export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src"
cd "$REPO_ROOT/src"

# learning rate lr = base_lr×batchsize / 256 
# lr = (4e-7)*1000*8/256 = 0.0000125

srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 torchrun --nnodes=2 --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  open_clip_train/main.py \
  --report-to=tensorboard \
  --zeroshot-frequency=1 \
  --train-data="/scratch-shared/mliang/data/datacomp-large-scale/dfn_200m/shards/{00000000..00020068}.tar" \
  --train-num-samples=128000000 \
  --sampling-probabilities-file="../data/DataComp/datacomp_large_dfn_200m/dinov2_embedding_vit_b16/datacomp_dfn_sampling_probs.json" \
  --imagenet-val="../data/imagenet_validation/val" \
  --csv-img-key=image \
  --csv-caption-key=caption \
  --model=ViT-B-16 \
  --aug-cfg scale='(0.40, 1.0)' \
  --pretrained="./logs/datacomp/pre-train/${pre_trained}/checkpoints/epoch_6.pt" \
  --resume='latest' \
  --batch-size=1000 \
  --lr=1.25e-5 \
  --beta1 0.9 \
  --beta2 0.95 \
  --wd=0.2 \
  --lr-warmup-epochs=0.1 \
  --epochs=1 \
  --workers=6 \
  --seed=42 \
  --local-loss \
  --gather-with-grad \
  --force-custom-text \
  --ddp-static-graph \
  --precision=amp_bf16 \
  --name datacomp/fine_tune/${pre_trained}_dynamic_finetune_8k
