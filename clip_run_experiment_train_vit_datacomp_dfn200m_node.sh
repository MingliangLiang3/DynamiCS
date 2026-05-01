#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=open_clip
#SBATCH --time=12:00:00
#SBATCH --output=./logs/train/my-experiment-%J.out
#SBATCH --error=./logs/train/my-experiment-%J.err
#SBATCH --mail-user=xxx
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/mliang/anaconda3/etc/profile.d/conda.sh
conda activate open_clip_venv

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME="eno"
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=22802

INDEX=$(($SLURM_ARRAY_TASK_ID))
PR_ARRAY=(pretrain_datacomp_large_dfn_200m_train_ViT-B-16_RECLIP_112x112_dynamic_0.5_1.28B)

pre_trained=${PR_ARRAY["$INDEX"]}
echo "pre_trained="$pre_trained

# learning rate lr = base_lr×batchsize / 256 
# lr = 8e-6*3500*8/256 = 0.000875
# 213330000 * 6 = 1280000000

export PYTHONPATH="$PYTHONPATH:$PWD/src"
cd src

srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 torchrun --nnodes=2 --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  open_clip_train/main.py \
  --save-frequency=1 \
  --zeroshot-frequency=1 \
  --report-to=tensorboard \
  --train-data="/scratch-shared/mliang/data/datacomp-large-scale/dfn_200m/shards/{00000000..00020068}.tar" \
  --train-num-samples=213330000 \
  --sampling-probabilities-file="../data/DataComp/datacomp_large_dfn_200m/dinov2_embedding_vit_b16/datacomp_dfn_sampling_probs.json" \
  --imagenet-val="../data/imagenet_validation/val" \
  --csv-img-key=image \
  --csv-caption-key=caption \
  --model=ViT-B-16 \
  --resume='latest' \
  --lr=8.75e-4 \
  --beta1 0.9 \
  --beta2 0.95 \
  --warmup 1600 \
  --wd=0.2 \
  --batch-size=3500 \
  --aug-cfg scale='(0.40, 1.0)' \
  --force-image-size=112 \
  --epochs=6 \
  --workers=12 \
  --seed=42 \
  --local-loss \
  --gather-with-grad \
  --force-custom-text \
  --ddp-static-graph \
  --precision=amp_bf16 \
  --name=snellius/datacomp/pre-train/${pre_trained}
