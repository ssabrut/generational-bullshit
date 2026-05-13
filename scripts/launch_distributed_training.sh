#!/bin/bash

# Distributed training launcher for Pixel2Mesh with DINOv2
# Usage: ./launch_distributed_training.sh [num_gpus] [batch_size] [epochs]

NUM_GPUS=${1:-1}
BATCH_SIZE=${2:-16}
EPOCHS=${3:-100}
PIX3D_ROOT=${4:-"../data/pix3d"}

echo "Launching distributed training with $NUM_GPUS GPUs"
echo "Batch size: $BATCH_SIZE per GPU"
echo "Epochs: $EPOCHS"
echo "Pix3D root: $PIX3D_ROOT"

# Set master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    train_pixel2mesh_dinov2.py \
    --pix3d-root $PIX3D_ROOT \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --hidden-dim 256 \
    --n-stages 3 \
    --n-gcn-blocks 3 \
    --num-workers 4

echo "Training completed!"
