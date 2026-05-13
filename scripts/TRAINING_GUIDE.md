# Distributed Training Guide for Pixel2Mesh with DINOv2

This guide explains how to run the distributed training script for 100 epochs.

## Scripts Overview

- **train_pixel2mesh_dinov2.py** - Main training script with DDP support
- **launch_distributed_training.sh** - Launcher script for easy distributed training
- **train.py** - Template for basic distributed training setup (reference)

## Prerequisites

```bash
pip install torch torchvision torch-distributed
pip install tqdm pillow numpy
```

## Running Training

### Single GPU
```bash
cd scripts
python train_pixel2mesh_dinov2.py \
    --pix3d-root ../data/pix3d \
    --batch-size 16 \
    --epochs 100 \
    --hidden-dim 256 \
    --n-stages 3 \
    --n-gcn-blocks 3
```

### Multiple GPUs (Distributed)
```bash
cd scripts
bash launch_distributed_training.sh 2 16 100 ../data/pix3d
```

Where:
- `2` = number of GPUs
- `16` = batch size per GPU
- `100` = number of epochs
- `../data/pix3d` = path to Pix3D dataset

### Manual Distributed Launch with torch.distributed.launch
```bash
cd scripts
export MASTER_ADDR=localhost
export MASTER_PORT=29500
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_pixel2mesh_dinov2.py \
    --pix3d-root ../data/pix3d \
    --batch-size 16 \
    --epochs 100
```

### Manual Environment Variables (Custom Network/Cluster)
You can directly set distributed training environment variables without using torch.distributed.launch:

**Node 0 (Master):**
```bash
cd scripts
MASTER_ADDR=192.168.1.10 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 GLOO_SOCKET_IFNAME=en7 \
python train_pixel2mesh_dinov2.py \
    --pix3d-root ../data/pix3d \
    --batch-size 16 \
    --epochs 100
```

**Node 1 (Worker):**
```bash
cd scripts
MASTER_ADDR=192.168.1.10 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 GLOO_SOCKET_IFNAME=en7 \
python train_pixel2mesh_dinov2.py \
    --pix3d-root ../data/pix3d \
    --batch-size 16 \
    --epochs 100
```

Where:
- `MASTER_ADDR` - IP address of the master node
- `MASTER_PORT` - Port for communication (must be open between nodes)
- `RANK` - Process rank (0 for master, 1,2,... for workers)
- `WORLD_SIZE` - Total number of processes
- `GLOO_SOCKET_IFNAME` - Network interface to use (e.g., en7, eth0)

## Command Line Arguments

```
--pix3d-root PATH           Path to Pix3D dataset (default: ../data/pix3d)
--batch-size N              Batch size per GPU (default: 16)
--epochs N                  Number of training epochs (default: 100)
--lr-fpn FLOAT              Learning rate for FPN (default: 1e-4)
--lr-gcn FLOAT              Learning rate for GCN (default: 3e-5)
--hidden-dim N              Hidden dimension (default: 256)
--n-stages N                Number of deformation stages (default: 3)
--n-gcn-blocks N            GCN blocks per stage (default: 3)
--num-workers N             Data loading workers (default: 4)
```

## Training Configuration

The script is configured for:
- **100 epochs** of training
- **Distributed Data Parallel (DDP)** for multi-GPU training
- **Cosine Annealing Learning Rate Scheduler** for smooth learning rate decay
- **Mixed precision training** (automatic on CUDA devices)
- **Gradient clipping** (norm=1.0) for stability
- **Checkpointing every 10 epochs** + final checkpoint

## Key Features

1. **DDP Support**: Automatic data distribution across GPUs
2. **Automatic Mixed Precision**: Faster training on CUDA with AMP
3. **Gradient Clipping**: Prevents gradient explosion
4. **Learning Rate Scheduling**: Cosine annealing for smooth convergence
5. **Regular Checkpointing**: Saves model every 10 epochs
6. **Validation Evaluation**: Tests on validation set every 5 epochs

## Loss Weights

The training uses stage-weighted loss:
- Stage 1: 0.2
- Stage 2: 0.3
- Stage 3: 0.5

Combined loss includes:
- Chamfer distance (w=1.0)
- Edge length regularization (w=0.1)
- Laplacian smoothness (w=0.5)

## Output Files

After training completes, you'll have:
- `checkpoint_epoch_10.pth` - Checkpoint after 10 epochs
- `checkpoint_epoch_20.pth` - Checkpoint after 20 epochs
- ... (every 10 epochs)
- `furniture3d_dinov2_final.pth` - Final checkpoint after 100 epochs

## Model Architecture

- **Encoder**: DINOv2 ViT-S/14 (frozen) + FPN neck
- **Feature Extraction**: 4-level FPN pyramid (128², 64², 32², 16²)
- **Deformation**: 3 progressive stages with GCN blocks
- **Total Parameters**: ~28.4M (6.3M trainable)

## Dataset Format

Expects Pix3D dataset structure:
```
pix3d/
├── pix3d.json
├── img/
│   └── [image files]
└── model/
    └── [OBJ files]
```

## Memory Requirements

- Single GPU (16GB VRAM): batch_size=8
- Multiple GPUs: batch_size=16 per GPU (scales well)

## Troubleshooting

**Out of Memory**: Reduce `--batch-size`

**Slow Data Loading**: Increase `--num-workers` (8-16 recommended)

**DDP Connection Issues**: Check MASTER_ADDR and MASTER_PORT are available

**Missing Encoder Weights**: First run downloads DINOv2 from torch.hub (requires internet)

## From Notebook to Production

This script converts the notebook training into:
- ✅ Distributed training support
- ✅ 100 epoch training target
- ✅ Proper checkpointing
- ✅ Command-line configurability
- ✅ Memory-efficient mixed precision
- ✅ Batch processing with DistributedSampler
