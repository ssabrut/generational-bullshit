#!/usr/bin/env bash
# Launch DDP training on NODE 1 (the worker / non-master machine).
#
# Run this on your second Mac AFTER node 0 has started.
#
# Required env vars:
#   MASTER_ADDR   — LAN IP of NODE 0 (NOT this machine). Same value as node 0.
#   MASTER_PORT   — Same port as node 0 (default 29500).
set -euo pipefail

cd "$(dirname "$0")/.."

MASTER_ADDR=192.168.1.10
MASTER_PORT=29500
NPROC_PER_NODE=-1
NNODES=2
NODE_RANK=1
GLOO_SOCKET_IFNAME=en7

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_DIR="${OUT_DIR:-runs/student_ddp}"

echo "[node 1] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[node 1] nnodes=$NNODES  nproc_per_node=$NPROC_PER_NODE  node_rank=$NODE_RANK"

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/train_ddp.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --out-dir "$OUT_DIR" \
    --categories chair sofa \
    "$@"
