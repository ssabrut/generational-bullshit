#!/usr/bin/env bash
# Launch DDP training on NODE 0 (the master / rank 0 machine).
#
# Run this on your primary Mac. It will wait for node 1 to connect before
# training starts.
#
# Required env vars (export before running, or edit defaults below):
#   MASTER_ADDR   — LAN IP of this machine (node 0). Find via `ipconfig getifaddr en0`.
#   MASTER_PORT   — Any free TCP port both macs can reach (default 29500).
#
# Optional:
#   NPROC_PER_NODE — Processes on this node (default 1; CPU DDP doesn't benefit
#                    from >1 per machine unless you have many cores to spare).
#   EPOCHS, BATCH_SIZE, OUT_DIR — passed through to the python script.
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

MASTER_ADDR=192.168.1.10
MASTER_PORT=29500
NPROC_PER_NODE=2
NNODES=2
NODE_RANK=0
export GLOO_SOCKET_IFNAME=en7

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_DIR="${OUT_DIR:-runs/student_ddp}"

# gloo prefers a specific interface; let it auto-detect by default. If the
# nodes can't find each other, set GLOO_SOCKET_IFNAME to your interface name
# (en0, en1, ...): `export GLOO_SOCKET_IFNAME=en0`.
echo "[node 0] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[node 0] nnodes=$NNODES  nproc_per_node=$NPROC_PER_NODE  node_rank=$NODE_RANK"

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
