#!/usr/bin/env bash
# Distributed training launcher for 3 MacBook M5s over LAN.
#
# Usage (run on each machine):
#   NODE_RANK=0 MASTER_ADDR=<mac1-ip> bash scripts/launch_distributed.sh   # on mac1 (master)
#   NODE_RANK=1 MASTER_ADDR=<mac1-ip> bash scripts/launch_distributed.sh   # on mac2
#   NODE_RANK=2 MASTER_ADDR=<mac1-ip> bash scripts/launch_distributed.sh   # on mac3
#
# Requirements:
#   - All machines must share the same Python env and have the same data at PIX3D_ROOT.
#   - Port 12355 must be open on the master node (mac1).
#   - MASTER_ADDR must be the LAN IP of mac1 (e.g. 192.168.1.10).

set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the LAN IP of the master node (mac1)}"
MASTER_PORT="${MASTER_PORT:-12355}"
NNODES="${NNODES:-3}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK=0 on mac1, NODE_RANK=1 on mac2, NODE_RANK=2 on mac3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"  # 1 process per machine (CPU/MPS)

echo "Starting node rank=${NODE_RANK} / ${NNODES} | master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py
