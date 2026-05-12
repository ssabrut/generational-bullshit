#!/usr/bin/env bash
# Distributed training launcher for 3 MacBook M5s over LAN.
#
# Usage (run on each machine):
#   NODE_RANK=0 MASTER_ADDR=10.10.10.1 bash scripts/launch_distributed.sh   # on mac1 (master)
#   NODE_RANK=1 MASTER_ADDR=10.10.10.1 bash scripts/launch_distributed.sh   # on mac2
#   NODE_RANK=2 MASTER_ADDR=10.10.10.1 bash scripts/launch_distributed.sh   # on mac3
#
# Requirements:
#   - All machines must share the same Python env and have the same data at PIX3D_ROOT.
#   - Port 12355 must be open on the master node (mac1).
#   - MASTER_ADDR must be the static LAN IP of mac1 on the wired switch (e.g. 10.10.10.1).
#   - Set GLOO_IFNAME to your USB-C Ethernet adapter interface (find it with: ifconfig | grep -A1 "10.10.10")
#     Common names on macOS: en6, en7, en8 — varies by adapter.

set -euo pipefail
 
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the LAN IP of the master node (mac1), e.g. 10.10.10.1}"
MASTER_PORT="${MASTER_PORT:-12355}"
NNODES="${NNODES:-3}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK=0 on mac1, NODE_RANK=1 on mac2, NODE_RANK=2 on mac3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Find the interface bound to the wired subnet automatically, or accept an override.
if [ -z "${GLOO_IFNAME:-}" ]; then
    GLOO_IFNAME=$(ifconfig | awk '/inet 10\.10\.10\./{found=1} found && /flags=/{print $1; exit}' RS='\n' | \
                  awk 'NR==1{print}' || true)
    # Fallback: walk interfaces in reverse to find the 10.10.10.x one
    if [ -z "${GLOO_IFNAME}" ]; then
        GLOO_IFNAME=$(ifconfig | grep -B5 "inet 10\.10\.10\." | awk '/^[a-z]/{iface=$1} /inet 10\.10\.10\./{print iface}' | tr -d ':')
    fi
    if [ -z "${GLOO_IFNAME}" ]; then
        echo "ERROR: Could not auto-detect the wired Ethernet interface."
        echo "Run: ifconfig | grep -B4 '10.10.10'   to find it, then rerun with GLOO_IFNAME=<name>"
        exit 1
    fi
fi

ERROR_FILE="/tmp/torchrun_rank${NODE_RANK}_error.json"
echo "Starting node rank=${NODE_RANK} / ${NNODES} | master=${MASTER_ADDR}:${MASTER_PORT} | iface=${GLOO_IFNAME}"
echo "Traceback (on failure): ${ERROR_FILE}"

GLOO_TRANSPORT=tcp \
GLOO_SOCKET_IFNAME="${GLOO_IFNAME}" \
TP_SOCKET_IFNAME="${GLOO_IFNAME}" \
TORCHELASTIC_ERROR_FILE="${ERROR_FILE}" \
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py
