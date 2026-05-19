#!/usr/bin/env bash
# Launch DDP training on ONE node of a multi-Mac CPU cluster.
#
# Usage:
#   ./scripts/launch_ddp.sh <NODE_RANK> [extra args passed to train_ddp.py]
#
# Examples:
#   # On the master Mac (rank 0):
#   ./scripts/launch_ddp.sh 0
#
#   # On the second Mac (rank 1):
#   ./scripts/launch_ddp.sh 1
#
#   # Single-node debug run (rank 0 with NNODES=1):
#   NNODES=1 ./scripts/launch_ddp.sh 0
#
# Required env vars (must be IDENTICAL on every node):
#   MASTER_ADDR        — LAN IP of the rank-0 machine (e.g. 192.168.1.10).
#   MASTER_PORT        — TCP port (default 29500). Must be reachable on rank 0.
#   NNODES             — Total number of machines (default 2).
#   NPROC_PER_NODE     — Processes per machine (default 1). MUST match on every node.
#   GLOO_SOCKET_IFNAME — (optional) network interface, e.g. en0, en7.
#
# Optional run config (forwarded to train_ddp.py):
#   EPOCHS, BATCH_SIZE, OUT_DIR, SUBDIVISIONS, N_STAGES
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

# ── parse positional NODE_RANK ────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <NODE_RANK> [extra args]" >&2
  echo "  NODE_RANK: 0 for master, 1..N-1 for workers" >&2
  exit 2
fi
NODE_RANK="$1"
shift  # remaining args go to train_ddp.py

# ── cluster config (override via env) ─────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-192.168.1.10}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Pin gloo to a specific NIC if you have multiple interfaces. Exported so
# torchrun's child python processes inherit it.
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME
fi

# ── run config (forwarded) ────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_DIR="${OUT_DIR:-runs/student_ddp}"
SUBDIVISIONS="${SUBDIVISIONS:-3}"
N_STAGES="${N_STAGES:-1}"

# ── sanity check ──────────────────────────────────────────────────────────────
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "Error: NODE_RANK=$NODE_RANK out of range [0, $((NNODES - 1))]" >&2
  exit 2
fi

GLOBAL_WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
echo "[node $NODE_RANK/$NNODES] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[node $NODE_RANK/$NNODES] nproc_per_node=$NPROC_PER_NODE  world_size=$GLOBAL_WORLD_SIZE"
echo "[node $NODE_RANK/$NNODES] gloo_iface=${GLOO_SOCKET_IFNAME:-auto}"

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
    --subdivisions "$SUBDIVISIONS" \
    --n-stages "$N_STAGES" \
    --categories chair sofa \
    "$@"
