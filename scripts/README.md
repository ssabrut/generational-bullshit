# Distributed training (2-Mac CPU cluster)

`train_ddp.py` runs the student model across two Macs over LAN using PyTorch
DDP with the `gloo` backend (CPU-only).

## What you'll need

- Both Macs on the same LAN, able to ping each other.
- Same Python env on both (same torch version, same package set).
- The dataset cache (`data/teacher_cache/`) present on **both** machines at
  the same relative path — DDP does not share files, each rank reads locally.
- An open TCP port reachable from node 1 → node 0 (default `29500`).

## Step 1 — Find node 0's LAN IP

On node 0 (the "master"):

```bash
ipconfig getifaddr en0      # WiFi
# or
ipconfig getifaddr en1      # Ethernet
```

Note the IP, e.g. `192.168.1.10`.

## Step 2 — Sync the cache

Make sure `data/teacher_cache/{chair,sofa}/...` exists on both machines.
Easiest: `rsync` from node 0 to node 1:

```bash
rsync -avh --progress data/teacher_cache/ user@<node1-ip>:/path/to/repo/data/teacher_cache/
```

## Step 3 — Start node 0

```bash
export MASTER_ADDR=192.168.1.10   # node 0's IP
export MASTER_PORT=29500
./scripts/launch_ddp_node0.sh
```

It will print `[node 0]` and then block waiting for node 1 to join.

## Step 4 — Start node 1

In a shell on the second Mac (note: the `MASTER_ADDR` value is **node 0's IP**, not
this machine's):

```bash
export MASTER_ADDR=192.168.1.10   # same as node 0
export MASTER_PORT=29500
./scripts/launch_ddp_node1.sh
```

Both processes will rendezvous, log world_size=2, and training will begin.

## Configuration

Both launch scripts honor these env vars (defaults shown):

| var              | default            | meaning |
|------------------|--------------------|---------|
| `MASTER_ADDR`    | (required)         | node 0's LAN IP |
| `MASTER_PORT`    | `29500`            | TCP port for gloo rendezvous |
| `NNODES`         | `2`                | total machines |
| `NPROC_PER_NODE` | `1`                | processes per machine |
| `EPOCHS`         | `20`               | training epochs |
| `BATCH_SIZE`     | `4`                | **per-rank** batch size |
| `OUT_DIR`        | `runs/student_ddp` | checkpoint + preview dir (on node 0 only) |

The global batch size is `BATCH_SIZE * NNODES * NPROC_PER_NODE`. With defaults:
`4 * 2 * 1 = 8`.

Any extra args are forwarded to `train_ddp.py`, e.g.:

```bash
./scripts/launch_ddp_node0.sh --lr 5e-4 --w-edge 0.005
```

## Troubleshooting

- **Hangs at "Waiting for nodes to join"** — firewall on node 0 is blocking
  `MASTER_PORT`. On macOS: System Settings → Network → Firewall → allow
  incoming connections for python, or temporarily disable.
- **"Connection refused"** — wrong `MASTER_ADDR` on node 1, or node 0 hasn't
  started yet. Start node 0 first.
- **gloo picks the wrong interface** — force it:
  `export GLOO_SOCKET_IFNAME=en0` (the interface that owns `MASTER_ADDR`).
- **All-reduce hangs mid-epoch** — usually means one rank crashed silently.
  Check both terminals' logs. Reduce `NPROC_PER_NODE` to 1 if you raised it.
- **Mismatched dataset sizes between ranks** — both caches must contain the
  same entries. `rsync` from a single source of truth.
- **MPS / CUDA** — intentionally not supported here. MPS doesn't speak DDP;
  CUDA isn't on Mac. This script is CPU-only.

## Output

Only **rank 0 (node 0)** writes artifacts to `OUT_DIR`:

- `best.pt`, `last.pt` — checkpoints (underlying model, not DDP wrapper)
- `history.json`, `loss_curve.png` — training metrics
- `preview_eNNN.png` — per-epoch validation previews
- `config.json` — run configuration
