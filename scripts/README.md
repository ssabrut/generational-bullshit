# Distributed training (2-Mac CPU cluster)

`train_ddp.py` runs the student model across N Macs over LAN using PyTorch
DDP with the `gloo` backend (CPU-only).

A single launcher script — `launch_ddp.sh` — is used on every node. The only
thing that differs between machines is the **`NODE_RANK` argument**.

## What you'll need

- Both Macs on the same LAN, able to ping each other.
- Same Python env on both (same torch version, same package set).
- Dataset cache (`data/teacher_cache/`) present on both machines at the same
  relative path — DDP does not share files, each rank reads locally.
- An open TCP port reachable from workers → master (default `29500`).

## Step 1 — Find the master's LAN IP

On the machine that will be node 0 (the master):

```bash
ipconfig getifaddr en0      # WiFi
# or
ipconfig getifaddr en1      # Ethernet
```

Note the IP, e.g. `192.168.1.10`.

## Step 2 — Sync the cache

```bash
rsync -avh --progress data/teacher_cache/ user@<worker-ip>:/path/to/repo/data/teacher_cache/
```

## Step 3 — Start the master (NODE_RANK=0)

```bash
export MASTER_ADDR=192.168.1.10   # this machine's IP
export MASTER_PORT=29500
export NNODES=2
./scripts/launch_ddp.sh 0
```

It will block waiting for the worker(s) to join.

## Step 4 — Start the worker (NODE_RANK=1)

On the second Mac, with **the same** `MASTER_ADDR` (pointing at node 0):

```bash
export MASTER_ADDR=192.168.1.10   # node 0's IP, not this machine's
export MASTER_PORT=29500
export NNODES=2
./scripts/launch_ddp.sh 1
```

Both processes will rendezvous and training will begin.

## Single-node debug

```bash
NNODES=1 ./scripts/launch_ddp.sh 0
```

## Configuration

Env vars (defaults shown). **`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and
`NPROC_PER_NODE` must be identical on every node.**

| var                  | default            | meaning                                  |
| -------------------- | ------------------ | ---------------------------------------- |
| `MASTER_ADDR`        | `192.168.1.10`     | node 0's LAN IP                          |
| `MASTER_PORT`        | `29500`            | TCP port for gloo rendezvous             |
| `NNODES`             | `2`                | total machines                           |
| `NPROC_PER_NODE`     | `1`                | processes per machine                    |
| `GLOO_SOCKET_IFNAME` | (auto)             | force a NIC, e.g. `en0`                  |
| `EPOCHS`             | `20`               | training epochs                          |
| `BATCH_SIZE`         | `4`                | **per-rank** batch size                  |
| `OUT_DIR`            | `runs/student_ddp` | checkpoint + preview dir (rank 0 only)   |

Global batch size = `BATCH_SIZE * NNODES * NPROC_PER_NODE`. Defaults: `4 * 2 * 1 = 8`.

Any extra arguments after `NODE_RANK` are forwarded to `train_ddp.py`:

```bash
./scripts/launch_ddp.sh 0 --lr 5e-4 --w-edge 0.005
```

## Troubleshooting

- **`AssertionError: local_world_size > 0`** — `NPROC_PER_NODE` is `<= 0`.
- **Hangs at rendezvous** — firewall on node 0 blocking `MASTER_PORT`, or
  workers using the wrong `MASTER_ADDR`. Start node 0 *before* the workers.
- **`Connection refused`** — wrong `MASTER_ADDR`, or master hasn't started.
- **gloo `op.nread == op.preamble.nbytes`** — collective ops out of sync.
  Make sure every rank participates in every `forward` / `broadcast` /
  `all_reduce`. Calling `ddp_model(...)` on a single rank will trigger this.
- **gloo picks the wrong interface** — `export GLOO_SOCKET_IFNAME=en0`
  (the interface that owns `MASTER_ADDR`).
- **Mismatched dataset sizes between ranks** — both caches must contain the
  same entries. Sync from a single source of truth.

## Output

Only rank 0 writes artifacts to `OUT_DIR`:

- `best.pt`, `last.pt` — checkpoints (underlying model, not the DDP wrapper)
- `history.json`, `loss_curve.png` — training metrics
- `preview_eNNN.png` — per-epoch validation previews
- `config.json` — run configuration
