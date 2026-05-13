import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(
        backend="gloo",         # CPU backend
        init_method="env://"   # reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    )

def cleanup():
    dist.destroy_process_group()

def train():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Running on rank {rank} of {world_size}")

    # Example model
    model = nn.Linear(10, 1)
    ddp_model = DDP(model)  # wraps model for distributed gradient sync

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Each rank gets its own data shard in real usage
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    train()