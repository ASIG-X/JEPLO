import os
import torch
import torch.distributed as dist
import lejepa as ds


def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Choose an available port
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )  # nccl for GPU, gloo for CPU


def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    """Demonstrates basic DDP usage."""
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # Create model and wrap with DDP
    if world_size == 1:
        Xs = []
        for r in range(2):
            torch.manual_seed(r)
            Xs.append(torch.randn(2, 2, device=rank))
        X = torch.cat(Xs, 0).requires_grad_(True)
    else:
        torch.manual_seed(rank)
        X = torch.randn(2, 2, device=rank, requires_grad=True)

    # print(f"rank {rank}. X:", X)
    uni_test = ds.univariate.Moments().to(rank)
    loss = uni_test(X)
    print(f"rank {rank}. loss:", loss)
    loss.mean().backward()
    print(f"rank {rank}. grad:", X.grad)
    cleanup()


def run_demo(demo_fn, world_size):
    """Spawns multiple processes for DDP."""
    torch.multiprocessing.spawn(
        demo_fn, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        print(f"Running DDP with 1 GPUs.")
        run_demo(demo_basic, 1)
        print(f"Running DDP with 2 GPUs.")
        run_demo(demo_basic, 2)
    else:
        print("No GPUs found. DDP requires GPUs for this example.")
