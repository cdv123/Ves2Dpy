import torch
import math
import os
from torch import distributed as dist
from dataclasses import dataclass


def set_bg_flow(bgFlow, speed):
    def get_flow(X):
        N = X.shape[0] // 2  # Assuming the input X is split into two halves
        x, y = X[:N, :], X[N:, :]
        if bgFlow == "relax":
            return torch.zeros_like(X)  # Relaxation
        elif bgFlow == "shear":
            return speed * torch.vstack((X[N:], torch.zeros_like(X[:N])))  # Shear
        elif bgFlow == "taylorGreen":
            vortexSize = 2.5
            scale = math.pi / vortexSize
            v_x = torch.sin(x * scale) * torch.cos(y * scale)
            v_y = -torch.cos(x * scale) * torch.sin(y * scale)
            vInf = vortexSize * torch.cat((v_x, v_y), dim=0)
            return speed * vInf
        elif bgFlow == "parabolic":
            return torch.vstack(
                (speed * (1 - (X[N:] / 0.375) ** 2), torch.zeros_like(X[:N]))
            )  # Parabolic
        elif bgFlow == "rotation":
            r = torch.sqrt(X[:N] ** 2 + X[N:] ** 2)
            theta = torch.atan2(X[N:], X[:N])
            return speed * torch.vstack(
                (-torch.sin(theta) / r, torch.cos(theta) / r)
            )  # Rotation
        elif bgFlow == "vortex":
            chanWidth = 2.5 * 2
            return speed * torch.cat(
                [
                    torch.sin(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.cos(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                    -torch.cos(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.sin(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                ],
                dim=0,
            )

        else:
            return torch.zeros_like(X)

    return get_flow


@dataclass
class CommInfo:
    rank: int
    numProcs: int
    device: torch.device


def init_distributed():
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")


    print(f"Rank {local_rank}/{world_size} on GPU {local_rank}")
    return CommInfo(local_rank, world_size, device)

def mpi_init_distributed():
    from mpi4py import MPI
    import socket
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        if ngpu == 0:
            raise RuntimeError("CUDA is available but no GPUs were found")

        # Simple single-node mapping
        local_rank = rank % ngpu

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cpu")

    host = socket.gethostname()
    print(f"MPI rank {rank}/{world_size} on {host} using {device}")

    return CommInfo(rank, world_size, device)
