import torch
import os
from typing import Optional

torch.set_default_dtype(torch.float32)
from distributed_tstep_biem import TStepBiem
from curve_batch_compile import Curve
import numpy as np
from torch import distributed as dist
from tqdm import tqdm


class BIEMSolver:
    def __init__(
        self, options, params, Xwalls, initPositions, comm_info, nv, new_num_ranks
    ):
        self.options = options
        self.params = params
        self.Xwalls = Xwalls
        self.finalTime = self.params["T"]
        self.rank = comm_info.rank
        self.device = comm_info.device
        self.new_num_ranks = new_num_ranks

        if new_num_ranks == 1:
            sub_ranks = [self.rank]
        else:
            sub_ranks = list(range(new_num_ranks))

        self.sub_group = dist.new_group(ranks=sub_ranks)
        if self.rank not in sub_ranks:
            self.active = False
            return

        self.active = True
        self.sub_rank = sub_ranks.index(self.rank)
        # broadcast src must be a GLOBAL rank, even with a process group.
        self.group_root = sub_ranks[0]
        torch.set_default_device(self.device)
        self.params["viscCont"] = self.params["viscCont"].to(self.device)
        self.RS = torch.zeros(
            3, params["nvbd"], device=self.device, dtype=initPositions.dtype
        )
        self.eta = torch.zeros(
            2 * params["Nbd"],
            params["nvbd"],
            device=self.device,
            dtype=initPositions.dtype,
        )

        self.oc = Curve()
        self.tt = TStepBiem(
            initPositions,
            self.Xwalls,
            self.options,
            self.params,
            rank=self.sub_rank,
            size=new_num_ranks,
            device=self.device,
            group=self.sub_group,
        )
        _, self.area0, self.len0 = self.oc.geomProp(initPositions)
        self.modes = torch.concatenate(
            (torch.arange(0, params["N"] // 2), torch.arange(-params["N"] // 2, 0))
        ).to(initPositions.device)

    def solve(
        self,
        initPositions: torch.Tensor,
        sigmaStore: torch.Tensor,
        out_file_name=None,
        start_time=0,
    ):
        torch.set_default_dtype(torch.float64)
        positions = initPositions.clone().to(self.device)
        # newSigma = torch.zeros_like(sigmaStore, dtype=torch.float64)
        newSigma = sigmaStore.clone().to(self.device)

        if not self.active:
            return positions, newSigma

        dist.broadcast(positions, src=self.group_root, group=self.sub_group)
        dist.broadcast(newSigma, src=self.group_root, group=self.sub_group)

        if self.options["confined"]:
            self.tt.initial_confined()

        num_steps = int(self.finalTime / self.params["dt"])
        print("Number of steps:", num_steps)

        for _ in tqdm(range(num_steps)):
            start_time += self.params["dt"]
            positionsNew, newSigma, self.eta, self.RS, _, _ = self.tt.time_step(
                positions, newSigma, self.eta, self.RS
            )
            self.eta = torch.zeros(
                2 * self.params["Nbd"],
                self.params["nvbd"],
                device=self.device,
                dtype=positions.dtype,
            )
            self.RS = torch.zeros(
                3, self.params["nvbd"], device=self.device, dtype=positions.dtype
            )
            if self.options["reparameterization"]:
                # Redistribute arc-length
                positions0 = positionsNew.clone()
                for _ in range(5):
                    positionsNew, allGood = self.oc.redistributeArcLength(
                        positionsNew, self.modes
                    )
                positions = self.oc.alignCenterAngle(positions0, positionsNew)
            else:
                positions = positionsNew

            if self.options["correctShape"]:
                positions = self.oc.correctAreaAndLengthAugLag(
                    positions.float(), self.area0, self.len0
                )

            if out_file_name is not None:
                output = np.concatenate(
                    ([start_time], positions.cpu().numpy().T.flatten())
                ).astype("float64")

                with open(out_file_name, "ab") as fid:
                    output.tofile(fid)

        return positions, newSigma


# Parallel solver for parareal
# Runs multiple serial solvers independently
class ParallelSolver:
    def __init__(
        self,
        options,
        params,
        Xwalls,
        numCores,
        rank,
        device,
    ):
        self.params = params.copy()
        self.device = device
        self.params["viscCont"] = self.params["viscCont"].to(self.device)
        self.numCores = numCores
        self.rank = rank
        self.options = options
        self.Xwalls = Xwalls
        self.nv = params["nv"]

    def run_solver(self, positions, sigmaStore, file_name, start_time):
        class _CommInfo:
            pass

        comm_info = _CommInfo()
        comm_info.rank = self.rank
        comm_info.device = self.device

        self.solver = BIEMSolver(
            self.options, self.params, self.Xwalls, positions, comm_info, self.nv, 1
        )

        return self.solver.solve(positions, sigmaStore, file_name, start_time)

    def solve(
        self,
        all_positions: torch.Tensor,
        all_sigma: torch.Tensor,
        all_positions_prime: torch.Tensor,
        all_sigma_prime: torch.Tensor,
        positions: torch.Tensor,
        sigmaStore: torch.Tensor,
        numCores: int,
        original_file_name: Optional[str] = None,
    ):
        torch.set_default_dtype(torch.float64)
        if self.rank == 0:
            pos_scatter_list = [all_positions[i].contiguous() for i in range(numCores)]
            sigma_scatter_list = [all_sigma[i].contiguous() for i in range(numCores)]
        else:
            pos_scatter_list = None
            sigma_scatter_list = None

        dist.scatter(tensor=positions, scatter_list=pos_scatter_list, src=0)
        dist.scatter(tensor=sigmaStore, scatter_list=sigma_scatter_list, src=0)

        file_name = f"{original_file_name}_{self.rank}"
        start_time = self.params["T"] * self.rank

        print(f"Rank {self.rank} Sigma dtype {sigmaStore.dtype}")
        print(f"Rank {self.rank} Positions dtype {positions.dtype}")
        print(f"Rank {self.rank} Default dtype {torch.get_default_dtype()}")

        positions, sigmaStore = self.run_solver(
            positions, sigmaStore, file_name, start_time
        )

        positions = positions.contiguous()
        sigmaStore = sigmaStore.contiguous()

        print(f"Positions shape", positions.shape)
        print(f"sigmaStore shape", sigmaStore.shape)

        if self.rank == 0:
            pos_gather_list = [
                all_positions_prime[i].contiguous() for i in range(numCores)
            ]
            sigma_gather_list = [
                all_sigma_prime[i].contiguous() for i in range(numCores)
            ]
        else:
            pos_gather_list = None
            sigma_gather_list = None

        dist.barrier()

        print(f"Rank {self.rank} Sigma dtype {sigmaStore.dtype}")
        print(f"Rank {self.rank} Positions dtype {positions.dtype}")

        if self.rank == 0:
            print(f"Rank {self.rank} Sigma prime dtype {all_sigma_prime.dtype}")
            print(f"Rank {self.rank} Positions prime dtype {all_positions_prime.dtype}")

        dist.gather(tensor=positions, gather_list=pos_gather_list, dst=0)
        dist.gather(tensor=sigmaStore, gather_list=sigma_gather_list, dst=0)

        if self.rank == 0:
            pos_gather_list = [all_positions[0]] + pos_gather_list
            sigma_gather_list = [all_sigma[0]] + sigma_gather_list
            all_positions_prime = torch.stack(pos_gather_list, dim=0)
            all_sigma_prime = torch.stack(sigma_gather_list, dim=0)

        dist.barrier()
        if self.rank == 0:
            file_names = [f"{original_file_name}_{i}" for i in range(numCores)]

            if original_file_name is not None:
                for file in file_names:
                    CHUNK_SIZE = 1024 * 1024
                    print("Writing to file")

                    with (
                        open(file, "rb") as infile,
                        open(original_file_name, "ab") as outfile,
                    ):
                        while True:
                            chunk = infile.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            outfile.write(chunk)
                    os.remove(file)

        return all_positions_prime, all_sigma_prime
