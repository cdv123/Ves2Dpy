import torch
import os
from typing import Optional
import torch.multiprocessing as mp
from parareal_vesnet import VesNetSolver

torch.set_default_dtype(torch.float32)
from tstep_biem import TStepBiem
from curve_batch_compile import Curve
import numpy as np

_worker: Optional["WorkerState"] = None


# Serial solver for parareal
class BIEMSolver:
    def __init__(self, options, params, Xwalls, initPositions):
        self.options = options
        self.params = params
        self.Xwalls = Xwalls
        print("Params", self.params)
        print("options", self.options)
        self.finalTime = self.params["T"]
        self.RS = torch.zeros(3, params["nvbd"])
        self.eta = torch.zeros(2 * params["Nbd"], params["nvbd"])

        self.oc = Curve()
        self.tt = TStepBiem(initPositions, self.Xwalls, self.options, self.params)
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
        positions = initPositions.clone()
        newSigma = sigmaStore.clone()
        if self.options["confined"]:
            self.tt.initial_confined()

        num_steps = int(self.finalTime / self.params["dt"])
        print("Number of steps:", num_steps)

        for _ in range(num_steps):
            start_time += self.params["dt"]
            positionsNew, newSigma, self.eta, self.RS, _, _ = self.tt.time_step(
                positions, newSigma, self.eta, self.RS
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


class WorkerState:
    def __init__(self, gpu_id, params, options, Xwalls, positionsShape, sigmaShape):
        torch.cuda.set_device(gpu_id)
        self.device = gpu_id
        torch.set_default_device(f"cuda:{self.device}")

        self.params = params
        self.params["viscCont"] = self.params["viscCont"].to(self.device)
        self.options = options
        self.Xwalls = Xwalls.to(device) if Xwalls else None

    def run_solver(self, initPositions, sigmaStore, file_name, start_time):
        positions = initPositions.to(self.device)
        sigma = sigmaStore.to(self.device)
        self.solver = BIEMSolver(self.options, self.params, self.Xwalls, positions)

        return self.solver.solve(positions, sigma, file_name, start_time)


# Parallel solver for parareal
# Runs multiple serial solvers independently
class ParallelSolver:
    def __init__(self, options, params, Xwalls, initPositions, sigmaStore, numCores):
        self.params = params.copy()
        self.numCores = numCores
        gpu_ids = list(range(torch.cuda.device_count()))

        if len(gpu_ids) < self.numCores:
            raise ValueError("Not enough GPUs")

        self.setupProcPool_(
            gpu_ids, params, options, Xwalls, initPositions.shape, sigmaStore.shape
        )

    def setupProcPool_(
        self,
        gpu_ids,
        params,
        options,
        Xwalls,
        positionsShape,
        sigmaShape,
    ):
        ctx = mp.get_context("spawn")
        self.gpu_queue = ctx.Queue()

        for gpu_id in gpu_ids:
            self.gpu_queue.put(gpu_id)

        self.pool = ctx.Pool(
            processes=self.numCores,
            initializer=ParallelSolver.init_worker,
            initargs=(
                self.gpu_queue,
                params,
                options,
                Xwalls,
                positionsShape,
                sigmaShape,
            ),
        )

    @staticmethod
    def init_worker(gpu_queue, params, options, Xwalls, positionsShape, sigmaShape):
        gpu_id = gpu_queue.get()
        global _worker
        _worker = WorkerState(
            gpu_id, params, options, Xwalls, positionsShape, sigmaShape
        )

    @staticmethod
    def run_worker(initPositions, sigmaStore, file_name=None, start_time=0):
        global _worker
        outPos, outSigma =  _worker.run_solver(
            initPositions, sigmaStore, file_name, start_time
        )

        return outPos.cpu(), outSigma.cpu()

    def _shutdown_pool(self):
        if self.pool is None:
            return

        self.pool.terminate()
        self.pool.join()
        self.pool = None

    def solve(
        self,
        positions: torch.Tensor,
        positionsPrime: torch.Tensor,
        sigmaStore: torch.Tensor,
        sigmaStorePrime: torch.Tensor,
        numCores: int,
        file_name: str = None,
    ):
        self.device = positions.device
        inputs = [positions[i - 1].cpu() for i in range(1, numCores + 1)]
        if file_name is not None:
            file_names = [f"file_name_{i}" for i in range(numCores)]
            start_times = [self.params["T"] * i for i in range(numCores)]
        try:
            if file_name is not None:
                results = self.pool.starmap(
                    ParallelSolver.run_worker,
                    zip(inputs, sigmaStore, file_names, start_times),
                )
            else:
                # results = self.pool.map(ParallelSolver.run_worker, inputs)
                results = self.pool.starmap(
                    ParallelSolver.run_worker, zip(inputs, sigmaStore)
                )
        except:
            self._shutdown_pool()
            raise

        for i, (positionsOut, sigmaOut) in enumerate(results, start=1):
            positionsPrime[i] = positionsOut.to(self.device)
            sigmaStorePrime[i] = sigmaOut.to(self.device)

        if file_name is not None:
            for file in file_names:
                CHUNK_SIZE = 1024 * 1024
                print("Writing to file")

                with open(file, "rb") as infile, open(file_name, "ab") as outfile:
                    while True:
                        chunk = infile.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        outfile.write(chunk)
                os.remove(file)

        return positionsPrime, sigmaStorePrime

    def close(self):
        self.pool.close()
        self.pool.join()
