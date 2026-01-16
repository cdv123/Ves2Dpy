import torch
from typing import Optional
import torch.multiprocessing as mp

# TODO add seperate prints for each level to make it clearer

torch.set_default_dtype(torch.float32)
from tstep_biem import TStepBiem
from curve_batch_compile import Curve

_worker: Optional["WorkerState"] = None


# Serial solver for parareal
class BIEMSolver:
    def __init__(self, options, params, Xwalls, initPositions):
        self.options = options
        self.params = params
        self.Xwalls = Xwalls
        self.finalTime = self.params["T"]
        self.RS = torch.zeros(3, params["nvbd"])
        self.eta = torch.zeros(2 * params["Nbd"], params["nvbd"])

        self.oc = Curve()
        self.tt = TStepBiem(initPositions, self.Xwalls, self.options, self.params)
        _, self.area0, self.len0 = self.oc.geomProp(initPositions)
        self.modes = torch.concatenate(
            (torch.arange(0, params["N"] // 2), torch.arange(-params["N"] // 2, 0))
        ).to(initPositions.device)

        print("Params:", self.params)

    def solve(self, initPositions: torch.Tensor, sigmaStore: torch.Tensor):
        print("Params:", self.params)
        positions = initPositions.clone()
        if self.options["confined"]:
            self.tt.initial_confined()

        num_steps = int(self.finalTime / self.params["dt"])
        print("Number of steps:", num_steps)

        for i in range(num_steps):
            print("Simple solver step:", i)
            positionsNew, sigmaStore, self.eta, self.RS, _, _ = self.tt.time_step(
                positions, sigmaStore, self.eta, self.RS
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

        return positions


class WorkerState:
    def __init__(self, gpu_id, params, options, Xwalls, positionsShape, sigmaShape):
        torch.cuda.set_device(gpu_id)
        self.device = gpu_id

        self.sigmaStore = torch.Tensor(sigmaShape)
        self.positions = torch.Tensor(positionsShape)
        self.params = params
        self.options = options
        self.Xwalls = Xwalls

    def run_solver(self, initPositions):
        self.solver = BIEMSolver(self.options, self.params, self.Xwalls, initPositions)
        self.positions = initPositions.to(self.device)

        return self.solver.solve(self.positions, self.sigmaStore)


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
            gpu_ids, options, params, Xwalls, initPositions.shape, sigmaStore.shape
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
    def run_worker(initPositions):
        global _worker
        return _worker.run_solver(initPositions)

    def solve(
        self, positions: torch.Tensor, positionsPrime: torch.Tensor, numCores: int
    ):
        inputs = [positions[i - 1] for i in range(1, numCores + 1)]
        results = self.pool.map(ParallelSolver.run_worker, inputs)

        for i, out in enumerate(results, start=1):
            positionsPrime[i] = out

        return positionsPrime

    def close(self):
        self.pool.close()
