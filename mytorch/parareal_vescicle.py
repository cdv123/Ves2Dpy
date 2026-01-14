import torch
from functools import partial
from itertools import product, repeat

torch.set_default_dtype(torch.float32)
from tstep_biem import TStepBiem
from curve_batch_compile import Curve
import torch.multiprocessing as mp

class PararealWorker:
    def __init__(self, positions, sigmaStore, outputPositions, solver):
        self.positions = positions
        self.sigmaStore = sigmaStore
        self.outputPositions = outputPositions
        self.solver = solver

    def __call__(self, i):
        self.outputPositions[i] = self.solver.solve(
            self.positions[i], self.sigmaStore[i]
        )
            

# Serial solver for parareal
class CoarseSolver:
    def __init__(self, options, params, Xwalls, finalTime, initPositions):
        self.options = options
        params["T"] = finalTime
        self.RS = torch.zeros(3, params["nvbd"])
        self.eta = torch.zeros(2 * params["Nbd"], params["nvbd"])

        self.oc = Curve()
        self.tt = TStepBiem(initPositions, Xwalls, self.options, params)
        _, self.area0, self.len0 = self.oc.geomProp(initPositions)
        self.modes = torch.concatenate(
            (torch.arange(0, params["N"] // 2), torch.arange(-params["N"] // 2, 0))
        ).to(initPositions.device)

        print("Params:", self.params)

    def solve(self, initPositions: torch.Tensor, sigmaStore: torch.Tensor):
        positions = initPositions.clone()
        if self.options["confined"]:
            self.tt.initialConfined()

        num_steps = int(self.finalTime / self.params["dt"])

        for i in range(num_steps):
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

def run_solver_parallel(positionsPrime, positions, sigmaStore, solve_func, index):
    positionsPrime[index] = solve_func(positions[index-1], sigmaStore[index-1])

# Parallel solver for parareal
class FineSolver:
    def __init__(self, options, params, Xwalls, finalTime, initPositions):
        self.solver = CoarseSolver(options, params, Xwalls, finalTime, initPositions)

    def solve(
        self,
        positions: torch.Tensor,
        positionsPrime: torch.Tensor,
        sigmaStore: torch.Tensor,
        numCores: int,
    ):
        positions.share_memory_()
        positionsPrime.share_memory_()
        sigmaStore.share_memory_()
        procs = []
        # Parallelize this
        for i in range(1, numCores + 1):
            positionsPrime[i] = self.solver.solve(positions[i-1], sigmaStore[i-1])

        return positionsPrime


class PararealSolver:
    def __init__(self, fineSolver, coarseSolver):
        self.fineSolver = fineSolver
        self.coarseSolver = coarseSolver

    def pararealSolve(
        self,
        initVesicles: torch.Tensor,
        sigma: torch.Tensor,
        numCores: int,
        endTime: float,
        pararealIter: int,
    ) -> torch.Tensor:
        self.endTime = endTime
        self.numCores = numCores
        self.initVesicles = initVesicles

        self.sigmaStore: torch.Tensor = torch.empty(
            self.numCores + 1, *sigma.shape, device=sigma.device
        )

        coarseSolutions: torch.Tensor = self.initSerialSweep(self.initVesicles)
        newCoarseSolutions: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device, dtype=torch.float32
        )
        latestVesicles = coarseSolutions.clone()

        parallelCorrections: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device, dtype=torch.float32
        )

        for k in range(pararealIter):
            print("Parareal Iteration: ", k)
            parallelCorrections = self.parallelSweep(
                    latestVesicles, parallelCorrections
            )

            self.serialSweepCorrection(
                latestVesicles,
                coarseSolutions,
                newCoarseSolutions,
                parallelCorrections,
            )

            newCoarseSolutions, coarseSolutions = coarseSolutions, newCoarseSolutions

        return latestVesicles

    def initSerialSweep(
        self,
        initCondition: torch.Tensor,
    ) -> torch.Tensor:
        print("Starting initial Serial Sweep")
        vesicleTimeSteps: torch.Tensor = torch.empty(
            self.numCores + 1,
            *initCondition.shape,
            device=initCondition.device,
            dtype=torch.float32,
        )

        vesicleTimeSteps[0] = initCondition

        for i in range(self.numCores):
            print(f"Iteration {i} in initial sweep")
            vesicleTimeSteps[i + 1] = self.coarseSolver.solve(
                vesicleTimeSteps[i], self.sigmaStore[i]
            )

        return vesicleTimeSteps

    def serialSweepCorrection(
        self,
        latestVesicles: torch.Tensor,
        previousCoarse: torch.Tensor,
        newCoarse: torch.Tensor,
        parallelCorrection: torch.Tensor,
    ):
        print("Starting serial Sweep Correction")
        for n in range(1, self.numCores + 1):
            newCoarse[n] = self.coarseSolver.solve(
                latestVesicles[n - 1], self.sigmaStore[n - 1]
            )
            latestVesicles[n] = newCoarse[n] + parallelCorrection[n] - previousCoarse[n]

    def parallelSweep(
        self,
        inputVesicles: torch.Tensor,
        newVesicles: torch.Tensor) -> torch.Tensor:
        print("Starting parallel sweep")
        return self.fineSolver.solve(
            inputVesicles,
            newVesicles,
            self.sigmaStore,
            self.numCores
        )

