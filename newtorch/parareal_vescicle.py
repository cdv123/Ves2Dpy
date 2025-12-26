import torch
from tstep_biem import TStepBiem
from curve_batch_compile import Curve


# Serial solver for parareal
class CoarseSolver:
    def __init__(self, options, params, Xwalls, finalTime, initPositions): 
        self.options = options
        self.params = params
        self.Xwalls = Xwalls
        self.finalTime = finalTime
        self.RS = torch.zeros(3, params['nvbd'])
        self.eta = torch.zeros(2 * params['Nbd'], params['nvbd'])

        self.oc = Curve()
        self.tt = TStepBiem(initPositions, self.Xwalls, self.options, self.params)
        _, self.area0, self.len0 = self.oc.geomProp(initPositions)
        self.modes = torch.concatenate((torch.arange(0, params['N'] // 2), 
                                        torch.arange(-params['N'] // 2, 0))).to(initPositions.device) 

    def solve(self, positions: torch.Tensor, sigmaStore: torch.Tensor):
        if self.options['confined']:
            self.tt.initialConfined()

        num_steps = int(self.finalTime/self.params["dt"])

        for _ in range(num_steps):
            positionsNew, sigmaStore, self.eta, self.RS, _, _ = self.tt.time_step(positions, sigmaStore, self.eta, self.RS) 
            if self.options['reparameterization']:
                # Redistribute arc-length
                positions0 = positionsNew.clone()
                for _ in range(5):
                    positionsNew, allGood = self.oc.redistributeArcLength(positionsNew, self.modes)
                positions = self.oc.alignCenterAngle(positions0, positionsNew)
            else:
                positions = positionsNew


            if self.options['correctShape']:
                positions = self.oc.correctAreaAndLengthAugLag(positions.float(), self.area0, self.len0)

        return positions




# Parallel solver for parareal
class FineSolver:
    def __init__(self, options, params, Xwalls, finalTime, initPositions): 
        options["dt"]/= 10
        self.solver = CoarseSolver(options, params, Xwalls, finalTime, initPositions)
    def solve(
        self,
        positions: torch.Tensor,
        sigmaStore: torch.Tensor,
        timeStepSize: float,
        numCores: int,
    ):
        # Parallelize this
        for i in range(numCores):
            positions[i] = self.solver.solve(positions[i], sigmaStore[i])

        return positions

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

        baseVesicles: torch.Tensor = self.initSerialSweep(
            self.initVesicles, self.endTime
        )

        timeStepSize = self.endTime / numCores

        latestVesicles: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
        )

        self.sigmaStore: torch.Tensor = torch.empty(
            self.numCores + 1, *sigma.shape, device=sigma.device
        )

        parallelCorrections: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
        )

        for k in range(pararealIter):
            parallelCorrections = self.parallelSweep(baseVesicles, timeStepSize)

            latestVesicles[k] = parallelCorrections[k]

            latestVesicles = self.serialSweepCorrection(
                baseVesicles,
                latestVesicles,
                parallelCorrections,
                timeStepSize,
                k,
            )

        return latestVesicles

    def initSerialSweep(
        self,
        initCondition: torch.Tensor,
        timeStepSize: float,
    ) -> torch.Tensor:
        vesicleTimeSteps: torch.Tensor = torch.empty(
            self.numCores + 1, *initCondition.shape, device=initCondition.device
        )

        vesicleTimeSteps[0] = initCondition

        for i in range(self.numCores + 1):
            vesicleTimeSteps[i + 1] = self.coarseSolver.solve(
                vesicleTimeSteps[i], self.sigmaStore[i], timeStepSize
            )

        return vesicleTimeSteps

    def serialSweepCorrection(
        self,
        baseVesicles: torch.Tensor,
        latestVesicles: torch.Tensor,
        parallelCorrection: torch.Tensor,
        timeStepSize: float,
        kIter: int,
    ):
        for n in range(kIter + 1, self.numCores + 1):
            latestVesicles[n] = parallelCorrection[n] + (
                self.coarseSolver.solve(
                    latestVesicles[n - 1], self.sigmaStore[n - 1], timeStepSize
                )[0]
                - self.coarseSolver.solve(
                    baseVesicles[n - 1], self.sigmaStore[n - 1], timeStepSize
                )[0]
            )

    def parallelSweep(
        self,
        inputVesicles: torch.Tensor,
        timeStepSize: float,
    ) -> torch.Tensor:
        return self.fineSolver.solve(
            inputVesicles, self.sigmaStore, timeStepSize, self.numCores
        )[0]
