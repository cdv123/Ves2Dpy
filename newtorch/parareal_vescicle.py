import torch


# Serial solver for parareal
class CoarseSolver:
    def solve(self, positions: torch.Tensor, sigmaStore: torch.Tensor):
        pass


# Parallel solver for parareal
class FineSolver:
    def solve(
        self,
        positions: torch.Tensor,
        sigmaStore: torch.Tensor,
        timeStepSize: float,
        numCores: int,
    ):
        pass


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
                )
                - self.coarseSolver.solve(
                    baseVesicles[n - 1], self.sigmaStore[n - 1], timeStepSize
                )
            )

    def parallelSweep(
        self,
        inputVesicles: torch.Tensor,
        timeStepSize: float,
    ) -> torch.Tensor:
        return self.fineSolver.solve(
            inputVesicles, self.sigmaStore, timeStepSize, self.numCores
        )
