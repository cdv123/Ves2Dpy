import torch


class PararealSolver:
    def __init__(self, parallelSolver, coarseSolver):
        self.parallelSolver = parallelSolver
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
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
        )
        latestVesicles = coarseSolutions.clone()

        latestVesicles: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
        )

        parallelCorrections: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
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

        self.parallelSolver.close()

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
        newVesicles: torch.Tensor,
    ) -> torch.Tensor:
        print("Starting parallel sweep")
        return self.parallelSolver.solve(
            inputVesicles, newVesicles, self.sigmaStore, self.numCores
        )
