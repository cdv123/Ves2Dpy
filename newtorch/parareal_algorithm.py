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
        file_name=None,
    ) -> torch.Tensor:
        self.endTime = endTime
        self.numCores = numCores
        self.initVesicles = initVesicles

        self.parallelCorrectionsSigma: torch.Tensor = torch.empty(
            self.numCores + 1, *sigma.shape, device=sigma.device
        )

        coarseSolutions, self.coarseSigma = self.initSerialSweep(
            self.initVesicles, sigma
        )
        newCoarseSolutions: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
        )
        latestVesicles = coarseSolutions.clone()

        self.newCoarseSigma: torch.Tensor = torch.zeros(
            self.numCores + 1, *sigma.shape, device=sigma.device
        )
        self.latestSigma: torch.Tensor = self.coarseSigma.clone()

        parallelCorrections: torch.Tensor = torch.empty(
            self.numCores + 1, *self.initVesicles.shape, device=initVesicles.device
        )

        for k in range(pararealIter):
            print("Parareal Iteration: ", k)
            parallelCorrections, self.parallelCorrectionsSigma = self.parallelSweep(
                latestVesicles, parallelCorrections
            )

            self.serialSweepCorrection(
                latestVesicles,
                coarseSolutions,
                newCoarseSolutions,
                parallelCorrections,
            )

            newCoarseSolutions, coarseSolutions = coarseSolutions, newCoarseSolutions
            self.newCoarseSigma, self.coarseSigma = (
                self.coarseSigma,
                self.newCoarseSigma,
            )

        if file_name is not None:
            print("Writing solution")
            self.parallelSweep(latestVesicles, parallelCorrections, file_name)

        return latestVesicles

    def initSerialSweep(
        self, initCondition: torch.Tensor, initSigma: torch.Tensor
    ) -> torch.Tensor:
        print("Starting initial Serial Sweep")
        vesicleTimeSteps: torch.Tensor = torch.empty(
            self.numCores + 1,
            *initCondition.shape,
            device=initCondition.device,
            dtype=torch.float32,
        )

        sigmaTimeSteps: torch.Tensor = torch.empty(
            self.numCores + 1,
            *initSigma.shape,
            device=initSigma.device,
            dtype=initSigma.dtype,
        )

        vesicleTimeSteps[0] = initCondition
        sigmaTimeSteps[0] = initSigma

        for i in range(self.numCores):
            print(f"Iteration {i} in initial sweep")
            vesicleTimeSteps[i + 1], sigmaTimeSteps[i + 1] = self.coarseSolver.solve(
                vesicleTimeSteps[i], sigmaTimeSteps[i]
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
            newCoarse[n], self.newCoarseSigma[n] = self.coarseSolver.solve(
                latestVesicles[n - 1], self.latestSigma[n - 1]
            )
            latestVesicles[n] = newCoarse[n] + parallelCorrection[n] - previousCoarse[n]
            self.latestSigma[n] = (
                self.newCoarseSigma[n]
                + self.parallelCorrectionsSigma[n]
                - self.coarseSigma[n]
            )

    def parallelSweep(
        self, inputVesicles: torch.Tensor, newVesicles: torch.Tensor, file_name=None
    ) -> torch.Tensor:
        print("Starting parallel sweep")
        return self.parallelSolver.solve(
            inputVesicles, newVesicles, self.numCores, file_name
        )
