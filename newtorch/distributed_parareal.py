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
        comm_info,
        file_name=None,
    ) -> torch.Tensor:
        self.endTime = endTime
        self.numCores = numCores
        self.initVesicles = initVesicles
        self.comm_info = comm_info

        self.parallelCorrectionsSigma: torch.Tensor = torch.empty(
            self.numCores + 1, *sigma.shape, device=comm_info.device, dtype=sigma.dtype
        )

        coarseSolutions, self.coarseSigma = self.initSerialSweep(
            self.initVesicles, sigma
        )
        newCoarseSolutions: torch.Tensor = torch.empty(
            self.numCores + 1,
            *self.initVesicles.shape,
            device=comm_info.device,
            dtype=initVesicles.dtype,
        )
        latestVesicles = coarseSolutions.clone()
        print("Latest vesicles shape", latestVesicles.shape)

        if comm_info.rank == 0:
            print("Latest Vesicles shape ", latestVesicles.shape)

        self.newCoarseSigma: torch.Tensor = torch.zeros(
            self.numCores + 1, *sigma.shape, device=comm_info.device, dtype=sigma.dtype
        )
        self.latestSigma: torch.Tensor = self.coarseSigma.clone()

        parallelCorrections: torch.Tensor = torch.empty(
            self.numCores + 1,
            *self.initVesicles.shape,
            device=comm_info.device,
            dtype=self.initVesicles.dtype,
        )

        for _ in range(pararealIter):
            parallelCorrections, self.parallelCorrectionsSigma = self.parallelSweep(
                latestVesicles, parallelCorrections
            )

            self.serialSweepCorrection(
                latestVesicles,
                coarseSolutions,
                newCoarseSolutions,
                parallelCorrections,
            )

            newCoarseSolutions, coarseSolutions = (
                coarseSolutions,
                newCoarseSolutions,
            )
            self.newCoarseSigma, self.coarseSigma = (
                self.coarseSigma,
                self.newCoarseSigma,
            )

        if file_name is not None:
            print("Writing solution")
            latestVesicles, self.parallelCorrectionsSigma = self.parallelSweep(
                latestVesicles, parallelCorrections, file_name
            )

        return latestVesicles[-1]

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
            vesicleTimeSteps[i + 1], sigmaTimeSteps[i + 1] = self.coarseSolver.solve(
                vesicleTimeSteps[i], sigmaTimeSteps[i]
            )

        return vesicleTimeSteps, sigmaTimeSteps

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
            inputVesicles,
            self.latestSigma,
            newVesicles,
            self.parallelCorrectionsSigma,
            inputVesicles[0].clone(),
            self.latestSigma[0].clone(),
            self.numCores,
            file_name,
        )
