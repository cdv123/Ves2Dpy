import numpy as np
import torch
from tqdm import tqdm
from numpy.polynomial.legendre import leggauss


class SDC:
    def __init__(self, coarseSolver, smallTimeStepSolver, dt):
        self.coarseSolver = coarseSolver
        self.smallTimeStepSolver = smallTimeStepSolver
        self.dt = dt

    def get_quadrature_nodes(self, num_nodes):
        nodes, weights = leggauss(num_nodes)
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights

        return nodes, weights

    def compute_sdc_nodes_and_Q(self, M):
        nodes, _ = leggauss(M)
        nodes = 0.5 * (nodes + 1)

        Q = np.zeros((M, M))

        for j in range(M):
            poly = np.poly1d([1.0])
            denom = 1.0

            for k in range(M):
                if k != j:
                    poly *= np.poly1d([1, -nodes[k]])
                    denom *= nodes[j] - nodes[k]

            poly /= denom

            int_poly = np.polyint(poly)

            for m in range(M):
                Q[m, j] = int_poly(nodes[m]) - int_poly(0)

        return nodes, Q

    def time_step(
        self,
        initPositions,
        num_nodes,
        solutionsStore,
        velocities,
        sigmaStore,
        num_iter,
        correctionStore,
        nodes,
        Q,
    ):
        solutionsStore[0] = initPositions

        tau_prev = 0

        for i in range(1, num_nodes + 1):
            tau_i = nodes[i - 1]
            dt_i = self.dt * (tau_i - tau_prev)

            self.coarseSolver.dt = dt_i
            solutionsStore[i] = self.coarseSolver.time_step(
                solutionsStore[i - 1], sigmaStore[i - 1]
            )

            tau_prev = tau_i

        for i in range(num_nodes + 1):
            velocities[i] = self.smallTimeStepSolver.get_velocity(solutionsStore[i])

        old_velocities = velocities.clone()

        for i in range(num_iter):
            tau_prev = 0
            for m in range(1, num_nodes + 1):
                tau_i = nodes[m - 1]
                dt_m = self.dt * (tau_i - tau_prev)
                self.coarseSolver.dt = dt_m
                correctionStore = torch.zeros_like(initPositions)
                self.computeIntegral(Q, old_velocities, correctionStore, m, num_nodes)

                correctionStore -= dt_m * old_velocities[m - 1]
                solutionsStore[m] = self.coarseSolver.time_step(
                    solutionsStore[m - 1], correction=correctionStore
                )

                velocities[m] = self.smallTimeStepSolver.get_velocity(solutionsStore[m])
                tau_prev = tau_i

            old_velocities = velocities.clone()

        return solutionsStore[-1]

    def computeIntegral(self, quadWeights, velocities, integralStore, m, num_nodes):
        for i in range(num_nodes + 1):
            integralStore += (
                self.dt * (quadWeights[m][i] - quadWeights[m - 1][i]) * velocities[i]
            )

    def sweep(
        self,
        initPositions,
        sigma,
        eta,
        RS,
        finalTime,
        modes,
        num_nodes,
        num_iter,
        options,
    ):
        nodes, Q = self.compute_sdc_nodes_and_Q(num_nodes)
        Xnew = initPositions

        solutionsStore = torch.zeros(
            num_nodes + 1,
            *initPositions.shape,
            device=initPositions.device,
            dtype=initPositions.dtype,
        )

        velocitiesStore = torch.zeros(
            num_nodes + 1,
            *initPositions.shape,
            device=initPositions.device,
            dtype=initPositions.dtype,
        )

        correctionsStore = torch.zeros(
            num_nodes + 1,
            *initPositions.shape,
            device=initPositions.device,
            dtype=initPositions.dtype,
        )

        sigmaStore = torch.zeros(
            num_nodes + 1, *sigma.shape, device=sigma.device, dtype=sigma.dtype
        )

        for step in tqdm(range(finalTime / self.dt)):
            Xnew = self.time_step(
                Xnew,
                num_nodes,
                solutionsStore,
                velocitiesStore,
                sigmaStore,
                num_iter,
                correctionsStore,
                nodes,
                Q,
            )

            sigmaStore = torch.zeros(
                num_nodes + 1,
                *initPositions.shape,
                device=initPositions.device,
                dtype=initPositions.dtype,
            )
