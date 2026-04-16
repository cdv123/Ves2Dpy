import math
import numpy as np
import torch
from torch import distributed as dist
from mpi4py import MPI
from petsc4py import PETSc

from curve_batch_compile import Curve
from capsules import capsules
from poten import Poten
from tools.filter import interpft_vec
from biem_support import dist_wrapper_allExactStokesSLTarget_compare2, naiveNearZoneInfo


torch.set_default_dtype(torch.float64)


def check_finite(name, x, rank):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        xmin = (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).min().item()
            if x.numel()
            else 0.0
        )
        xmax = (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).max().item()
            if x.numel()
            else 0.0
        )
        raise RuntimeError(
            f"[rank {rank}] {name} has non-finite values; "
            f"bad={bad}, shape={tuple(x.shape)}, min={xmin}, max={xmax}"
        )


class _PetscMatCtx:
    """Matrix-free operator for PETSc KSP.

    Global unknown ordering is kept identical to the current code:
        torch.cat((X, sigma), dim=0).T.reshape(-1)
    i.e. vesicle-major blocks of size 3*N.
    """

    def __init__(self, parent, vesicle, vesicle_local):
        self.parent = parent
        self.vesicle = vesicle
        self.vesicle_local = vesicle_local
        self.comm = MPI.COMM_WORLD
        self.global_size = 3 * vesicle.N * vesicle.nv
        self.local_size = 3 * vesicle.N * parent.chunk

    def createVecs(self, A):
        x = PETSc.Vec().createMPI(
            size=(self.local_size, self.global_size), comm=A.comm
        )
        y = x.duplicate()
        try:
            x.setType(PETSc.Vec.Type.MPICUDA)
            y.setType(PETSc.Vec.Type.MPICUDA)
        except PETSc.Error:
            # Keep running if PETSc was not built with CUDA vector types.
            pass
        return x, y

    def mult(self, A, x, y):
        y_arr = self.parent.time_matvec_petsc(x, self.vesicle, self.vesicle_local)
        y.setArray(y_arr)


class _PetscPCCtx:
    """Python PC applying the existing block-diagonal LU preconditioner locally."""

    def __init__(self, parent):
        self.parent = parent

    def apply(self, pc, x, y):
        y_arr = self.parent.preconditionerBD_petsc(x)
        y.setArray(y_arr)


class TStepBiem:
    """
    PETSc/CUDA rewrite of the Krylov solve path.

    The vesicle physics, near-singular corrections, ordering of unknowns,
    and local block-diagonal preconditioner are preserved. The main change is
    that GMRES is now managed by PETSc with a matrix-free MATPYTHON operator and
    a PCPYTHON preconditioner.
    """

    def __init__(self, X, Xwalls, options, prams, rank, size, device):
        oc = Curve()

        self.Xwalls = Xwalls
        _, self.area, self.length = oc.geomProp(X)

        self.dt = prams["dt"]
        self.rank = rank
        self.size = size
        self.nv = X.shape[1]
        if self.nv % size != 0:
            raise ValueError(
                f"nv={self.nv} must be divisible by num ranks={size} with the current decomposition"
            )
        self.chunk = self.nv // size

        self.start = rank * self.chunk
        self.end = (rank + 1) * self.chunk
        self.device = device
        self.comm = PETSc.COMM_WORLD
        self.mpi_comm = MPI.COMM_WORLD

        self.currentTime = 0.0
        self.finalTime = prams["T"]
        self.kappa = prams["kappa"]

        self.viscCont = prams["viscCont"]
        self.viscCont_local = self.viscCont[self.start : self.end].to(self.device)

        self.gmresTol = prams["gmresTol"]
        self.gmresMaxIter = prams["gmresMaxIter"]
        self.gmresRestart = prams.get("gmresRestart", 20)

        self.farField = lambda X_: self.bg_flow(
            X_,
            self.Xwalls,
            options["farField"],
            Speed=prams["farFieldSpeed"],
            chanWidth=prams["chanWidth"],
            vortexSize=prams["vortexSize"],
        )

        self.confined = self.Xwalls is not None
        self.areaLenTol = prams["areaLenTol"]

        self.repulsion = options["repulsion"]
        if prams["nv"] == 1 and not self.confined:
            self.repulsion = False

        self.repStrength = prams["repStrength"]
        self.minDist = prams["minDist"]
        self.op = Poten(prams["N"])

        self.usePreco = options["usePreco"]
        self.bdiagVes = None

        if self.confined:
            self.initial_confined()
            self.eta = None
            self.RS = None
            self.matFreeWalls = options["matFreeWalls"]
        else:
            self.opWall = None

        self.bdiagTen = None
        self.bdiagWall = None
        self.wallDLP = None
        self.wallN0 = None
        self.wallDLPandRSmat = None
        self.haveWallMats = False
        self.Galpert = None
        self.Galpert_local = None
        self.D = None
        self.lapDLP = None
        self.DLPnoCorr = None
        self.SLPnoCorr = None
        self.NearV2V = None
        self.NearW2V = None
        self.NearV2W = None
        self.NearW2W = None
        self.invM11 = None
        self.invM22 = None
        self._matvec_calls = 0

    def initial_confined(self):
        Nbd = self.Xwalls.shape[0] // 2
        nvbd = self.Xwalls.shape[1]
        self.opWall = Poten(Nbd)
        uwalls = self.farField([])
        self.walls = capsules(
            self.Xwalls, None, uwalls, torch.zeros(nvbd, 1), torch.zeros(nvbd, 1)
        )
        self.wallDLP = self.opWall.stokesDLmatrix(self.walls)
        self.wallN0 = self.opWall.StokesN0Matrix(self.walls)
        self.bdiagWall = self.walls.wallsPrecond()

    def _torch_local_from_petsc(self, vec: PETSc.Vec) -> torch.Tensor:
        arr = vec.getArray(readonly=True)
        return torch.from_numpy(np.asarray(arr)).to(self.device, dtype=torch.float64)

    def _allgather_global_density(self, f_local: torch.Tensor) -> torch.Tensor:
        """Gather only the source density needed for off-rank interactions.

        With the current decomposition, the matrix-free operator is local in the
        unknown block (Xm, sigma) for traction/divergence evaluation. The only
        quantity that must be made global for vesicle-vesicle interactions is the
        traction jump density f.
        """
        parts = [torch.zeros_like(f_local) for _ in range(self.size)]
        dist.all_gather(parts, f_local.contiguous())
        return torch.cat(parts, dim=1)

    def _make_global_rhs_vec(self, rhs_local: torch.Tensor, vec_type=None) -> PETSc.Vec:
        rhs_np = rhs_local.detach().cpu().numpy().astype(np.float64, copy=False)
        b = PETSc.Vec().createMPI(
            size=(rhs_np.size, rhs_np.size * self.size), comm=self.comm
        )
        if vec_type is not None:
            try:
                b.setType(vec_type)
            except PETSc.Error:
                pass
        b.setArray(rhs_np)
        return b

    def _make_initial_guess_vec(self, x_local: torch.Tensor, template: PETSc.Vec) -> PETSc.Vec:
        x0_np = x_local.detach().cpu().numpy().astype(np.float64, copy=False)
        x0 = template.duplicate()
        x0.setArray(x0_np)
        return x0

    def _build_block_diag_preconditioner(self, vesicle, vesicle_local, N, alpha_local):
        Ben_local, Ten_local, Div_local = vesicle_local.computeDerivs()

        I = torch.eye(2 * N, device=self.device, dtype=torch.float64).unsqueeze(-1).repeat(
            1, 1, self.chunk
        )
        Z = torch.zeros((N, N, self.chunk), device=self.device, dtype=torch.float64)

        G_Ben_local = torch.matmul(
            self.Galpert_local.permute(2, 0, 1), Ben_local.permute(2, 0, 1)
        ).permute(1, 2, 0)
        G_Ten_local = torch.matmul(
            self.Galpert_local.permute(2, 0, 1), Ten_local.permute(2, 0, 1)
        ).permute(1, 2, 0)
        DivZ = torch.cat([Div_local, Z], dim=1)

        alpha_inv = 1.0 / alpha_local.view(1, 1, self.chunk)
        top_left = I + self.dt * vesicle.kappa * G_Ben_local
        top_right = -self.dt * G_Ten_local * alpha_inv
        top = torch.cat([top_left, top_right], dim=1)
        mat_all = torch.cat([top, DivZ], dim=0)

        LU, P = torch.linalg.lu_factor(mat_all.permute(2, 0, 1))
        self.bdiagVes = {"LU": LU, "pivots": P}

    def _solve_linear_system_petsc(self, rhs_local, init_local, vesicle, vesicle_local):
        local_size = rhs_local.numel()
        global_size = local_size * self.size

        A = PETSc.Mat().create(comm=self.comm)
        A.setSizes(((local_size, global_size), (local_size, global_size)))
        A.setType(PETSc.Mat.Type.PYTHON)
        A.setPythonContext(_PetscMatCtx(self, vesicle, vesicle_local))
        A.setUp()

        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.setTolerances(rtol=self.gmresTol, max_it=self.gmresMaxIter)
        ksp.setGMRESRestart(self.gmresRestart)
        ksp.setInitialGuessNonzero(True)
        ksp.setNormType(PETSc.KSP.NormType.PRECONDITIONED)

        if self.usePreco:
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(_PetscPCCtx(self))
        else:
            ksp.getPC().setType(PETSc.PC.Type.NONE)

        # Keep runtime overrides available, e.g. -ksp_monitor -log_view -vec_type mpicuda
        ksp.setFromOptions()

        # Match the operator vector type to CUDA when PETSc supports it.
        vec_type = None
        try:
            vec_type = PETSc.Vec.Type.MPICUDA
        except AttributeError:
            vec_type = None

        b = self._make_global_rhs_vec(rhs_local, vec_type=vec_type)
        x = self._make_initial_guess_vec(init_local, b)

        self._matvec_calls = 0
        ksp.solve(b, x)

        its = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        iflag = reason <= 0

        x_local = torch.from_numpy(np.asarray(x.getArray(readonly=True))).to(
            self.device, dtype=torch.float64
        )

        x.destroy()
        b.destroy()
        ksp.destroy()
        A.destroy()

        return x_local, its, iflag

    def time_step(self, Xstore, sigStore, etaStore, RSstore):
        vesicle = capsules(Xstore, sigStore, None, self.kappa, self.viscCont)
        Xstore_local = Xstore[:, self.start : self.end].to(self.device)
        sigStore_local = sigStore[:, self.start : self.end].to(self.device)

        vesicle_local = capsules(
            Xstore_local, sigStore_local, None, self.kappa, self.viscCont_local
        )

        N = Xstore.shape[0] // 2
        nv = Xstore.shape[1]

        alpha_local = ((1.0 + self.viscCont_local) / 2).double()

        op = self.op
        self.Galpert_local = op.stokesSLmatrix(vesicle_local).contiguous()
        self.NearV2V = naiveNearZoneInfo(Xstore, interpft_vec(Xstore, op.Nup))

        # Build the global block-diagonal self-interaction matrix once per time step.
        # This is needed by the near-singular correction when evaluating vself for
        # all source vesicles, but it should not be rebuilt inside every Krylov matvec.
        if nv > 1:
            galpert_parts = [torch.zeros_like(self.Galpert_local) for _ in range(self.size)]
            dist.all_gather(galpert_parts, self.Galpert_local)
            self.Galpert = torch.cat(galpert_parts, dim=2).contiguous()
        else:
            self.Galpert = self.Galpert_local

        rhs1_local = Xstore_local.clone()
        rhs2_local = torch.zeros((N, self.chunk), dtype=Xstore.dtype, device=self.device)

        if self.repulsion:
            repulsion_local = vesicle_local.dist_repulsionForce(
                Xstore,
                self.repStrength,
                self.minDist,
                self.start,
                self.end,
                Xstore_local,
                self.chunk,
                self.device,
            )

            rep_parts = [torch.zeros_like(repulsion_local) for _ in range(self.size)]
            dist.all_gather(rep_parts, repulsion_local)
            repulsion_global = torch.cat(rep_parts, dim=1)

            Frepulsion_local = op.exactStokesSLdiag(
                vesicle_local, self.Galpert_local, repulsion_local
            )
            SLP = lambda X: op.exactStokesSLdiag(vesicle, self.Galpert, X)
            Frepulsion_local += op.dist_nearSingInt_rbf(
                vesicle,
                repulsion_global,
                SLP,
                self.NearV2V,
                dist_wrapper_allExactStokesSLTarget_compare2,
                vesicle_local,
                True,
                self.start,
                self.end,
                self.rank,
            )

            rhs1_local = rhs1_local + self.dt * Frepulsion_local @ torch.diag(
                1.0 / alpha_local
            )

        vInf_local = self.farField(Xstore_local).double()
        rhs1_local = rhs1_local + self.dt * vInf_local @ torch.diag(1.0 / alpha_local)
        rhs2_local = rhs2_local + vesicle_local.surfaceDiv(Xstore_local)

        rhs_local = torch.cat([rhs1_local, rhs2_local], dim=0).T.reshape(-1).to(torch.float64)

        if self.usePreco:
            self._build_block_diag_preconditioner(vesicle, vesicle_local, N, alpha_local)

        init_global = torch.cat((Xstore, sigStore), dim=0).double().T.reshape(-1)
        init_local = init_global[self.start * 3 * N : self.end * 3 * N].contiguous()

        Xn_local, iter_count, iflag = self._solve_linear_system_petsc(
            rhs_local, init_local, vesicle, vesicle_local
        )

        x_parts = [torch.zeros_like(Xn_local) for _ in range(self.size)]
        dist.all_gather(x_parts, Xn_local)
        Xn = torch.cat(x_parts, dim=0)

        eta = None
        RS = None
        Xn_reshaped = Xn.view(nv, 3, N)
        X_ = Xn_reshaped[:, 0:2, :].reshape(nv, 2 * N).transpose(0, 1).clone()
        sigma_ = Xn_reshaped[:, 2, :].T.clone().to(dtype=torch.float64)

        return X_, sigma_, eta, RS, iter_count, iflag

    def time_matvec_petsc(self, x_petsc: PETSc.Vec, vesicle, vesicle_local):
        self._matvec_calls += 1

        x_local = self._torch_local_from_petsc(x_petsc)

        op = self.op
        N = vesicle.N
        dtype = x_local.dtype

        valPos_local = torch.zeros((2 * N, self.chunk), dtype=dtype, device=self.device)

        # PETSc hands each rank only its local vesicle block. Keep the matvec local
        # as long as possible and communicate only the source density required for
        # off-rank vesicle interactions.
        Xn_reshaped_local = x_local.view(self.chunk, 3, N)
        Xm_local = (
            Xn_reshaped_local[:, 0:2, :].reshape(self.chunk, 2 * N).transpose(0, 1).clone()
        )
        sigmaM_local = Xn_reshaped_local[:, 2, :].T.clone()

        f_local = vesicle_local.tracJump(Xm_local, sigmaM_local)
        check_finite("f_local", f_local, self.rank)
        alpha_local = (1 + vesicle_local.viscCont) / 2

        Gf_local = op.exactStokesSLdiag(vesicle_local, self.Galpert_local, f_local)
        check_finite("Gf_local", Gf_local, self.rank)

        check_finite("Galpert", self.Galpert, self.rank)

        SLP = lambda X: op.exactStokesSLdiag(vesicle, self.Galpert, X)
        f = self._allgather_global_density(f_local)
        check_finite("f", f, self.rank)

        Fslp_local = op.dist_nearSingInt_rbf(
            vesicle,
            f,
            SLP,
            self.NearV2V,
            dist_wrapper_allExactStokesSLTarget_compare2,
            vesicle_local,
            True,
            self.start,
            self.end,
            self.rank,
        )
        check_finite("Fslp_local", Fslp_local, self.rank)

        valPos_local -= self.dt * Gf_local / alpha_local
        valPos_local -= self.dt * Fslp_local / alpha_local

        valTen = vesicle_local.surfaceDiv(Xm_local)
        valPos_local += Xm_local

        val_local = torch.cat(
            [valPos_local.reshape(2, N, self.chunk), valTen.reshape(1, N, self.chunk)],
            dim=0,
        ).reshape(3 * N, self.chunk)
        val_local = val_local.permute(1, 0).reshape(-1).contiguous()

        return val_local.detach().cpu().numpy().astype(np.float64, copy=False)

    def preconditionerBD_petsc(self, x_petsc: PETSc.Vec):
        x_local = self._torch_local_from_petsc(x_petsc)

        N = self.bdiagVes["LU"].shape[1] // 3
        zves_local = x_local.view(self.chunk, 3 * N).unsqueeze(-1)

        val_local = torch.linalg.lu_solve(
            self.bdiagVes["LU"],
            self.bdiagVes["pivots"],
            zves_local,
        ).squeeze(-1)

        return val_local.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)

    def bg_flow(self, X, Xwalls, *args, **kwargs):
        N = X.shape[0] // 2
        nv = X.shape[1]
        x, y = X[:N, :], X[N:, :]
        speed = kwargs.get("Speed", 1.0)

        if "relaxation" in args:
            vInf = torch.zeros((2 * N, nv), dtype=X.dtype, device=X.device)
        elif "extensional" in args:
            vInf = torch.cat((-x, y), dim=0)
        elif "parabolic" in args:
            chanWidth = kwargs.get("chanWidth", 1.0)
            v_x = 1 - (y / chanWidth) ** 2
            v_y = torch.zeros_like(v_x)
            vInf = torch.cat((v_x, v_y), dim=0)
        elif "taylorGreen" in args:
            vortexSize = kwargs.get("vortexSize", 1.0)
            scale = math.pi / vortexSize
            v_x = torch.sin(x * scale) * torch.cos(y * scale)
            v_y = -torch.cos(x * scale) * torch.sin(y * scale)
            vInf = vortexSize * torch.cat((v_x, v_y), dim=0)
        elif "vortex" in args:
            chanWidth = kwargs.get("chanWidth", 2.5)
            vInf = torch.cat(
                [
                    torch.sin(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.cos(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                    -torch.cos(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.sin(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                ],
                dim=0,
            )
        elif "shear" in args:
            v_x = y
            v_y = torch.zeros_like(y)
            vInf = torch.cat((v_x, v_y), dim=0)
        elif any(flow in args for flow in ["choke", "doublechoke", "choke2", "tube"]):
            xwalls = Xwalls[: Xwalls.shape[0] // 2, 0]
            ywalls = Xwalls[Xwalls.shape[0] // 2 :, 0]
            Nbd = xwalls.numel()
            vInf = torch.zeros((2 * Nbd, 1), dtype=X.dtype, device=X.device)
            ind = torch.abs(xwalls) > 0.8 * torch.max(xwalls)
            y_scaled = ywalls[ind] / torch.max(ywalls)
            mollifier = torch.exp(1 / (y_scaled**2 - 1))
            mollifier[torch.isinf(mollifier)] = 0
            vx = mollifier / math.exp(-1)
            vInf[ind, 0] = vx
        elif "couette" in args:
            xwalls = Xwalls[: Xwalls.shape[0] // 2, :]
            ywalls = Xwalls[Xwalls.shape[0] // 2 :, :]
            Nbd = xwalls.shape[0]
            mean_y2 = torch.mean(ywalls[:, 1])
            mean_x2 = torch.mean(xwalls[:, 1])
            rot_x = -ywalls[:, 1] + mean_y2
            rot_y = xwalls[:, 1] - mean_x2
            vInf = torch.cat(
                (
                    torch.zeros((2 * Nbd, 1), dtype=X.dtype, device=X.device),
                    torch.cat((rot_x, rot_y)).unsqueeze(1),
                ),
                dim=1,
            )
        elif "doubleCouette" in args:
            xwalls = Xwalls[: Xwalls.shape[0] // 2, :]
            ywalls = Xwalls[Xwalls.shape[0] // 2 :, :]
            Nbd = xwalls.shape[0]
            mean_y2 = torch.mean(ywalls[:, 1])
            mean_x2 = torch.mean(xwalls[:, 1])
            mean_y3 = torch.mean(ywalls[:, 2])
            mean_x3 = torch.mean(xwalls[:, 2])
            rot_2 = torch.cat((-ywalls[:, 1] + mean_y2, xwalls[:, 1] - mean_x2))
            rot_3 = torch.cat((ywalls[:, 2] - mean_y3, -xwalls[:, 2] + mean_x3))
            vInf = torch.cat(
                (
                    torch.zeros((2 * Nbd, 1), dtype=X.dtype, device=X.device),
                    rot_2.unsqueeze(1),
                    rot_3.unsqueeze(1),
                ),
                dim=1,
            )
        else:
            raise ValueError("Unknown or missing flow type in bg_flow.")

        vInf *= speed
        return vInf

