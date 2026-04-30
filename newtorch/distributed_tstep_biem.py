import math
import time
import numpy as np
import torch
from torch import distributed as dist

from curve_batch_compile import Curve
from capsules import capsules
from poten import Poten
from tools.filter import interpft_vec
from biem_support_scaled import dist_wrapper_allExactStokesSLTarget_compare2, scalableNearZoneInfo
import cupy as cp

if torch.cuda.is_available():
    from cupyx.scipy.sparse.linalg import gmres, LinearOperator
else:
    from scipy.sparse.linalg import gmres, LinearOperator


torch.set_default_dtype(torch.float32)


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
            f"[rank {rank}] {name} has non-finite values; bad={bad}, shape={tuple(x.shape)}, min={xmin}, max={xmax}"
        )


class gmres_counter:
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1


class TStepBiem:
    """
    Matrix-free distributed BIEM time-stepper.

    Important limitation:
    cupyx.scipy.sparse.linalg.gmres is not MPI-distributed, so every rank still
    owns the full Krylov vector. The expensive operator application and block-
    diagonal preconditioner are distributed across ranks, and only the minimum
    global gathers required by the replicated GMRES are performed.
    """

    def __init__(self, X, Xwalls, options, prams, rank, size, device, group=None):
        oc = Curve()

        self.Xwalls = Xwalls
        _, self.area, self.length = oc.geomProp(X)

        self.dt = prams["dt"]
        self.rank = rank
        self.size = size
        self.group = group
        self.nv = X.shape[1]
        if self.nv % size != 0:
            raise ValueError(
                f"nv={self.nv} must be divisible by world_size={size} for this implementation."
            )
        self.chunk = self.nv // size
        self.start = rank * self.chunk
        self.end = (rank + 1) * self.chunk
        self.device = device
        self.dtype = X.dtype

        self.currentTime = 0.0
        self.finalTime = prams["T"]
        self.kappa = prams["kappa"]
        self.viscCont = prams["viscCont"]
        self.viscCont_local = self.viscCont[self.start : self.end].to(self.device)
        self.gmresTol = prams["gmresTol"]
        self.gmresMaxIter = prams["gmresMaxIter"]
        self.areaLenTol = prams["areaLenTol"]

        self.farField = lambda X_: self.bg_flow(
            X_,
            self.Xwalls,
            options["farField"],
            Speed=prams["farFieldSpeed"],
            chanWidth=prams["chanWidth"],
            vortexSize=prams["vortexSize"],
        )

        self.confined = self.Xwalls is not None
        self.repulsion = options["repulsion"]
        if prams["nv"] == 1 and not self.confined:
            self.repulsion = False
        self.repStrength = prams["repStrength"]
        self.minDist = prams["minDist"]

        self.op = Poten(prams["N"], group=self.group)
        self.usePreco = options["usePreco"]
        self.matFreeWalls = options.get("matFreeWalls", False)

        self.bdiagVes = None
        self.bdiagTen = None
        self.bdiagWall = None
        self.wallDLP = None
        self.wallN0 = None
        self.wallDLPandRSmat = None
        self.haveWallMats = False
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

        # Timestep cache: rebuilt once per time-step, reused by every GMRES matvec.
        self._vesicle = None
        self._vesicle_local = None
        self._N = None
        self._alpha = None
        self._alpha_local = None
        self._global_vec_len = None
        self._rhs = None
        self._rhs_local = None
        self._gather_rhs_buf = None
        self._gather_f_buf = None
        self._gather_val_buf = None
        self._local_val_shape = None
        self._local_rhs_shape = None
        self._local_force_shape = None

        if self.confined:
            self.initial_confined()
            self.eta = None
            self.RS = None
        else:
            self.opWall = None

            N = prams["N"]

        self._local_force_shape = (2 * N, self.chunk)
        self._local_rhs_shape = (3 * N * self.chunk,)
        self._local_val_shape = (3 * N * self.chunk,)

        self._gather_rhs_buf = [
            torch.empty(self._local_rhs_shape, device=self.device)
            for _ in range(self.size)
        ]
        self._gather_f_buf = [
            torch.empty(self._local_force_shape, device=self.device)
            for _ in range(self.size)
        ]
        self._gather_val_buf = [
            torch.empty(self._local_val_shape, device=self.device)
            for _ in range(self.size)
        ]

    def initial_confined(self):
        Nbd = self.Xwalls.shape[0] // 2
        nvbd = self.Xwalls.shape[1]
        self.opWall = Poten(Nbd, group=self.group)
        uwalls = self.farField([])
        self.walls = capsules(
            self.Xwalls, None, uwalls, torch.zeros(nvbd, 1), torch.zeros(nvbd, 1)
        )
        self.wallDLP = self.opWall.stokesDLmatrix(self.walls)
        self.wallN0 = self.opWall.stokesN0matrix(self.walls)
        self.bdiagWall = self.walls.wallsPrecond()

    def _cupy_to_torch(self, x, dtype=torch.float64):
        if isinstance(x, cp.ndarray):
            return torch.utils.dlpack.from_dlpack(x.toDlpack()).to(
                self.device, dtype=dtype
            )
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device, dtype=dtype)
        return x.to(self.device, dtype=dtype)

    def _torch_to_cupy(self, x):
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))

    def _build_timestep_cache(self, Xstore, sigStore):
        self._vesicle = capsules(Xstore, sigStore, None, self.kappa, self.viscCont)
        self.op._dist_rbf_cache = {}

        Xstore_local = Xstore[:, self.start : self.end].to(self.device)
        sigStore_local = sigStore[:, self.start : self.end].to(self.device)
        self._vesicle_local = capsules(
            Xstore_local,
            sigStore_local,
            None,
            self.kappa,
            self.viscCont_local,
        )

        N = Xstore.shape[0] // 2
        self._N = N
        self._alpha = ((1.0 + self.viscCont) / 2).to(self.device, dtype=Xstore.dtype)
        self._alpha_local = ((1.0 + self.viscCont_local) / 2).to(
            self.device, dtype=Xstore.dtype
        )
        self._global_vec_len = self.nv * 3 * N

        # Build these once per step, not once per matvec.
        self.Galpert_local = self.op.stokesSLmatrix(self._vesicle_local).contiguous()

        self.NearV2V = scalableNearZoneInfo(
            Xstore,
            self.op.Nup,
            local_start=self.start,
            local_end=self.end,
        )

        self.SLP = self._make_global_diag_slp()

    def _build_block_diagonal_preconditioner(self):
        if not self.usePreco:
            self.bdiagVes = None
            return

        N = self._N
        vesicle_local = self._vesicle_local
        alpha_local = self._alpha_local

        Ben_local, Ten_local, Div_local = vesicle_local.computeDerivs()
        I = (
            torch.eye(2 * N, device=self.device, dtype=self.dtype)
            .unsqueeze(-1)
            .repeat(1, 1, self.chunk)
        )
        Z = torch.zeros((N, N, self.chunk), device=self.device, dtype=self.dtype)

        G_Ben_local = torch.matmul(
            self.Galpert_local.permute(2, 0, 1),
            Ben_local.permute(2, 0, 1),
        ).permute(1, 2, 0)
        G_Ten_local = torch.matmul(
            self.Galpert_local.permute(2, 0, 1),
            Ten_local.permute(2, 0, 1),
        ).permute(1, 2, 0)
        DivZ = torch.cat([Div_local, Z], dim=1)

        alpha_inv = (1.0 / alpha_local).view(1, 1, self.chunk)
        top_left = I + self.dt * self.kappa * G_Ben_local
        top_right = -self.dt * G_Ten_local * alpha_inv
        top = torch.cat([top_left, top_right], dim=1)
        mat_all = torch.cat([top, DivZ], dim=0)

        LU, pivots = torch.linalg.lu_factor(mat_all.permute(2, 0, 1).contiguous())
        self.bdiagVes = {"LU": LU, "pivots": pivots}


    def _make_global_diag_slp(self):
        def _slp(f_global):
            N = self._N
            f_local = f_global[:, self.start : self.end].contiguous()
            vself_local = self.op.exactStokesSLdiag(
                self._vesicle_local, self.Galpert_local, f_local
            ).contiguous()
            #vself_local = self.op.exactStokesSLdiag_matrix_free(
            #    self._vesicle_local, self.Galpert_local, f_local
            #).contiguous()
            gathered = [torch.empty_like(vself_local) for _ in range(self.size)]
            dist.all_gather(gathered, vself_local, group=self.group)
            return torch.cat(gathered, dim=1).contiguous()

        return _slp

    def reset_profile(self):
        self.prof = {
            "cache_total": 0.0,
            "galpert": 0.0,
            "nearzone": 0.0,
            "preconditioner_setup": 0.0,
            "rhs_total": 0.0,
            "rhs_gather": 0.0,
            "gmres_total": 0.0,
            "matvec_total": 0.0,
            "tracjump": 0.0,
            "self_slp": 0.0,
            "gather_f": 0.0,
            "near_sing": 0.0,
            "gather_val": 0.0,
            "preconditioner_apply": 0.0,
        }


    def _assemble_rhs(self, Xstore):
        N = self._N
        Xstore_local = Xstore[:, self.start : self.end].to(self.device)
        vesicle = self._vesicle
        vesicle_local = self._vesicle_local
        alpha_local = self._alpha_local

        rhs1_local = Xstore_local.clone()
        rhs2_local = torch.zeros(
            (N, self.chunk), dtype=Xstore.dtype, device=self.device
        )

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
            repulsion_gather = [
                torch.empty_like(repulsion_local) for _ in range(self.size)
            ]
            dist.all_gather(repulsion_gather, repulsion_local, group=self.group)
            repulsion_global = torch.cat(repulsion_gather, dim=1).contiguous()

            #Frepulsion_local = self.op.exactStokesSLdiag(
            #    vesicle_local, self.Galpert_local, repulsion_local
            #)
            Frepulsion_local = self.op.exactStokesSLdiag(
                vesicle_local, self.Galpert_local, repulsion_local
            )
            SLP = self.SLP 
            Frepulsion_local += self.op.dist_nearSingInt_rbf(
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
                group=self.group,
            )
            rhs1_local = rhs1_local + self.dt * Frepulsion_local @ torch.diag(
                1.0 / alpha_local
            )

        vInf_local = self.farField(Xstore_local)
        rhs1_local = rhs1_local + self.dt * vInf_local @ torch.diag(1.0 / alpha_local)
        rhs2_local = rhs2_local + vesicle_local.surfaceDiv(Xstore_local)

        rhs_local = (
            torch.cat([rhs1_local, rhs2_local], dim=0).T.reshape(-1).contiguous()
        )
        dist.all_gather(self._gather_rhs_buf, rhs_local, group=self.group)
        self._rhs_local = rhs_local
        self._rhs = torch.cat(self._gather_rhs_buf, dim=0).contiguous()

    def time_step(self, Xstore, sigStore, etaStore, RSstore):
        # self.start_time = time.perf_counter()
        self.reset_profile()
        self._build_timestep_cache(Xstore, sigStore)
        # self.prof["cache_total"] = time.perf_counter() - self.start_time
        self._build_block_diagonal_preconditioner()
        # self.prof["preconditioner_setup"] = time.perf_counter() - self.start_time
        self._assemble_rhs(Xstore)

        initGMRES = (
            torch.cat((Xstore, sigStore), dim=0)
            .T.reshape(-1)
            .to(self.device, dtype=torch.float64)
        )
        counter = gmres_counter(disp=True)

        gmres_func = lambda X: self.time_matvec(X)
        cupy_lin_op = LinearOperator(
            (self._global_vec_len, self._global_vec_len), gmres_func
        )

        if self.usePreco:
            precond_lin_op = LinearOperator(
                (self._global_vec_len, self._global_vec_len), self.preconditionerBD
            )
        else:
            precond_lin_op = None

        rhs = self._rhs
        if torch.cuda.is_available():
            Xn, info = gmres(
                cupy_lin_op,
                self._torch_to_cupy(rhs),
                rtol=self.gmresTol,
                maxiter=self.gmresMaxIter,
                M=precond_lin_op,
                x0=self._torch_to_cupy(initGMRES),
                callback=counter,
            )
        else:
            Xn, info = gmres(
                cupy_lin_op,
                rhs.cpu().numpy(),
                rtol=self.gmresTol,
                maxiter=self.gmresMaxIter,
                M=precond_lin_op,
                x0=initGMRES.cpu().numpy(),
                callback=counter,
            )
        # self.prof["gmres_total"] = time.perf_counter() - self.start_time

        iflag = info != 0
        Xn = self._cupy_to_torch(Xn, dtype=torch.float32)

        N = self._N
        eta = None
        RS = None
        Xn_reshaped = Xn.view(self.nv, 3, N)
        X_ = Xn_reshaped[:, 0:2, :].reshape(self.nv, 2 * N).transpose(0, 1).clone()
        sigma_ = Xn_reshaped[:, 2, :].T.clone().to(dtype=torch.float64)
        # end_time = time.perf_counter()

        # if self.rank == 0:
            # print("Total time:", end_time - self.start_time)
            # print(self.prof)

        return X_, sigma_, eta, RS, counter.niter, iflag

    def time_matvec(self, Xn):
        # start_matvec = time.perf_counter()
        Xn = self._cupy_to_torch(Xn, dtype=torch.float64)

        op = self.op
        vesicle = self._vesicle
        vesicle_local = self._vesicle_local
        N = self._N

        Xn_reshaped = Xn.view(self.nv, 3, N)
        Xn_reshaped_local = Xn_reshaped[self.start : self.end]
        Xm_local = (
            Xn_reshaped_local[:, 0:2, :]
            .reshape(self.chunk, 2 * N)
            .transpose(0, 1)
            .contiguous()
        )
        sigmaM_local = Xn_reshaped_local[:, 2, :].T.contiguous()

        f_local = vesicle_local.tracJump(Xm_local, sigmaM_local)
        # f_local_time = time.perf_counter() - start_matvec

        Gf_local = op.exactStokesSLdiag(vesicle_local, self.Galpert_local, f_local)
        # Gf_local_time = time.perf_counter()- start_matvec

        dist.all_gather(self._gather_f_buf, f_local, group=self.group)
        # gf_dist_time = time.perf_counter()- start_matvec
        f = torch.cat(self._gather_f_buf, dim=1).contiguous()

        SLP = self.SLP
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
            group=self.group,
        )
        # fslp_local_time = time.perf_counter()- start_matvec

        alpha_local = ((1.0 + vesicle_local.viscCont) / 2).to(
            self.device, dtype=Gf_local.dtype
        )
        valPos_local = (
            Xm_local
            - self.dt * Gf_local / alpha_local
            - self.dt * Fslp_local / alpha_local
        )
        # valPos_local_time = time.perf_counter()- start_matvec
        valTen = vesicle_local.surfaceDiv(Xm_local)

        val_local = (
            torch.cat(
                [
                    valPos_local.reshape(2, N, self.chunk),
                    valTen.reshape(1, N, self.chunk),
                ],
                dim=0,
            )
            .reshape(3 * N, self.chunk)
            .permute(1, 0)
            .reshape(-1)
            .contiguous()
        )

        dist.all_gather(self._gather_val_buf, val_local, group=self.group)
        # gather_val_buf_local_time = time.perf_counter() - start_matvec 
        val_global = torch.cat(self._gather_val_buf, dim=0).contiguous()
        # matvec_time = time.perf_counter() - start_matvec
        #if self.rank == 0:
        #    print("f_local_time", f_local_time)
        #    print("Gf_local_time", Gf_local_time)
        #    print("gf_dist_time", gf_dist_time)
        #    print("fslp_local_time", fslp_local_time)
        #    print("valPos_local_time", valPos_local_time)
        #    print("gather_val_buf_local_time", gather_val_buf_local_time)
        #    print("matvec_time", matvec_time)
        #    self.prof["matvec_total"] += matvec_time


        if torch.cuda.is_available():
            return self._torch_to_cupy(val_global)
        return val_global.cpu().numpy()

    def preconditionerBD(self, z):
        z = self._cupy_to_torch(z, dtype=torch.float64)

        N = self._N
        nv = self.nv
        zves = z[: 3 * N * nv].view(nv, 3 * N)
        zves_local = zves[self.start : self.end].unsqueeze(-1)

        val_local = torch.linalg.lu_solve(
            self.bdiagVes["LU"],
            self.bdiagVes["pivots"],
            zves_local,
        ).squeeze(-1)
        val_local_flat = val_local.reshape(-1).contiguous()

        gathered = [torch.empty_like(val_local_flat) for _ in range(self.size)]
        dist.all_gather(gathered, val_local_flat, group=self.group)
        val_global = torch.cat(gathered, dim=0).contiguous()

        if torch.cuda.is_available():
            return self._torch_to_cupy(val_global)
        return val_global.cpu().numpy()

    def RSlets(X, center, stokeslet, rotlet):
        x, y = X[0], X[1]
        cx, cy = center[0], center[1]
        dx = x - cx
        dy = y - cy
        rho2 = dx**2 + dy**2 + 1e-14
        LogTerm_x = -0.5 * torch.log(rho2) * stokeslet[0]
        rorTerm_x = (dx * dx * stokeslet[0] + dx * dy * stokeslet[1]) / rho2
        RotTerm_x = (dy / rho2) * rotlet
        velx = (1 / (4 * math.pi)) * (LogTerm_x + rorTerm_x) + RotTerm_x
        LogTerm_y = -0.5 * torch.log(rho2) * stokeslet[1]
        rorTerm_y = (dy * dx * stokeslet[0] + dy * dy * stokeslet[1]) / rho2
        RotTerm_y = -(dx / rho2) * rotlet
        vely = (1 / (4 * math.pi)) * (LogTerm_y + rorTerm_y) + RotTerm_y
        return torch.stack([velx, vely], dim=0)

    def wallsPrecond(o):
        walls = o.walls
        Nbd = walls.N
        nvbd = walls.nv
        oc = Curve()
        x, y = oc.getXY(walls.X)
        nory, norx = oc.getXY(walls.xt)
        nory = -nory
        sa = walls.sa
        cx, cy = oc.getXY(walls.center)
        Ntot = 2 * Nbd * nvbd
        Nstokes = 3 * (nvbd - 1)
        M11 = torch.zeros((Ntot, Ntot), dtype=torch.float64)
        M12 = torch.zeros((Ntot, Nstokes), dtype=torch.float64)
        M21 = torch.zeros((Nstokes, Ntot), dtype=torch.float64)
        jump = -0.5
        M11[: 2 * Nbd, : 2 * Nbd] += o.wallN0[:, :, 0]
        for k in range(nvbd):
            istart = 2 * Nbd * k
            iend = istart + 2 * Nbd
            M11[istart:iend, istart:iend] += (
                jump * torch.eye(2 * Nbd, dtype=torch.float64) + o.wallDLP[:, :, k]
            )
        for ktar in range(nvbd):
            itar = 2 * Nbd * ktar
            jtar = itar + 2 * Nbd
            K = list(range(ktar)) + list(range(ktar + 1, nvbd))
            for ksou in K:
                isou = 2 * Nbd * ksou
                jsou = isou + 2 * Nbd
                xtar = x[:, ktar].unsqueeze(1).repeat(1, Nbd)
                ytar = y[:, ktar].unsqueeze(1).repeat(1, Nbd)
                xsou = x[:, ksou].unsqueeze(0).repeat(Nbd, 1)
                ysou = y[:, ksou].unsqueeze(0).repeat(Nbd, 1)
                norxtmp = norx[:, ksou].unsqueeze(0).repeat(Nbd, 1)
                norytmp = nory[:, ksou].unsqueeze(0).repeat(Nbd, 1)
                satmp = sa[:, ksou].unsqueeze(0).repeat(Nbd, 1)
                rho2 = (xtar - xsou) ** 2 + (ytar - ysou) ** 2
                coeff = (
                    (1 / math.pi)
                    * ((xtar - xsou) * norxtmp + (ytar - ysou) * norytmp)
                    * satmp
                    / rho2**2
                )
                D = torch.cat(
                    [
                        coeff * (xtar - xsou) ** 2,
                        coeff * (xtar - xsou) * (ytar - ysou),
                        coeff * (ytar - ysou) * (xtar - xsou),
                        coeff * (ytar - ysou) ** 2,
                    ],
                    dim=1,
                ).reshape(2 * Nbd, 2 * Nbd) * (2 * math.pi / Nbd)
                M11[itar:jtar, isou:jsou] = D
        for k in range(nvbd - 1):
            icol = 3 * k
            istart = 2 * Nbd * (k + 1)
            iend = istart + Nbd
            M21[icol, istart:iend] = (2 * math.pi / Nbd) * sa[:, k + 1]
            M21[icol + 2, istart:iend] = (
                (2 * math.pi / Nbd) * sa[:, k + 1] * y[:, k + 1]
            )
            istart += Nbd
            iend += Nbd
            M21[icol + 1, istart:iend] = (2 * math.pi / Nbd) * sa[:, k + 1]
            M21[icol + 2, istart:iend] -= (
                (2 * math.pi / Nbd) * sa[:, k + 1] * x[:, k + 1]
            )
        for k in range(nvbd - 1):
            for ktar in range(nvbd):
                dx = x[:, ktar] - cx[k + 1]
                dy = y[:, ktar] - cy[k + 1]
                rho2 = dx**2 + dy**2
                istart = 2 * Nbd * ktar
                iend = istart + Nbd
                base = 3 * k
                M12[istart:iend, base] += (
                    1 / (4 * math.pi) * (-0.5 * torch.log(rho2) + dx * dx / rho2)
                )
                M12[istart + Nbd : iend + Nbd, base] += (
                    1 / (4 * math.pi) * (dx * dy / rho2)
                )
                M12[istart:iend, base + 1] += 1 / (4 * math.pi) * (dy * dx / rho2)
                M12[istart + Nbd : iend + Nbd, base + 1] += (
                    1 / (4 * math.pi) * (-0.5 * torch.log(rho2) + dy * dy / rho2)
                )
                M12[istart:iend, base + 2] += dy / rho2
                M12[istart + Nbd : iend + Nbd, base + 2] -= dx / rho2
        M22 = -2 * math.pi * torch.eye(3 * (nvbd - 1), dtype=torch.float64)
        top = torch.cat([M11, M12], dim=1)
        bottom = torch.cat([M21, M22], dim=1)
        M = torch.cat([top, bottom], dim=0)
        if not o.matFreeWalls:
            o.wallDLPandRSmat = M
        return torch.linalg.inv(M)

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
            vInf[ind, 0] = mollifier / math.exp(-1)
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
        return vInf * speed


