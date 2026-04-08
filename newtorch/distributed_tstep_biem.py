import torch
from torch import distributed as dist
import numpy as np
import math

torch.set_default_dtype(torch.float32)
from curve_batch_compile import Curve
from capsules import capsules
from poten import Poten
from tools.filter import interpft_vec, interpft
from biem_support import wrapper_allExactStokesSLTarget_compare2, naiveNearZoneInfo
import cupy as cp

if torch.cuda.is_available():
    from cupyx.scipy.sparse.linalg import gmres, LinearOperator
else:
    from scipy.sparse.linalg import gmres, LinearOperator


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1


class TStepBiem:
    """
    This class defines the functions required to advance the geometry
    forward in time. Handles both implicit and explicit vesicle-vesicle
    interactions, different inextensibility conditions, viscosity
    contrast, solid walls vs. unbounded flows. This class also
    implements the adaptive time stepping strategy where the errors in
    length, area, and the residual are monitored to adaptively choose a
    time step size.
    """

    def __init__(self, X, Xwalls, options, prams, rank, size, device):
        oc = Curve()

        self.Xwalls = Xwalls  # Points coordinates on walls
        _, self.area, self.length = oc.geomProp(X)  # vesicles' initial area and length

        self.dt = prams["dt"]  # Time step size
        self.rank = rank
        self.size = size
        self.nv = X.shape[1]
        self.chunk = self.nv // size

        self.start = rank * self.chunk
        self.end = (rank + 1) * self.chunk
        self.device = device

        # Method always starts at time 0
        self.currentTime = 0.0

        # Need the time horizon for adaptive time stepping
        self.finalTime = prams["T"]

        # vesicles' bending stiffness
        self.kappa = prams["kappa"]

        # vesicles' viscosity contrast
        self.viscCont = prams["viscCont"]

        # GMRES tolerance
        self.gmresTol = prams["gmresTol"]
        # maximum number of gmres iterations
        self.gmresMaxIter = prams["gmresMaxIter"]

        # Far field boundary condition
        self.farField = lambda X_: self.bg_flow(
            X_,
            self.Xwalls,
            options["farField"],
            Speed=prams["farFieldSpeed"],
            chanWidth=prams["chanWidth"],
            vortexSize=prams["vortexSize"],
        )

        # whether geometry is bounded or not
        self.confined = self.Xwalls is not None

        # allowable error in area and length
        self.areaLenTol = prams["areaLenTol"]

        # use repulsion in the model
        self.repulsion = options["repulsion"]
        # if there is only one vesicle, turn off repulsion
        if prams["nv"] == 1 and not self.confined:
            self.repulsion = False

        # repulsion strength
        self.repStrength = prams["repStrength"]
        # minimum distance to activate repulsion
        self.minDist = prams["minDist"]

        # class for evaluating potential so that we don't have
        # to keep building quadrature matrices
        self.op = Poten(prams["N"])

        # use block-diagonal preconditioner?
        self.usePreco = options["usePreco"]
        self.bdiagVes = None  # precomputed inverse of block-diagonal preconditioner

        # build poten classes for walls
        if self.confined:
            self.initialConfined()  # create wall related matrices (preconditioner, DL potential, etc)

            self.eta = None  # density on wall
            self.RS = None  # rotlets and stokeslets defined at the center

            # Compute wall-wall interactions matrix free
            self.matFreeWalls = options["matFreeWalls"]
        else:
            self.opWall = None

        # precomputed inverse of block-diagonal preconditioner for tension solve
        self.bdiagTen = None

        # precomputed inverse of block-diagonal preconditioner only for wall-wall interactions
        self.bdiagWall = None

        # Double-layer potential due to solid walls
        self.wallDLP = None

        # Modification of double-layer potential to remove the rank deficiency on outer boundary
        self.wallN0 = None

        # wall2wall interaction matrix computed in initialConfined
        self.wallDLPandRSmat = None

        # do we have wall matrices computed before?
        self.haveWallMats = False

        # Single-layer stokes potential matrix using Alpert quadrature
        self.Galpert = None

        # Double-layer stokes potential matrix
        self.D = None

        # Double-layer Laplace potential matrix
        self.lapDLP = None

        # Double-layer stokes potential matrix without correction
        self.DLPnoCorr = None

        # Single-layer stokes potential matrix without correction
        self.SLPnoCorr = None

        # near-singular integration for vesicle to vesicle
        self.NearV2V = None

        # near-singular integration for wall to vesicle
        self.NearW2V = None

        # near-singular integration for vesicle to wall
        self.NearV2W = None

        # near-singular integration for wall to wall
        self.NearW2W = None

        # Inverse of the blocks of wall-2-wall interaction matrix
        self.invM11 = None
        self.invM22 = None

    def initial_confined(self):
        """
        Builds wall-related matrices for the confined domain simulation.
        """
        Nbd = self.Xwalls.shape[0] // 2  # number of points per wall
        nvbd = self.Xwalls.shape[1]  # number of walls

        # If the walls are discretized with the same Nbd
        self.opWall = Poten(Nbd)

        # Velocity on solid walls from the no-slip boundary condition
        uwalls = self.farField([])

        # Build the wall vesicle-like structure
        self.walls = capsules(
            self.Xwalls, None, uwalls, torch.zeros(nvbd, 1), torch.zeros(nvbd, 1)
        )

        # Build and store the double-layer potential matrix for walls
        self.wallDLP = self.opWall.stokesDLmatrix(self.walls)

        # Matrix N0 to remove rank-1 deficiency
        self.wallN0 = self.opWall.stokesN0matrix(self.walls)

        # Block diagonal preconditioner for the solid walls
        self.bdiagWall = self.walls.wallsPrecond()

    def time_step(self, Xstore, sigStore, etaStore, RSstore):
        """
        Advances the solution one timestep using implicit vesicle-vesicle interactions.
        """
        vesicle = capsules(Xstore, sigStore, None, self.kappa, self.viscCont)
        # print("After vesicle", torch.cuda.memory_allocated())
        Xstore_local = Xstore[:, self.start : self.end].to(self.device)
        sigStore_local = sigStore[:, self.start : self.end].to(self.device)

        vesicle_local = capsules(
            Xstore_local, sigStore_local, None, self.kappa, self.viscCont
        )

        N = Xstore.shape[0] // 2  # Number of points per vesicle
        nv = Xstore.shape[1]  # Number of vesicles

        # Constant in front of time derivative
        alpha = (1.0 + self.viscCont) / 2
        alpha = alpha.float()

        # Build single-layer potential matrix
        op = self.op
        # self.Galpert = op.stokesSLmatrix(vesicle)
        self.Galpert_local = op.stokesSLmatrix(vesicle_local)
        # print("After galpert", torch.cuda.memory_allocated())

        # print("nearZone finished")
        # Right-hand side components
        rhs1_local = Xstore_local
        rhs2_local = torch.zeros(
            (N, self.chunk), dtype=Xstore.dtype, device=Xstore.device
        )

        # --- Repulsion ---
        if self.repulsion:
            repulsion_local = vesicle.repulsionForce(
                Xstore_local, self.repStrength, self.minDist
            )
            Galpert_gather = [
                torch.zeros_like(self.Galpert_local) for _ in range(self.size)
            ]
            dist.all_gather(Galpert_gather, self.Galpert_local)
            self.Galpert = torch.cat(Galpert_gather, dim=2)

            repulsion_gather = [
                torch.zeros_like(repulsion_local) for _ in range(self.size)
            ]
            dist.all_gather(repulsion_gather, repulsion_local)
            repulsion_global = torch.cat(repulsion_gather, dim=1)

            Frepulsion_local = op.exactStokesSLdiag(
                vesicle_local, self.Galpert_local, repulsion_local
            )
            SLP = lambda X: op.exactStokesSLdiag(vesicle, self.Galpert, X)

            # Frepulsion += op.nearSingInt(vesicle, repulsion, SLP, self.NearV2V, op.exactStokesSL, vesicle, True)
            Frepulsion_local += op.dist_nearSingInt_rbf(
                vesicle,
                repulsion_global,
                SLP,
                self.NearV2V,
                wrapper_allExactStokesSLTarget_compare2,  # change to make it target vesicles
                vesicle_local,
                True,
                self.start,
                self.end,
            )

            rhs1_local = rhs1_local + self.dt * Frepulsion_local @ torch.diag(
                1.0 / alpha
            )

        # --- Far-field background flow ---
        vInf_local = self.farField(Xstore_local)
        rhs1_local = rhs1_local + self.dt * vInf_local @ torch.diag(1.0 / alpha)

        # --- Inextensibility condition ---
        rhs2_local = rhs2_local + vesicle.surfaceDiv(Xstore_local)

        # --- Stack RHS ---
        rhs_local = torch.cat([rhs1_local, rhs2_local], dim=0)
        rhs_local = rhs_local.T.reshape(-1)

        # --- Preconditioner ---
        if self.usePreco:
            Ben_local, Ten_local, Div_local = vesicle_local.computeDerivs()
            bdiagVes_LU = torch.zeros((3 * N, 3 * N, self.chunk))
            bdiagVes_P = torch.zeros((3 * N, nv), dtype=torch.int32)

            I = torch.eye(2 * N).unsqueeze(-1).repeat(1, 1, self.chunk)  # (2N, 2N, nv)
            Z = torch.zeros((N, N, self.chunk))  # (N, N, nv)

            # Shared blocks
            G_Ben_local = torch.matmul(
                self.Galpert_local.permute(2, 0, 1), Ben_local.permute(2, 0, 1)
            ).permute(1, 2, 0)  # (2N, N, nv)
            G_Ten_local = torch.matmul(
                self.Galpert_local.permute(2, 0, 1), Ten_local.permute(2, 0, 1)
            ).permute(1, 2, 0)  # (2N, N, nv)
            DivZ = torch.cat([Div_local, Z], dim=1)  # (N, 3N, nv)

            # Compute alpha-inv terms
            alpha_inv = 1.0 / alpha.view(1, 1, self.chunk)  # shape (1, 1, nv)

            # Build top-left block
            top_left = I + self.dt * vesicle.kappa * G_Ben_local

            # Top-right block (same in both cases)
            top_right = -self.dt * G_Ten_local * alpha_inv

            # Combine top blocks
            top = torch.cat([top_left, top_right], dim=1)  # (2N, 3N, nv)

            # Combine with bottom block
            mat_all = torch.cat([top, DivZ], dim=0)  # (3N, 3N, nv)

            LU, P = torch.linalg.lu_factor(mat_all.permute(2, 0, 1))
            bdiagVes_LU = LU
            bdiagVes_P = P

            self.bdiagVes = {"LU": bdiagVes_LU, "pivots": bdiagVes_P}

        # --- GMRES solve ---
        initGMRES = torch.cat((Xstore, sigStore), dim=0).double().T.reshape(-1)

        global matvecs
        matvecs = 0

        gmres_func = lambda X: self.time_matvec(X, Xstore_local, vesicle, vesicle_local)
        cupy_lin_op = LinearOperator(
            (initGMRES.shape[0], initGMRES.shape[0]), gmres_func
        )
        torch.cuda.empty_cache()

        counter = gmres_counter(disp=True)

        if self.usePreco:
            # print(f"gmres tol {self.gmresTol}")
            precond_lin_op = LinearOperator(
                (initGMRES.shape[0], initGMRES.shape[0]), self.preconditionerBD
            )
            if torch.cuda.is_available():
                Xn, info = gmres(
                    cupy_lin_op,
                    cp.asarray(rhs),
                    tol=self.gmresTol,
                    maxiter=self.gmresMaxIter,
                    M=precond_lin_op,
                    x0=cp.asarray(initGMRES),
                    callback=counter,
                )
            else:
                Xn, info = gmres(
                    cupy_lin_op,
                    rhs,
                    rtol=self.gmresTol,
                    maxiter=self.gmresMaxIter,
                    M=precond_lin_op,
                    x0=initGMRES,
                )
        else:
            Xn, info = gmres(
                cupy_lin_op, rhs_local, tol=self.gmresTol, maxiter=self.gmresMaxIter
            )

        # print(f"gmres takes {counter.niter} iterations")
        # print("After gmres", torch.cuda.memory_allocated())

        iflag = info != 0
        iter = counter.niter
        Xn = torch.as_tensor(Xn, dtype=torch.float32)

        # --- Unstack results ---
        eta = None
        RS = None

        Xn_reshaped = Xn.view(nv, 3, N)
        X_ = Xn_reshaped[:, 0:2, :].reshape(nv, 2 * N).transpose(0, 1).clone()
        sigma_ = Xn_reshaped[:, 2, :].T.clone().to(dtype=torch.float64)

        return X_, sigma_, eta, RS, iter, iflag

    def time_matvec(self, Xn, Xn_local, vesicle, vesicle_local):
        """
        Matrix-vector product for GMRES

        Assumes no walls and no viscosity difference between vesicle and surrounding fluid
        """
        Xn_local = Xn[:, self.start : self.end].to(self.device)
        global matvecs
        matvecs += 1
        if isinstance(Xn_local, np.ndarray):
            Xn_local = (
                torch.from_numpy(Xn_local).to(self.device).double()
            )  # Convert to tensor
        elif isinstance(Xn_local, cp.ndarray):
            Xn_local = torch.as_tensor(Xn_local).to(self.device).double()
        else:
            Xn_local = Xn_local.to(self.device).double()

        op = self.op
        N = vesicle.N

        device = Xn_local.device  # Use same device as input
        dtype = Xn_local.dtype

        valPos_local = torch.zeros((2 * N, self.chunk), dtype=dtype, device=device)

        Xn_reshaped_local = Xn_local.view(self.chunk, 3, N)  # [nv, 3, N]
        Xm_local = (
            Xn_reshaped_local[:, 0:2, :]
            .reshape(self.chunk, 2 * N)
            .transpose(0, 1)
            .clone()
        )
        sigmaM_local = Xn_reshaped_local[:, 2, :].T.clone()

        f_local = vesicle_local.tracJump(Xm_local, sigmaM_local)
        alpha = (1 + vesicle_local.viscCont) / 2

        Gf_local = op.exactStokesSLdiag(vesicle_local, self.Galpert_local, f_local)
        Galpert_gather = [
            torch.zeros_like(self.Galpert_local) for _ in range(self.size)
        ]
        dist.all_gather(Galpert_gather, self.Galpert_local)
        self.Galpert = torch.cat(Galpert_gather, dim=2)

        SLP = lambda X: op.exactStokesSLdiag(vesicle, self.Galpert, X)
        f_gather = [torch.zeros_like(f_local) for _ in range(self.size)]
        dist.all_gather(f_gather, f_local)
        f = torch.cat(f_gather, dim=1)

        Fslp_local = op.dist_nearSingInt_rbf(
            vesicle,
            f,
            SLP,
            self.NearV2V,
            wrapper_allExactStokesSLTarget_compare2,
            vesicle_local,
            True,
            self.start,
            self.end,
        )

        valPos_local -= self.dt * Gf_local / alpha
        valPos_local -= self.dt * Fslp_local / alpha

        valTen = vesicle_local.surfaceDiv(Xm_local)
        valPos_local += Xm_local

        val_reshaped = torch.cat(
            [valPos_local.reshape(2, N, self.chunk), valTen.reshape(1, N, self.chunk)],
            dim=0,
        ).reshape(3 * N, self.chunk)
        val = val_reshaped.permute(1, 0).reshape(-1)

        val_gather = [torch.zeros_like(val) for _ in range(self.size)]
        dist.all_gather(val_gather, val)
        val_global = torch.cat(val_gather, dim=0)

        return (
            cp.asarray(val_global)
            if torch.cuda.is_available()
            else np.asarray(val_global)
        )

    def RSlets(X, center, stokeslet, rotlet):
        """
        Evaluate the velocity due to stokeslet and rotlet terms in 2D.

        Args:
            X         : Tensor of shape (2, N), evaluation points.
            center    : Tensor of shape (2,), the center of the stokeslet/rotlet.
            stokeslet: Tensor of shape (2,), the stokeslet vector.
            rotlet   : Scalar (float or tensor), the rotlet strength.

        Returns:
            vel: Tensor of shape (2, N), velocity at each evaluation point.
        """
        # Separate x and y components
        x, y = X[0], X[1]
        cx, cy = center[0], center[1]

        dx = x - cx
        dy = y - cy
        rho2 = dx**2 + dy**2

        # Avoid division by zero
        eps = 1e-14
        rho2 = rho2 + eps

        # Compute x-component of velocity
        LogTerm_x = -0.5 * torch.log(rho2) * stokeslet[0]
        rorTerm_x = (1.0 / rho2) * (dx * dx * stokeslet[0] + dx * dy * stokeslet[1])
        RotTerm_x = (dy / rho2) * rotlet
        velx = (1 / (4 * math.pi)) * (LogTerm_x + rorTerm_x) + RotTerm_x

        # Compute y-component of velocity
        LogTerm_y = -0.5 * torch.log(rho2) * stokeslet[1]
        rorTerm_y = (1.0 / rho2) * (dy * dx * stokeslet[0] + dy * dy * stokeslet[1])
        RotTerm_y = -(dx / rho2) * rotlet
        vely = (1 / (4 * math.pi)) * (LogTerm_y + rorTerm_y) + RotTerm_y

        # Combine components
        vel = torch.stack([velx, vely], dim=0)
        return vel

    def preconditionerBD(o, z):
        """
        Applies block-diagonal preconditioner to the GMRES input vector `z`.

        Args:
            o: Object with attributes:
                - bdiagVes: Contains LU decompositions (L, U) for each vesicle.
                - bdiagWall: Preconditioner matrix (or operator) for the wall.
            z: Flattened torch tensor of shape (3*N*nv + 2*Nbd*nvbd + ...)

        Returns:
            val: Preconditioned vector as torch tensor.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).double()  # Convert to tensor
        elif isinstance(z, cp.ndarray):
            z = torch.as_tensor(z).double()

        nv = o.bdiagVes["LU"].shape[0]  # number of vesicles
        N = o.bdiagVes["LU"].shape[1] // 3  # points per vesicle

        # Extract vesicle part
        zves = z[: 3 * N * nv]
        zves_batched = zves.view(nv, 3 * N).unsqueeze(-1)
        # Use batched LU solve: LU is (3N, 3N, nv), pivots is (3N, nv)
        valVes_batched = torch.linalg.lu_solve(
            o.bdiagVes["LU"], o.bdiagVes["pivots"], zves_batched
        )
        valVes = valVes_batched.squeeze(-1).reshape(-1)

        # Combine and return
        val = torch.cat([valVes, valWalls], dim=0) if o.confined else valVes
        return cp.asarray(val) if torch.cuda.is_available() else np.asarray(val)

    def wallsPrecond(o):
        """
        Computes the inverse of the double-layer potential matrix for Stokes flow
        in a bounded domain. This is used as a wall preconditioner in vesicle simulations.

        Args:
            o: Object with attributes:
                - walls: Contains geometry and panel data
                - wallN0: Initial matrix block (torch.Tensor)
                - wallDLP: Self and interaction terms (torch.Tensor)
                - matFreeWalls: If False, store full inverse in `o.wallDLPandRSmat`

        Returns:
            Mat: Inverse of full wall interaction matrix
        """

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

        # Diagonal jump terms
        jump = -0.5
        M11[: 2 * Nbd, : 2 * Nbd] += o.wallN0[:, :, 0]

        for k in range(nvbd):
            istart = 2 * Nbd * k
            iend = istart + 2 * Nbd
            M11[istart:iend, istart:iend] += (
                jump * torch.eye(2 * Nbd, dtype=torch.float64) + o.wallDLP[:, :, k]
            )

        # Off-diagonal interactions
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

        # M21 — integrals of densities
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

        # M12 — velocity from rotlets/stokeslets
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

        # M22 — identity block for constraints
        M22 = -2 * math.pi * torch.eye(3 * (nvbd - 1), dtype=torch.float64)

        # Stack full block matrix
        top = torch.cat([M11, M12], dim=1)
        bottom = torch.cat([M21, M22], dim=1)
        M = torch.cat([top, bottom], dim=0)

        if not o.matFreeWalls:
            o.wallDLPandRSmat = M

        # Invert the matrix
        Mat = torch.linalg.inv(M)

        return Mat

    def bg_flow(self, X, Xwalls, *args, **kwargs):
        """
        Computes the velocity field at the points X. vInf is either background or wall velocity.
        Flows are given by:
            relaxation:     (0,0)
            extensional:    (-x,y)
            parabolic:      (k(1-y/r)^2,0)
            taylorGreen:    (sin(x)cos(y),-cos(x)sin(y))
            shear:          (ky,0)
            choke:          poiseuille-like flow at intake and outtake
            doublechoke:    same as choke
            couette:        rotating boundary
            doubleCouette   two rotating boundaries
            tube            poiseuille flow in a tube
        """

        N = X.shape[0] // 2  # number of points per vesicle
        nv = X.shape[1]  # number of vesicles

        # Separate out x and y coordinates of vesicles
        x, y = X[:N, :], X[N:, :]

        # Get speed
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

        # Scale the velocity
        vInf *= speed
        return vInf
