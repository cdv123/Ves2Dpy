import torch

torch.set_default_dtype(torch.float32)
import sys

sys.path.append("..")
from capsules import capsules
from tools.filter import (
    filterShape,
    interpft,
    upsample_fft,
    downsample_fft,
)
from model_zoo.get_network_torch import (
    RelaxNetwork,
    TenSelfNetwork,
    MergedAdvNetwork,
    MergedTenAdvNetwork,
    MergedNearFourierNetwork,
)

from math import ceil, sqrt
from typing import Tuple

from torch import distributed as dist

torch.set_default_dtype(torch.float32)


# CHANGE HERE
# @torch.compile(backend='cudagraphs')
@torch.jit.script
def allExactStokesSLTarget_compare1(
    vesicleX, vesicle_sa, f, tarX, length: float = 1.0, offset: int = 0
):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """

    N, nv = vesicleX.shape[0] // 2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0] // 2, tarX.shape[1]

    if nv <= 1:
        # No other vesicles → no interactions
        stokesSLPtar = torch.zeros(
            (2 * Ntar, ntar),
            dtype=tarX.dtype,
            device=vesicleX.device,
        )

        empty = torch.empty(0, dtype=torch.long, device=vesicleX.device)
        return stokesSLPtar, (empty, empty, empty)

    stokesSLPtar = torch.zeros(
        (2 * Ntar, ntar), dtype=tarX.dtype, device=vesicleX.device
    )

    mask = ~torch.eye(nv, dtype=torch.bool, device=vesicleX.device)

    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    indices = (
        torch.arange(nv, device=vesicleX.device)[None,]
        .expand(nv, -1)[mask]
        .view(nv, nv - 1)
    )
    indices = indices[offset : offset + ntar]

    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1).unsqueeze(2)  # (N, (nv-1), 1, ntar)
    deny = den[N:, indices].permute(0, 2, 1).unsqueeze(2)

    diffx = (
        tarX[None, None, :Ntar, ...]
        - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None]
    )  # broadcasting, (N, (nv-1), Ntar, ntar)
    diffy = (
        tarX[None, None, Ntar:, ...]
        - vesicleX[N:, indices].permute(0, 2, 1)[:, :, None]
    )

    dis2 = diffx**2 + diffy**2

    ids = torch.where(
        torch.max((dis2.reshape(N, nv - 1, -1) < (length / Ntar) ** 2), dim=0)[0]
    )

    ids = (ids[0], ids[1] // ntar, ids[1] % ntar)

    l = len(ids[0])
    ids_ = (
        torch.arange(N, device=f.device)[:, None].expand(-1, l).reshape(-1),
        ids[0][None, :].expand(N, -1).reshape(-1),
        ids[1][None, :].expand(N, -1).reshape(-1),
        ids[2][None, :].expand(N, -1).reshape(-1),
    )

    coeff = (diffx * denx + diffy * deny) / dis2

    stokesSLPtar[:Ntar, :] = torch.sum(
        (coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )
    stokesSLPtar[Ntar:, :] = torch.sum(
        (coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )

    return stokesSLPtar / (4 * torch.pi), (ids[0], ids[1], ids[2] + offset)


# CHANGE HERE
# @torch.compile(backend='cudagraphs')
@torch.jit.script
def allExactStokesSLTarget_compare2(
    vesicleX,
    vesicle_sa,
    f,
    tarX,
    ids0,
    ids1,
    ids2,
    length: float = 1.0,
    offset: int = 0,
):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """

    N, nv = vesicleX.shape[0] // 2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0] // 2, tarX.shape[1]

    if nv <= 1:
        # No other vesicles → no interactions
        stokesSLPtar = torch.zeros(
            (2 * Ntar, ntar),
            dtype=tarX.dtype,
            device=vesicleX.device,
        )

        return stokesSLPtar

    stokesSLPtar = torch.zeros(
        (2 * Ntar, ntar), dtype=vesicleX.dtype, device=vesicleX.device
    )

    mask = ~torch.eye(nv, dtype=torch.bool, device=vesicleX.device)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = (
        torch.arange(nv, device=vesicleX.device)[None,]
        .expand(nv, -1)[mask]
        .view(nv, nv - 1)
    )
    indices = indices[offset : offset + ntar]

    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1).unsqueeze(2)  # (N, (nv-1), nv)
    deny = den[N:, indices].permute(0, 2, 1).unsqueeze(2)

    diffx = (
        tarX[None, None, :Ntar, ...]
        - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None]
    )  # broadcasting, (N, (nv-1), Ntar, nv)
    diffy = (
        tarX[None, None, Ntar:, ...]
        - vesicleX[N:, indices].permute(0, 2, 1)[:, :, None]
    )

    dis2 = diffx**2 + diffy**2

    l = len(ids0)
    ids_ = (
        torch.arange(N, device=f.device)[:, None].expand(-1, l).reshape(-1),
        ids0[None, :].expand(N, -1).reshape(-1),
        ids1[None, :].expand(N, -1).reshape(-1),
        ids2[None, :].expand(N, -1).reshape(-1),
    )

    coeff = (diffx * denx + diffy * deny) / dis2

    stokesSLPtar[:Ntar, :] = torch.sum(
        (coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )
    stokesSLPtar[Ntar:, :] = torch.sum(
        (coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )

    return stokesSLPtar / (4 * torch.pi)


class MLARM_manyfree_py(torch.jit.ScriptModule):
    def __init__(
        self,
        dt,
        vinf,
        oc,
        use_repulsion,
        repStrength,
        rbf_upsample: int,
        advNetInputNorm,
        advNetOutputNorm,
        relaxNetInputNorm,
        relaxNetOutputNorm,
        nearNetInputNorm,
        nearNetOutputNorm,
        tenSelfNetInputNorm,
        tenSelfNetOutputNorm,
        tenAdvNetInputNorm,
        tenAdvNetOutputNorm,
        device,
        logger,
        rank,
        size,
        nv,
    ):
        super().__init__()
        self.rank = rank
        self.num_ranks = size
        self.size = size

        self.chunk = nv // size

        self.start = rank * self.chunk
        self.end = (rank + 1) * self.chunk

        self.dt = dt  # time step size
        self.vinf = (
            vinf  # background flow (analytic -- itorchut as function of vesicle config)
        )
        self.oc = oc  # curve class
        self.kappa = 1  # bending stiffness is 1 for our simulations
        self.device = device
        self.logger = logger
        # Flag for repulsion
        self.use_repulsion = use_repulsion
        self.repStrength = repStrength
        self.rbf_upsample = rbf_upsample

        # Normalization values for advection (translation) networks
        self.advNetInputNorm = advNetInputNorm
        self.advNetOutputNorm = advNetOutputNorm
        self.mergedAdvNetwork = MergedAdvNetwork(
            self.advNetInputNorm.to(device),
            self.advNetOutputNorm.to(device),
            model_path="/cosma/apps/do022/dc-dubo2/mergedNetsN32_March12/2024Oct_ves_merged_adv.pth",
            device=device,
        )

        # Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.relaxNetwork = RelaxNetwork(
            self.dt,
            self.relaxNetInputNorm.to(device),
            self.relaxNetOutputNorm.to(device),
            # model_path="../trained/ves_relax_DIFF_June8_625k_dt1e-5.pth",
            model_path="/cosma/home/do022/dc-dubo2/vesicle-fork/downsample32/Ves_relax_1e-5_downsample_DIFF.pth",
            device=device,
        )

        # Normalization values for near field networks
        self.nearNetInputNorm = nearNetInputNorm
        self.nearNetOutputNorm = nearNetOutputNorm
        self.nearNetwork = MergedNearFourierNetwork(
            self.nearNetInputNorm.to(device),
            self.nearNetOutputNorm.to(device),
            # model_path="../trained/ves_merged_disth_nearFourier.pth",
            model_path="/cosma/apps/do022/dc-dubo2/mergedNetsN32_March12/ves_merged_disth_nearFourier.pth",
            device=device,
        )

        # Normalization values for tension-self network
        self.tenSelfNetInputNorm = tenSelfNetInputNorm
        self.tenSelfNetOutputNorm = tenSelfNetOutputNorm
        self.tenSelfNetwork = TenSelfNetwork(
            self.tenSelfNetInputNorm.to(device),
            self.tenSelfNetOutputNorm.to(device),
            # model_path = "../trained/Ves_2024Oct_selften_12blks_loss_0.00566cuda1.pth",
            model_path="/cosma/home/do022/dc-dubo2/vesicle-fork/downsample32/ves_downsample_selften_zerolevel.pth",
            device=device,
        )

        # Normalization values for tension-advection networks
        self.tenAdvNetInputNorm = tenAdvNetInputNorm
        self.tenAdvNetOutputNorm = tenAdvNetOutputNorm
        self.tenAdvNetwork = MergedTenAdvNetwork(
            self.tenAdvNetInputNorm.to(device),
            self.tenAdvNetOutputNorm.to(device),
            # model_path="../trained/2024Oct_ves_merged_advten.pth",
            # model_path="../../latest_128/adv_ten/2024Oct_ves_merged_advten.pth",
            model_path="/cosma/apps/do022/dc-dubo2/mergedNetsN32_March12/2024Oct_merged_advten.pth",
            device=device,
        )

        self.first_iter = True

    def time_step_many_noinfo(
        self, Xold, tenOld, nlayers=3, local_indicies=None, targets=None
    ):
        torch.cuda.set_device(self.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]
        Nup = ceil(sqrt(N)) * N
        vesicleUp = capsules(upsample_fft(Xold, Nup), [], [], self.kappa, 1)

        # Compute velocity induced by repulsion force
        repForce = torch.zeros_like(Xold)

        # Compute bending forces + old tension forces
        fTen = vesicle.tensionTerm(tenOld)
        fBend = vesicleUp.bendingTerm(vesicleUp.X)  # upsampled bending term
        fBend = downsample_fft(fBend, N)

        tracJump = fBend + fTen  # total elastic force

        Xold_local = Xold[:, self.start : self.end].to(self.device)
        Xstand_local, standardizationValues_local = self.standardizationStep(Xold_local)

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        (
            velx_real_local,
            vely_real_local,
            velx_imag_local,
            vely_imag_local,
            xlayers_local,
            ylayers_local,
        ) = self.predictNearLayers(Xstand_local, standardizationValues_local, nlayers)

        velx_real_local = velx_real_local.contiguous()
        vely_real_local = vely_real_local.contiguous()
        velx_imag_local = velx_imag_local.contiguous()
        vely_imag_local = vely_imag_local.contiguous()
        xlayers_local = xlayers_local.contiguous()
        ylayers_local = ylayers_local.contiguous()
        standardizationValues_local = list(standardizationValues_local)

        for i in range(len(standardizationValues_local)):
            standardizationValues_local[i] = standardizationValues_local[i].contiguous()

        if self.first_iter:
            self.gather_velx_real = [
                torch.zeros_like(velx_real_local, device=self.device)
                for _ in range(self.size)
            ]
            self.gather_vely_real = [
                torch.zeros_like(vely_real_local, device=self.device)
                for _ in range(self.size)
            ]
            self.gather_velx_imag = [
                torch.zeros_like(velx_imag_local, device=self.device)
                for _ in range(self.size)
            ]
            self.gather_vely_imag = [
                torch.zeros_like(vely_imag_local, device=self.device)
                for _ in range(self.size)
            ]
            self.gather_standardizationValues = [
                torch.zeros_like(standardizationValues_local[0], device=self.device)
                for _ in range(self.size)
            ]
            # Last row has int data type, need another buffer
            self.gather_last_standardValues = [
                torch.zeros_like(standardizationValues_local[-1], device=self.device)
                for _ in range(self.size)
            ]

        dist.all_gather(self.gather_velx_real, velx_real_local)
        dist.all_gather(self.gather_vely_real, vely_real_local)
        dist.all_gather(self.gather_velx_imag, velx_imag_local)
        dist.all_gather(self.gather_vely_imag, vely_imag_local)

        velx_real = torch.cat(self.gather_velx_real, dim=0)
        vely_real = torch.cat(self.gather_vely_real, dim=0)
        velx_imag = torch.cat(self.gather_velx_imag, dim=0)
        vely_imag = torch.cat(self.gather_vely_imag, dim=0)

        if self.first_iter:
            self.gather_xlayers = [
                torch.zeros_like(xlayers_local, device=self.device)
                for _ in range(self.size)
            ]
            self.gather_ylayers = [
                torch.zeros_like(ylayers_local, device=self.device)
                for _ in range(self.size)
            ]

        dist.all_gather(self.gather_xlayers, xlayers_local)
        dist.all_gather(self.gather_ylayers, ylayers_local)

        standardizationValues = []
        for i in range(len(standardizationValues_local)):
            if i == len(standardizationValues_local) - 1:
                dist.all_gather(
                    self.gather_last_standardValues,
                    standardizationValues_local[i],
                )

                standardizationValues.append(
                    torch.cat(self.gather_last_standardValues, dim=0)
                )
                continue

            dist.all_gather(
                self.gather_standardizationValues,
                standardizationValues_local[i],
            )

            standardizationValues.append(
                torch.cat(self.gather_standardizationValues, dim=0)
            )

        xlayers = torch.cat(self.gather_xlayers, dim=2)
        ylayers = torch.cat(self.gather_ylayers, dim=2)

        # NEED SYNCHRONIZATION
        info_rbf, info_stokes = None, None

        if self.rbf_upsample <= 2:
            # const = 0.495 * self.len0[0].item() * 4
            # const = 1.7 / 128
            const = 0.0132
        if self.rbf_upsample == 2:
            xlayers = interpft(xlayers.reshape(N, -1), N * 2)
            ylayers = interpft(ylayers.reshape(N, -1), N * 2)
            # build coordinate tensor

        all_X = torch.concat(
            (xlayers.reshape(-1, 1, nv), ylayers.reshape(-1, 1, nv)), dim=1
        )  # (nlayers * N, 2, nv), 2 for x and y
        # all_X = all_X /const * N
        all_X = all_X / const
        matrices = torch.exp(
            -torch.sum((all_X[:, None] - all_X[None, ...]) ** 2, dim=-2)
        )
        matrices += (
            torch.eye(all_X.shape[0], device=self.device).unsqueeze(-1) * 1e-4
        ).expand(-1, -1, nv)  # (nlayers*N, nlayers*N, nv)

        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))

        farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions_noinfo(
            vesicle,
            vesicleUp,
            info_rbf,
            info_stokes,
            L,
            tracJump,
            repForce,
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            xlayers,
            ylayers,
            standardizationValues,
            nlayers,
            first=True,
        )

        farFieldtracJump = filterShape(farFieldtracJump, 16)
        # print(
        #    f"monitoring 1st farfieldtracjump magnitude: {torch.max(torch.abs(farFieldtracJump))}"
        # )

        vinf = vback + farFieldtracJump
        vinf_local = vinf[:, self.start : self.end].to(self.device, non_blocking=True)

        vBack_local = self.invTenMatOnVback(
            Xstand_local, standardizationValues_local, vinf_local
        )

        selfBendSolve_local = self.invTenMatOnSelfBend(
            Xstand_local, standardizationValues_local
        )

        tenNew_local = -(vBack_local + selfBendSolve_local)
        gather_list = [torch.zeros_like(tenNew_local) for _ in range(self.size)]

        dist.all_gather(gather_list, tenNew_local)
        tenNew = torch.cat(gather_list, dim=1)

        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        farFieldtracJump, _, _ = self.computeStokesInteractions_noinfo(
            vesicle,
            vesicleUp,
            info_rbf,
            info_stokes,
            L,
            tracJump,
            repForce,
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            xlayers,
            ylayers,
            standardizationValues,
            nlayers,
            first=False,
        )

        farFieldtracJump = filterShape(farFieldtracJump, 16)

        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(
            torch.isinf(farFieldtracJump)
        ):
            print("before_farFieldtacJump has nan or inf")

        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(
            torch.isinf(farFieldtracJump)
        ):
            print("farFieldtacJump has nan or inf")

        vbackTotal = vback + farFieldtracJump

        Xlocal = Xold[:, self.start : self.end].to(self.device, non_blocking=True)
        vbackTotal_local = vbackTotal[:, self.start : self.end].to(
            self.device, non_blocking=True
        )

        Xadv_local = self.translateVinfwTorch(
            Xlocal, Xstand_local, standardizationValues_local, vbackTotal_local
        )

        if torch.any(torch.isnan(Xadv_local)) or torch.any(torch.isinf(Xadv_local)):
            print("Xadv input has nan or inf")

        Xadv_local = filterShape(Xadv_local, 16)
        Xnew_local = self.relaxWTorchNet(Xadv_local)
        gather_list = [
            torch.zeros_like(Xnew_local, device=self.device) for _ in range(self.size)
        ]

        modes = torch.concatenate(
            (torch.arange(0, N // 2), torch.arange(-N // 2, 0))
        ).to(Xold.device)  # .double()

        XnewC_local = Xnew_local.clone()
        # start.record()
        for _ in range(5):
            Xnew_local, flag = self.oc.redistributeArcLength(Xnew_local, modes)
            if flag:
                break
        Xnew_local = self.oc.alignCenterAngle(XnewC_local, Xnew_local.to(Xold.device))

        # start.record()
        with torch.enable_grad():
            Xnew_local = self.oc.correctAreaAndLengthAugLag(
                Xnew_local, self.area0_local, self.len0_local
            )

        Xnew_local = filterShape(Xnew_local.to(Xold.device), 16)

        dist.all_gather(gather_list, Xnew_local)
        Xnew = torch.cat(gather_list, dim=1)

        # print(f"monitoring tenNew magnitude: {torch.max(torch.abs(tenNew))}")
        # print(
        #    f"monitoring farfieldtracjump magnitude: {torch.max(torch.abs(farFieldtracJump))}"
        # )
        # np.save("debug_last_tenNew.npy", tenNew.cpu().numpy())
        self.first_iter = False
        return Xnew, tenNew

    # @torch.compile(backend='cudagraphs')
    def predictNearLayers(
        self,
        Xstand,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        nlayers: int = 3,
        local_indicies=None,
    ):
        # print('Near network predicting')
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        oc = self.oc

        maxLayerDist = self.len0[0].item() / N  # length = 1, h = 1/N;

        # Create the layers around a vesicle on which velocity calculated
        tracersX_ = torch.zeros(
            (2 * N, nlayers, nv), dtype=Xstand.dtype, device=self.device
        )
        if nlayers == 5:
            dlayer = torch.linspace(
                -maxLayerDist,
                maxLayerDist,
                nlayers,
                dtype=Xstand.dtype,
                device=self.device,
            )
            tracersX_[:, 2] = Xstand
            _, tang = oc.diffProp_jac_tan(Xstand)
            rep_nx = tang[N:, :, None].expand(-1, -1, nlayers - 1)
            rep_ny = -tang[:N, :, None].expand(-1, -1, nlayers - 1)
            dx = rep_nx * dlayer[[0, 1, 3, 4]]  # (N, nv, nlayers-1)
            dy = rep_ny * dlayer[[0, 1, 3, 4]]
            tracersX_[:, [0, 1, 3, 4]] = torch.permute(
                torch.vstack(
                    [
                        torch.repeat_interleave(
                            Xstand[:N, :, None], nlayers - 1, dim=-1
                        )
                        + dx,
                        torch.repeat_interleave(
                            Xstand[N:, :, None], nlayers - 1, dim=-1
                        )
                        + dy,
                    ]
                ),
                (0, 2, 1),
            )
        else:
            dlayer = torch.linspace(
                0, maxLayerDist, nlayers, dtype=Xstand.dtype, device=self.device
            )
            tracersX_[:, 0] = Xstand
            _, tang, _ = oc.diffProp(Xstand)
            rep_nx = torch.repeat_interleave(tang[N:, :, None], nlayers - 1, dim=-1)
            rep_ny = torch.repeat_interleave(-tang[:N, :, None], nlayers - 1, dim=-1)
            dx = rep_nx * dlayer[1:]  # (N, nv, nlayers-1)
            dy = rep_ny * dlayer[1:]
            tracersX_[:, 1:] = torch.permute(
                torch.vstack(
                    [
                        torch.repeat_interleave(
                            Xstand[:N, :, None], nlayers - 1, dim=-1
                        )
                        + dx,
                        torch.repeat_interleave(
                            Xstand[N:, :, None], nlayers - 1, dim=-1
                        )
                        + dy,
                    ]
                ),
                (0, 2, 1),
            )

        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(
            net_pred
        )

        # print(f"------ rel err for half nearNetwork is {torch.norm(net_pred - net_pred_)/torch.norm(net_pred)}")

        if nlayers == 5:
            inner_input_net = self.innerNearNetwork.preProcess(Xstand)
            inner_net_pred = self.innerNearNetwork.forward(inner_input_net)
            inner_velx_real, inner_vely_real, inner_velx_imag, inner_vely_imag = (
                self.innerNearNetwork.postProcess(inner_net_pred)
            )

            velx_real = torch.concat((inner_velx_real, velx_real), dim=-1)
            vely_real = torch.concat((inner_vely_real, vely_real), dim=-1)
            velx_imag = torch.concat((inner_velx_imag, velx_imag), dim=-1)
            vely_imag = torch.concat((inner_vely_imag, vely_imag), dim=-1)

        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues
        Xl_ = self.destandardize(
            tracersX_.reshape(N * 2, -1),
            (
                scaling[None, :].expand(nlayers, -1).reshape(-1),
                rotate[None, :].expand(nlayers, -1).reshape(-1),
                rotCenter.tile((1, nlayers)),
                trans.tile((1, nlayers)),
                sortIdx.tile((nlayers, 1)),
            ),
        )

        xlayers_ = torch.zeros((N, nlayers, nv), dtype=Xstand.dtype)
        ylayers_ = torch.zeros((N, nlayers, nv), dtype=Xstand.dtype)
        xlayers_ = Xl_[
            :N, torch.arange(nlayers * nv, device=self.device).reshape(nlayers, nv)
        ]
        ylayers_ = Xl_[
            N:, torch.arange(nlayers * nv, device=self.device).reshape(nlayers, nv)
        ]

        return velx_real, vely_real, velx_imag, vely_imag, xlayers_, ylayers_

    # @torch.jit.script_method
    def buildVelocityInNear(
        self,
        tracJump,
        velx_real,
        vely_real,
        velx_imag,
        vely_imag,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        nlayers,
    ):
        nv = tracJump.shape[1]
        N = tracJump.shape[0] // 2
        # nlayers = 5
        _, rotate, _, _, sortIdx = standardizationValues

        fstand = self.standardize(
            tracJump,
            torch.zeros((2, nv), dtype=tracJump.dtype, device=self.device),
            rotate,
            torch.zeros((2, nv), dtype=tracJump.dtype, device=self.device),
            torch.tensor([1.0], device=self.device),
            sortIdx,
        )
        z = fstand[:N] + 1.0j * fstand[N:]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)

        velx_stand_ = torch.einsum(
            "vnml, mv -> nvl", velx_real, fstandRe
        ) + torch.einsum("vnml, mv -> nvl", velx_imag, fstandIm)
        vely_stand_ = torch.einsum(
            "vnml, mv -> nvl", vely_real, fstandRe
        ) + torch.einsum("vnml, mv -> nvl", vely_imag, fstandIm)

        vx_ = torch.zeros((nv, nlayers, N), device=self.device, dtype=tracJump.dtype)
        vy_ = torch.zeros((nv, nlayers, N), device=self.device, dtype=tracJump.dtype)

        vx_[torch.arange(nv), :, sortIdx.T] = velx_stand_
        vy_[torch.arange(nv), :, sortIdx.T] = vely_stand_

        VelBefRot_ = torch.concat((vx_, vy_), dim=-1)  # (nv, nlayers, 2N)

        if nv == 1:
            VelRot_ = self.rotationOperator(
                VelBefRot_.reshape(-1, 2 * N).T,
                torch.repeat_interleave(-rotate, nlayers, dim=0),
                torch.zeros(2, nv, device=self.device),
            )
        else:
            VelRot_ = self.rotationOperator(
                VelBefRot_.reshape(-1, 2 * N).T,
                torch.repeat_interleave(-rotate, nlayers, dim=0),
                torch.zeros(nv * nlayers, device=self.device),
            )
            VelRot_ = VelRot_.T.reshape(nv, nlayers, 2 * N).permute(2, 1, 0)
        velx_ = VelRot_[:N]  # (N, nlayers, nv)
        vely_ = VelRot_[N:]

        return velx_, vely_

    def computeStokesInteractions_noinfo(
        self,
        vesicle,
        vesicleUp,
        info_rbf,
        info_stokes,
        L,
        trac_jump,
        repForce,
        velx_real,
        vely_real,
        velx_imag,
        vely_imag,
        xlayers,
        ylayers,
        standardizationValues,
        nlayers,
        first: bool,
        upsample=True,
    ):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(
            trac_jump + repForce,
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            standardizationValues,
            nlayers,
        )
        rep_velx, rep_vely = self.buildVelocityInNear(
            repForce,
            velx_real[..., 2:3],
            vely_real[..., 2:3],
            velx_imag[..., 2:3],
            vely_imag[..., 2:3],
            standardizationValues,
            1,
        )

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        length = 1.0
        # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
        totalForceUp = upsample_fft(totalForce, Nup)

        # start.record()
        if first:
            # NEED SYNCHRONIZATION
            fn = allExactStokesSLTarget_compare1
            if nv > 1048:
                num_parts = 10
                far_fields = []
                info_stokes_parts = [[], [], []]

                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (
                        (i + 1) * nv // num_parts if i < num_parts - 1 else None
                    )  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    far_field, info_stokes = fn(
                        vesicleUp.X,
                        vesicleUp.sa,
                        totalForceUp,
                        vesicle.X[:, start:end],
                        length,
                        offset=offset,
                    )

                    far_fields.append(far_field)
                    for j in range(3):
                        info_stokes_parts[j].append(info_stokes[j])

                far_field_1 = torch.concat(far_fields, dim=-1)
                info_stokes = tuple(
                    torch.cat(parts, dim=0) for parts in info_stokes_parts
                )

            elif nv > 504:
                far_field_1_1, info_stokes_1 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, : nv // 4],
                    length,
                )
                far_field_1_2, info_stokes_2 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, nv // 4 : nv // 2],
                    length,
                    offset=nv // 4,
                )
                far_field_1_3, info_stokes_3 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, nv // 2 : 3 * nv // 4],
                    length,
                    offset=nv // 2,
                )
                far_field_1_4, info_stokes_4 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, 3 * nv // 4 :],
                    length,
                    offset=3 * nv // 4,
                )
                far_field_1 = torch.concat(
                    (far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1
                )
                info_stokes = (
                    torch.cat(
                        (
                            info_stokes_1[0],
                            info_stokes_2[0],
                            info_stokes_3[0],
                            info_stokes_4[0],
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            info_stokes_1[1],
                            info_stokes_2[1],
                            info_stokes_3[1],
                            info_stokes_4[1],
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            info_stokes_1[2],
                            info_stokes_2[2],
                            info_stokes_3[2],
                            info_stokes_4[2],
                        ),
                        dim=0,
                    ),
                )
            elif nv > 100:
                far_field_1_1, info_stokes_1 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, : nv // 2],
                    length,
                )
                far_field_1_2, info_stokes_2 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, nv // 2 :],
                    length,
                    offset=nv // 2,
                )
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2), dim=-1)
                info_stokes = (
                    torch.cat((info_stokes_1[0], info_stokes_2[0]), dim=0),
                    torch.cat((info_stokes_1[1], info_stokes_2[1]), dim=0),
                    torch.cat((info_stokes_1[2], info_stokes_2[2]), dim=0),
                )
            else:
                far_field_1, info_stokes = fn(
                    vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, length
                )
            id1 = info_stokes[2] * N + info_stokes[1]
            id2 = info_stokes[0] + 1 * (info_stokes[0] >= info_stokes[2])
            info_rbf = (id1, id2)

        else:
            # NEED SYNCHRONIZATION
            fn = allExactStokesSLTarget_compare2
            if nv > 1048:
                far_fields = []
                num_parts = 10
                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (
                        (i + 1) * nv // num_parts if i < num_parts - 1 else None
                    )  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    mask = (
                        (start <= info_stokes[2]) & (info_stokes[2] < end)
                        if i < num_parts - 1
                        else (start <= info_stokes[2])
                    )

                    far_field = fn(
                        vesicleUp.X,
                        vesicleUp.sa,
                        totalForceUp,
                        vesicle.X[:, start:end],
                        info_stokes[0][mask],
                        info_stokes[1][mask],
                        info_stokes[2][mask] - start,
                        offset=offset,
                    )

                    far_fields.append(far_field)

                far_field_1 = torch.concat(far_fields, dim=1)

            elif nv > 504:
                far_field_1 = torch.concat(
                    (
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, : nv // 4],
                            info_stokes[0][info_stokes[2] < nv // 4],
                            info_stokes[1][info_stokes[2] < nv // 4],
                            info_stokes[2][info_stokes[2] < nv // 4],
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, nv // 4 : nv // 2],
                            info_stokes[0][
                                (nv // 4 <= info_stokes[2]) & (info_stokes[2] < nv // 2)
                            ],
                            info_stokes[1][
                                (nv // 4 <= info_stokes[2]) & (info_stokes[2] < nv // 2)
                            ],
                            info_stokes[2][
                                (nv // 4 <= info_stokes[2]) & (info_stokes[2] < nv // 2)
                            ]
                            - nv // 4,
                            offset=nv // 4,
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, nv // 2 : 3 * nv // 4],
                            info_stokes[0][
                                (nv // 2 <= info_stokes[2])
                                & (info_stokes[2] < 3 * nv // 4)
                            ],
                            info_stokes[1][
                                (nv // 2 <= info_stokes[2])
                                & (info_stokes[2] < 3 * nv // 4)
                            ],
                            info_stokes[2][
                                (nv // 2 <= info_stokes[2])
                                & (info_stokes[2] < 3 * nv // 4)
                            ]
                            - nv // 2,
                            offset=nv // 2,
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, 3 * nv // 4 :],
                            info_stokes[0][3 * nv // 4 <= info_stokes[2]],
                            info_stokes[1][3 * nv // 4 <= info_stokes[2]],
                            info_stokes[2][3 * nv // 4 <= info_stokes[2]] - 3 * nv // 4,
                            offset=3 * nv // 4,
                        ),
                    ),
                    dim=1,
                )
            elif nv > 100:
                far_field_1 = torch.concat(
                    (
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, : nv // 2],
                            info_stokes[0][info_stokes[2] < nv // 2],
                            info_stokes[1][info_stokes[2] < nv // 2],
                            info_stokes[2][info_stokes[2] < nv // 2],
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, nv // 2 :],
                            info_stokes[0][nv // 2 <= info_stokes[2]],
                            info_stokes[1][nv // 2 <= info_stokes[2]],
                            info_stokes[2][nv // 2 <= info_stokes[2]] - nv // 2,
                            offset=nv // 2,
                        ),
                    ),
                    dim=1,
                )

            else:
                far_field_1 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X,
                    info_stokes[0],
                    info_stokes[1],
                    info_stokes[2],
                )

        selfRepVel = torch.concat((rep_velx.squeeze(1), rep_vely.squeeze(1)), dim=0)
        return far_field_1 + selfRepVel, info_rbf, info_stokes

    # @torch.jit.script_method
    # @torch.compile(backend='cudagraphs')
    def translateVinfwTorch(
        self,
        Xold,
        Xstand,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        vinf,
    ):
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        # Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(Xold)
        _, rotate, _, _, sortIdx = standardizationValues

        # Xinp = self.mergedAdvNetwork.preProcess(Xstand)
        Xpredict = self.mergedAdvNetwork.forward(Xstand)

        Z11r_ = torch.zeros((N, N, nv), dtype=Xstand.dtype, device=self.device)
        Z12r_ = torch.zeros_like(Z11r_, device=self.device)
        Z21r_ = torch.zeros_like(Z11r_, device=self.device)
        Z22r_ = torch.zeros_like(Z11r_, device=self.device)

        Z11r_[:, 1:] = torch.permute(Xpredict[:, :, 0, :N], (2, 0, 1))
        Z21r_[:, 1:] = torch.permute(Xpredict[:, :, 0, N:], (2, 0, 1))
        Z12r_[:, 1:] = torch.permute(Xpredict[:, :, 1, :N], (2, 0, 1))
        Z22r_[:, 1:] = torch.permute(Xpredict[:, :, 1, N:], (2, 0, 1))

        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        vinf_stand = self.standardize(
            vinf,
            torch.zeros((2, nv), dtype=vinf.dtype, device=self.device),
            rotate,
            torch.zeros((2, nv), dtype=vinf.dtype, device=self.device),
            1,
            sortIdx,
        )
        z = vinf_stand[:N] + 1.0j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack(
            (
                torch.einsum("NiB,iB ->NB", Z11r_, V1)
                + torch.einsum("NiB,iB ->NB", Z12r_, V2),
                torch.einsum("NiB,iB ->NB", Z21r_, V1)
                + torch.einsum("NiB,iB ->NB", Z22r_, V2),
            )
        )

        Xnew = torch.zeros_like(Xold)
        MVinf = torch.zeros_like(MVinf_stand)
        idx = torch.vstack([sortIdx.T, sortIdx.T + N])
        MVinf[idx, torch.arange(nv, device=self.device)] = MVinf_stand
        MVinf = self.rotationOperator(
            MVinf, -rotate, torch.zeros((2, nv), dtype=MVinf.dtype)
        )
        Xnew = Xold + self.dt * vinf - self.dt * MVinf

        return Xnew

    def relaxWTorchNet(self, Xmid):
        # RELAXATION w/ NETWORK
        Xin, standardizationValues = self.standardizationStep(Xmid)

        Xpred = self.relaxNetwork.forward(Xin)
        Xnew = self.destandardize(Xpred, standardizationValues)

        return Xnew

    # @torch.compile(backend='cudagraphs')
    def invTenMatOnVback(self, Xstand, standardizationValues, vinf):
        # Approximate inv(Div*G*Ten)*Div*vExt
        # number of vesicles
        nv = Xstand.shape[1]
        # number of points of exact solve
        N = Xstand.shape[0] // 2

        # Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(X)
        _, rotate, _, _, sortIdx = standardizationValues

        input = self.tenAdvNetwork.preProcess(Xstand)
        Xpredict = self.tenAdvNetwork.forward(input)
        out = self.tenAdvNetwork.postProcess(Xpredict)  # shape: (127, nv, 2, 128)

        # Approximate the multiplication Z = inv(DivGT)DivPhi_k
        Z1 = torch.zeros((N, N, nv), dtype=Xstand.dtype, device=self.device)
        Z2 = torch.zeros((N, N, nv), dtype=Xstand.dtype, device=self.device)

        Z1[:, 1:] = torch.permute(out[:, :, 0], (2, 0, 1))
        Z2[:, 1:] = torch.permute(out[:, :, 1], (2, 0, 1))

        vBackSolve = torch.zeros((N, nv), dtype=Xstand.dtype, device=self.device)
        vinfStand = self.standardize(
            vinf,
            torch.zeros((2, nv), dtype=Xstand.dtype, device=self.device),
            rotate,
            torch.zeros((2, nv), dtype=Xstand.dtype, device=self.device),
            1,
            sortIdx,
        )
        z = vinfStand[:N] + 1.0j * vinfStand[N:]
        zh = torch.fft.fft(z, dim=0)

        V1_ = torch.real(zh)
        V2_ = torch.imag(zh)

        # Compute the approximation to inv(Div*G*Ten)*Div*vExt
        MVinfStand = torch.einsum("NiB,iB ->NB", Z1, V1_) + torch.einsum(
            "NiB,iB ->NB", Z2, V2_
        )

        # Destandardize the multiplication
        vBackSolve[sortIdx.T, torch.arange(nv, device=self.device)] = MVinfStand

        return vBackSolve

    # @torch.compile(backend='cudagraphs')
    def invTenMatOnSelfBend(self, Xstand, standardizationValues):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = Xstand.shape[1]  # number of vesicles
        N = Xstand.shape[0] // 2

        # Xstand, scaling, _, _, _, sortIdx = self.standardizationStep(X)
        scaling, _, _, _, sortIdx = standardizationValues

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)

        tenPred = torch.zeros((N, nv), dtype=Xstand.dtype, device=self.device)
        if nv == 1:
            col = (
                torch.arange(nv, device=self.device)
                .unsqueeze(0)
                .expand(sortIdx.shape[0], -1)
            )
            tenPred[sortIdx, col] = tenPredictStand / scaling**2
        else:
            tenPred[sortIdx.T, torch.arange(nv, device=self.device)] = (
                tenPredictStand / scaling**2
            )

        return tenPred

    # @torch.compile(backend='cudagraphs')
    def standardizationStep(self, Xin):
        # compatible with multi ves
        X = Xin.clone()
        N = X.shape[0] // 2
        # % Equally distribute points in arc-length
        modes = torch.concatenate(
            (torch.arange(0, N // 2), torch.arange(-N // 2, 0))
        ).to(X.device)  # .double()
        for _ in range(5):
            X, flag = self.oc.redistributeArcLength(X, modes)
            # if flag:
            #     break

        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)

        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, (scaling, rotate, rotCenter, trans, multi_sortIdx)

    # @torch.compile(backend='cudagraphs')
    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        nv = X.shape[1]

        Xrotated = self.rotation_trans_Operator(X, rotation, rotCenter, translation)

        XrotSort = torch.vstack(
            (
                Xrotated[multi_sortIdx.T, torch.arange(nv, device=self.device)],
                Xrotated[multi_sortIdx.T + N, torch.arange(nv, device=self.device)],
            )
        )

        XrotSort = scaling * XrotSort
        return XrotSort

    # @torch.compile(backend='cudagraphs')
    def destandardize(
        self,
        XrotSort,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """compatible with multiple ves"""
        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues

        N = len(sortIdx[0])
        nv = XrotSort.shape[1]

        # Scale back
        XrotSort = XrotSort / scaling

        # Change ordering back
        X = torch.zeros_like(XrotSort)
        X[sortIdx.T, torch.arange(nv, device=self.device)] = XrotSort[:N]
        X[sortIdx.T + N, torch.arange(nv, device=self.device)] = XrotSort[N:]

        # Take translation back
        X = self.translateOp(X, -trans)
        # Take rotation back
        X = self.rotationOperator(X, -rotate, rotCenter)

        # X = self.trans_rotation_Operator(X, -rotate, rotCenter, -trans)

        return X

    def referenceValues(self, Xref):
        """Shan: compatible with multi ves"""

        oc = self.oc
        N = len(Xref) // 2
        # nv = Xref.shape[1]
        tempX = Xref.clone()

        # Find the physical center
        rotCenter = oc.getPhysicalCenter(tempX)
        multi_V = oc.getPrincAxesGivenCentroid(tempX, rotCenter)
        rotation = torch.arctan2(multi_V[0], multi_V[1])

        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center_ = oc.getPhysicalCenter(Xref)  # redundant?
        translation = -center_

        Xref = self.translateOp(Xref, translation)

        theta = torch.arctan2(Xref[N:], Xref[:N])
        start_id = torch.argmin(torch.where(theta < 0, 100, theta), dim=0)
        multi_sortIdx = (
            start_id + torch.arange(N, device=self.device).unsqueeze(-1)
        ) % N
        multi_sortIdx = multi_sortIdx.int().T

        length = oc.geomProp_length(Xref)
        scaling = 1.0 / length

        return translation, rotation, rotCenter, scaling, multi_sortIdx

    def rotationOperator(self, X, theta, rotCent):
        """Shan: compatible with multi ves
        theta of shape (1,nv), rotCent of shape (2,nv)"""
        Xrot = torch.zeros_like(X, device=self.device)
        x = X[: len(X) // 2] - rotCent[0]
        y = X[len(X) // 2 :] - rotCent[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[: len(X) // 2] = xrot + rotCent[0]
        Xrot[len(X) // 2 :] = yrot + rotCent[1]
        return Xrot

    def translateOp(self, X, transXY):
        """Shan: compatible with multi ves
        transXY of shape (2,nv)"""
        Xnew = torch.zeros_like(X)
        Xnew[: len(X) // 2] = X[: len(X) // 2] + transXY[0]
        Xnew[len(X) // 2 :] = X[len(X) // 2 :] + transXY[1]
        return Xnew

    def rotation_trans_Operator(self, X, theta, rotCent, transXY):
        """
        combining rotate and trans
        """

        Xrot = torch.zeros_like(X, device=self.device)
        x = X[: len(X) // 2] - rotCent[0]
        y = X[len(X) // 2 :] - rotCent[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[: len(X) // 2] = xrot + rotCent[0] + transXY[0]
        Xrot[len(X) // 2 :] = yrot + rotCent[1] + transXY[1]
        return Xrot
