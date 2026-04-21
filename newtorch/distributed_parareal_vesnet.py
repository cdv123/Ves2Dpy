import torch
import logging
from curve_batch_compile import Curve
from poten import Poten
from distributed_wrapper_MLARM_batch_compile_N128 import MLARM_manyfree_py
from helper_functions import set_bg_flow
from torch import distributed as dist
import numpy as np
from tqdm import tqdm


class VesNetSolver:
    def __init__(
        self, options, params, Xwalls, initPositions, comm_info, nv, new_num_ranks
    ):
        self.rank = comm_info.rank
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(comm_info.device)
        device = comm_info.device
        self.device = device
        cur_dtype = torch.float32

        # Build MLARM class to take time steps using networks
        # Load the normalization (mean, std) values for the networks
        # ADV Net retrained in Oct 2024
        adv_net_input_norm = np.load(
            "../../files2runVes2Dpy/2024Oct_adv_fft_tot_in_para.npy"
        )
        adv_net_output_norm = np.load(
            "../../files2runVes2Dpy/2024Oct_adv_fft_tot_out_para.npy"
        )
        # Relax Net for dt = 1E-5 (DIFF_June8)
        relax_net_input_norm = np.array(
            [
                -8.430413700466488e-09,
                0.06278684735298157,
                6.290720477863943e-08,
                0.13339413702487946,
            ]
        )
        relax_net_output_norm = np.array(
            [
                -2.884585348361668e-10,
                0.00020574081281665713,
                -5.137390512999218e-10,
                0.0001763451291481033,
            ]
        )

        nearNetInputNorm = np.load("../../files2runVes2Dpy/in_param_disth_allmode.npy")
        nearNetOutputNorm = np.load(
            "../../files2runVes2Dpy/out_param_disth_allmode.npy"
        )

        # self ten network updated by using a 156k dataset
        tenSelfNetInputNorm = np.array(
            [
                0.00017108717293012887,
                0.06278623640537262,
                0.002038202714174986,
                0.13337858021259308,
            ]
        )
        tenSelfNetOutputNorm = np.array([337.7627868652344, 466.6429138183594])

        tenAdvNetInputNorm = np.load(
            "../../files2runVes2Dpy/2024Oct_advten_in_para_allmodes.npy"
        )
        tenAdvNetOutputNorm = np.load(
            "../../files2runVes2Dpy/2024Oct_advten_out_para_allmodes.npy"
        )

        vinf = set_bg_flow(options["farField"], params["farFieldSpeed"])
        self.oc = Curve()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        rbf_upsample = -1
        self.new_num_ranks = new_num_ranks

        sub_ranks = list(range(new_num_ranks))
        if comm_info.rank < new_num_ranks:
            self.sub_group = dist.new_group(ranks=sub_ranks)
        else:
            self.sub_group = None
            return

        self.mlarm = MLARM_manyfree_py(
            params["dt"],
            vinf,
            self.oc,
            False,
            1e2,
            rbf_upsample,
            torch.from_numpy(adv_net_input_norm).to(cur_dtype),
            torch.from_numpy(adv_net_output_norm).to(cur_dtype),
            torch.from_numpy(relax_net_input_norm).to(cur_dtype),
            torch.from_numpy(relax_net_output_norm).to(cur_dtype),
            torch.from_numpy(nearNetInputNorm).to(cur_dtype),
            torch.from_numpy(nearNetOutputNorm).to(cur_dtype),
            torch.from_numpy(tenSelfNetInputNorm).to(cur_dtype),
            torch.from_numpy(tenSelfNetOutputNorm).to(cur_dtype),
            torch.from_numpy(tenAdvNetInputNorm).to(cur_dtype),
            torch.from_numpy(tenAdvNetOutputNorm).to(cur_dtype),
            device=device,
            logger=logger,
            rank=comm_info.rank,
            size=new_num_ranks,
            nv=nv,
        )

        self.mlarm.nearNetwork.model.eval()
        self.mlarm.relaxNetwork.model.eval()
        self.mlarm.tenSelfNetwork.model.eval()
        self.mlarm.tenAdvNetwork.model.eval()
        self.mlarm.mergedAdvNetwork.model.eval()
        self.mlarm.nearNetwork.model = torch.compile(
            self.mlarm.nearNetwork.model, mode="reduce-overhead"
        )
        # self.mlarm.advNetwork.model  = torch.compile(self.mlarm.advNetwork.model,  mode="max-autotune")
        self.mlarm.relaxNetwork.model = torch.compile(
            self.mlarm.relaxNetwork.model, mode="reduce-overhead"
        )
        self.mlarm.tenSelfNetwork.model = torch.compile(
            self.mlarm.tenSelfNetwork.model, mode="reduce-overhead"
        )
        self.mlarm.tenAdvNetwork.model = torch.compile(
            self.mlarm.tenAdvNetwork.model, mode="reduce-overhead"
        )

        self.area0, self.len0 = self.oc.geomProp(initPositions)[1:]

        self.mlarm.area0 = self.area0
        # mlarm.area0 = torch.ones((nv), device=X.device, dtype=torch.float32) * 0.0524
        self.mlarm.len0 = self.len0
        self.mlarm.area0_local = self.area0[self.mlarm.start : self.mlarm.end]
        self.mlarm.len0_local = self.len0[self.mlarm.start : self.mlarm.end] 
        # mlarm.len0 = torch.ones((nv), device=X.device, dtype=torch.float32)
        self.mlarm.op = Poten(params["N"])

        self.options = options
        self.params = params

        self.Xwalls = Xwalls
        print("Params", self.params)
        print("options", self.options)

        self.finalTime = self.params["T"]
        self.nlayers = 3

        self.modes = torch.concatenate(
            (torch.arange(0, params["N"] // 2), torch.arange(-params["N"] // 2, 0))
        ).to(initPositions.device)

    def solve(
        self,
        initPositions: torch.Tensor,
        sigmaStore: torch.Tensor,
        out_file_name=None,
        start_time=0,
        rank=None,
    ):
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.float32)
        positions = initPositions.clone()
        dist.broadcast(positions, src=0)
        dist.broadcast(sigmaStore, src=0)
        if self.rank > self.new_num_ranks:
            return positions, sigmaStore

        sigmaStore = sigmaStore.to(dtype=torch.float32)

        for _ in range(10):
            positions, flag = self.oc.redistributeArcLength(positions, self.modes)
            if flag:
                break

        num_steps = int(self.finalTime / self.params["dt"])
        print("Vesnet Solver Sweep")
        print("Number of steps:", num_steps)

        for i in tqdm(range(num_steps)):
            start_time += self.params["dt"]

            with torch.no_grad():
                positions, sigmaStore = self.mlarm.time_step_many_noinfo(
                    positions, sigmaStore, self.nlayers
                )

            if out_file_name is not None:
                output = np.concatenate(
                    ([start_time], positions.cpu().numpy().T.flatten())
                ).astype("float64")

                with open(out_file_name, "ab") as fid:
                    output.tofile(fid)

        return positions, sigmaStore.to(dtype=torch.float64)
