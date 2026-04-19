# %%
import torch
from helper_functions import init_distributed

comm_info = init_distributed()
torch.cuda.set_device(comm_info.device)
import numpy as np

torch.set_default_dtype(torch.float32)
import torch.backends.cudnn as cudnn

cudnn.benchmark = False
cudnn.deterministic = True
import torch._dynamo
from torch.profiler import profile, ProfilerActivity

torch._dynamo.reset()
# from curve_batch import Curve
from curve_batch_compile import Curve

from distributed_wrapper_MLARM_batch_compile_N128 import MLARM_manyfree_py
from math import sqrt
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.filter import interpft, interpft_vec
import logging
from torch import distributed as dist
from poten import Poten
from parse_args import parse_cli, modify_options_params

torch.cuda.set_device(comm_info.device)
torch.set_default_device(comm_info.device)

device = comm_info.device
cur_dtype = torch.float32

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# handler = logging.FileHandler('try_N128_VF25_TG_nv128.log')
handler = logging.FileHandler("try_N128_shear.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load curve_py
oc = Curve(logger)

# File name
# fileName = './output_N128/linshi.bin'  # To save simulation data
# fileName = './output_N128/ls.bin'  # To save simulation data
# fileName = './output_N128/lsls.bin'  # To save simulation data
# fileName = './output_N128/TG48.bin'  # To save simulation data


def set_bg_flow(bgFlow, speed):
    def get_flow(X):
        N = X.shape[0] // 2  # Assuming the input X is split into two halves
        if bgFlow == "relax":
            return torch.zeros_like(X)  # Relaxation
        elif bgFlow == "shear":
            return speed * torch.vstack((X[N:], torch.zeros_like(X[:N])))  # Shear
        elif bgFlow == "taylorGreen":
            return speed * torch.vstack(
                (
                    torch.sin(X[:N]) * torch.cos(X[N:]),
                    -torch.cos(X[:N]) * torch.sin(X[N:]),
                )
            )  # Taylor-Green
        elif bgFlow == "parabolic":
            return torch.vstack(
                (speed * (1 - (X[N:] / 0.375) ** 2), torch.zeros_like(X[:N]))
            )  # Parabolic
        elif bgFlow == "rotation":
            r = torch.sqrt(X[:N] ** 2 + X[N:] ** 2)
            theta = torch.atan2(X[N:], X[:N])
            return speed * torch.vstack(
                (-torch.sin(theta) / r, torch.cos(theta) / r)
            )  # Rotation
        elif bgFlow == "vortex":
            chanWidth = 2.5 * 2
            return speed * torch.cat(
                [
                    torch.sin(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.cos(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                    -torch.cos(X[: X.shape[0] // 2] / chanWidth * torch.pi)
                    * torch.sin(X[X.shape[0] // 2 :] / chanWidth * torch.pi),
                ],
                dim=0,
            )

        else:
            return torch.zeros_like(X)

    return get_flow


# Flow specification
# bgFlow = 'shear'
# speed = 2000
# vinf = set_bg_flow(bgFlow, speed)
args = parse_cli
options = {}
params = {}
fileName, Xics = modify_options_params(args, options, params)

bgFlow = options["farField"]
speed = params["farFieldSpeed"]
vinf = set_bg_flow(bgFlow, speed)


# Time stepping
dt = params["dt"]  # Time step size
Th = params["T"] * dt  # Time horizon

# Vesicle discretization
N = 128  # Number of points to discretize vesicle
nlayers = 3
rbf_upsample = -1


if params["nv"] == 1:
    Xics = Xics - Xics.mean()

X0 = torch.from_numpy(Xics).to(device).float()

if X0.shape[0] != 2 * N:
    X0 = interpft_vec(X0, N)

nv = X0.shape[1]
print("Number of vesicles", nv)
area0, len0 = oc.geomProp(X0)[1:]
print(f"area0 is {area0}")
print(f"len0 is {len0}")
X = X0.clone().to(device)

# %matplotlib inline
# plt.figure()
# plt.plot(X0[:128], X0[128:])
# plt.axis('scaled')
# plt.show()

# %%
print(f"We have {nv} vesicles")
Ten = torch.from_numpy(np.zeros((128, nv))).to(device).float()

# Build MLARM class to take time steps using networks
# Load the normalization (mean, std) values for the networks
# ADV Net retrained in Oct 2024
adv_net_input_norm = np.load("../../files2runVes2Dpy/2024Oct_adv_fft_tot_in_para.npy")
adv_net_output_norm = np.load("../../files2runVes2Dpy/2024Oct_adv_fft_tot_out_para.npy")
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
# nearNetInputNorm = np.load("../trained/in_param_allmode.npy")
# nearNetOutputNorm = np.load("../trained/out_param_allmode.npy")
nearNetInputNorm = np.load("../../files2runVes2Dpy/in_param_disth_allmode.npy")
nearNetOutputNorm = np.load("../../files2runVes2Dpy/out_param_disth_allmode.npy")
# tenSelfNetInputNorm = np.array([2.980232033378272e-11, 0.06010082736611366,
#                         -1.0086939616904544e-10, 0.13698545098304749])
# tenSelfNetOutputNorm = np.array([327.26141357421875, 375.0673828125 ])

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

torch.set_default_dtype(torch.float32)

mlarm = MLARM_manyfree_py(
    dt,
    vinf,
    oc,
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
    size=comm_info.numProcs,
    nv=nv,
)

mlarm.nearNetwork.model.eval()
mlarm.relaxNetwork.model.eval()
mlarm.tenSelfNetwork.model.eval()
mlarm.tenAdvNetwork.model.eval()
mlarm.mergedAdvNetwork.model.eval()
# mlarm.nearNetwork.model = torch.compile(mlarm.nearNetwork.model, mode="reduce-overhead")
# mlarm.advNetwork.model  = torch.compile(mlarm.advNetwork.model,  mode="max-autotune")

print("default dtype:", torch.get_default_dtype())
print("near param dtype:", next(mlarm.nearNetwork.model.parameters()).dtype)
print("relax param dtype:", next(mlarm.relaxNetwork.model.parameters()).dtype)
print("tenSelf param dtype:", next(mlarm.tenSelfNetwork.model.parameters()).dtype)
print("tenAdv param dtype:", next(mlarm.tenAdvNetwork.model.parameters()).dtype)

# mlarm.relaxNetwork.model = torch.compile(
#    mlarm.relaxNetwork.model, mode="reduce-overhead"
# )
# mlarm.tenSelfNetwork.model = torch.compile(
#    mlarm.tenSelfNetwork.model, mode="reduce-overhead"
# )
# mlarm.tenAdvNetwork.model = torch.compile(
#    mlarm.tenAdvNetwork.model, mode="reduce-overhead"
# )
# print(type(mlarm.nearNetwork))

area0, len0 = oc.geomProp(X)[1:]
mlarm.area0 = area0
# mlarm.area0 = torch.ones((nv), device=X.device, dtype=torch.float32) * 0.0524
mlarm.len0 = len0

mlarm.area0_local = area0[mlarm.start : mlarm.end]
mlarm.len0_local = len0[mlarm.start : mlarm.end]
# mlarm.len0 = torch.ones((nv), device=X.device, dtype=torch.float32)
mlarm.op = Poten(N)

modes = (
    torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0)))
    .to(X.device)
    .float()
)
for _ in range(10):
    X, flag = oc.redistributeArcLength(X, modes)
    if flag:
        break

# ellipse = np.load("relaxed_shape.npy")
# ellipse = torch.from_numpy(ellipse).float().to(device)
# center_ = oc.getPhysicalCenter(ellipse)
# ellipse[:32, :] -= center_[0]
# ellipse[32:, :] -= center_[1]
# mlarm.ellipse = interpft_vec(ellipse, N)
# logger.info(f"center is {oc.getPhysicalCenter(mlarm.ellipse)}")


# Save the initial data
with open(fileName, "wb") as fid:
    np.array([N, nv]).flatten().astype("float64").tofile(fid)
    X.cpu().numpy().T.flatten().astype("float64").tofile(fid)

# Evolve in time
currtime = 0
# it = 0
print("Tension dtype", Ten.dtype)

print(f"using 3 layers, {mlarm.rbf_upsample} upsampling, saved as {fileName}")
# while currtime < Th:

# mlarm.time_step_many_noinfo = torch.compile(
#    mlarm.time_step_many_noinfo,
#    fullgraph=True,
#    mode="reduce-overhead"
# )

# with torch.inference_mode():
for it in tqdm(range(int(Th // dt))):
    # for it in range(20):
    # Take a time step
    # tStart = time.time()

    # X, Ten = mlarm.time_step_many(X, Ten)
    with torch.no_grad():
        X, Ten = mlarm.time_step_many_noinfo(X, Ten, nlayers)
        # X, Ten = mlarm.time_step_many_noinfo_exactVelLayer(X, Ten, nlayers)
    ## np.save(f"shape_t{currtime}.npy", X)
    # tEnd = time.time()

    ## Find error in area and length
    # area, length = oc.geomProp(X)[1:]
    # errArea = torch.max(torch.abs(area - mlarm.area0) / mlarm.area0)
    # errLen = torch.max(torch.abs(length - mlarm.len0) / mlarm.len0)
    # if options["reparameterization"]:
    # Redistribute arc-length
    ## Update counter and time
    ## it += 1
    currtime += dt

    ## Print time step info
    # print('********************************************')
    # print(f'{it+1}th time step for N=128, time: {currtime}')
    # print(f'Solving with networks takes {tEnd - tStart} sec.')
    # print(f'Error in area and length: {max(errArea, errLen)}')
    # print('********************************************\n')

    ## Save data
    if comm_info.rank == 0:
        output = np.concatenate(([currtime], X.cpu().numpy().T.flatten())).astype(
            "float64"
        )
        with open(fileName, "ab") as fid:
            output.tofile(fid)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
