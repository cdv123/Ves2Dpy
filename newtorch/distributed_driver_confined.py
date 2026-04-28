import torch
import time

from helper_functions import CommInfo, init_distributed

comm_info = init_distributed()
torch.cuda.set_device(comm_info.device)

torch.set_default_dtype(torch.float32)
import numpy as np
from curve_batch_compile import Curve
from capsules import capsules
import time
from distributed_tstep_biem_scaled import TStepBiem
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from tools.filter import filterShape, interpft_vec
from torch.profiler import profile, ProfilerActivity
from parse_args import parse_cli, modify_options_params
torch.set_default_dtype(torch.float64)


rank = comm_info.rank
size = comm_info.numProcs


def initVes2D(options=None, prams=None):
    """
    Initialize vesicle simulation parameters and options.

    Args:
        options (dict): Simulation options.
        prams (dict): Physical and numerical parameters.

    Returns:
        options (dict), prams (dict): Fully populated with defaults.
    """

    if options is None:
        options = {}
    if prams is None:
        prams = {}

    # --- Default parameters ---
    defaultPram = {
        "N": 32,
        "nv": 1,
        "Nbd": 0,
        "nvbd": 0,
        "T": 1,
        "dt": 1e-5,
        "kappa": 1e-1,
        "viscCont": 1,
        "gmresTol": 1e-5,
        "gmresMaxIter": 1000,
        "areaLenTol": 1e-2,
        "repStrength": 900,
        "minDist": 0.4,
        "farFieldSpeed": 1000,
        "chanWidth": 2.5,
        "vortexSize": 2.5,
    }

    defaultOpt = {
        "farField": "shear",
        "repulsion": False,
        "correctShape": False,
        "reparameterization": True,
        "usePreco": True,
        "matFreeWalls": False,
        "confined": False,
    }

    # --- Fill in missing prams ---
    for key, val in defaultPram.items():
        prams.setdefault(key, val)

    # --- Fill in missing options ---
    for key, val in defaultOpt.items():
        options.setdefault(key, val)

    # --- Geometry-dependent fix ---
    if not options["confined"]:
        prams["Nbd"] = 0
        prams["nvbd"] = 0

    # --- Ensure viscCont is array of correct length ---
    if isinstance(prams["viscCont"], (int, float)):
        prams["viscCont"] = [prams["viscCont"]] * prams["nv"]

    return options, prams


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_device(comm_info.device)

prams = {}
options = {
    "farField": "taylorGreen",
    "repulsion": False,
    "correctShape": True,
    "reparameterization": True,
    "usePreco": True,
    "matFreeWalls": False,
    "confined": False,
}

args = parse_cli()
fileName, Xics = modify_options_params(args, options, prams)
if prams["nv"] == 1:
    Xics = Xics - Xics.mean()

# Assume oc is your geometry utility class (like curve_py in MATLAB)
oc = Curve()  # You need to define this with required methods

# ------------------------------
# Create geometry for confinement
# ------------------------------
prams["Nbd"] = 32
prams["nvbd"] = 0
# prams["nvbd"] = 0

t = torch.linspace(0, 2 * torch.pi, prams["Nbd"])  # end_point = False
rad1 = 1.0  # inner cylinder radius
rad2 = 2.0  # outer cylinder radius

# Outer and inner walls
x = torch.cat([rad2 * torch.cos(t), rad1 * torch.cos(-t)])
y = torch.cat([rad2 * torch.sin(t), rad1 * torch.sin(-t)])
# Xwalls = torch.vstack((x, y))
Xwalls = None

# ------------------------------
# Create vesicles
# ------------------------------

# Initial shape

sigma = None
X = torch.from_numpy(Xics).float().to(device)
X = interpft_vec(X, prams["N"]).to(device)


# ------------------------------
# Simulation parameters and options
# ------------------------------
prams["T"] = prams["T"] * prams["dt"]
prams["kappa"] = 1.0
prams["viscCont"] = torch.ones(prams["nv"])
prams["gmresTol"] = 1e-10
prams["areaLenTol"] = 1e-2
prams["repStrength"] = 1e5
prams["minDist"] = 1.0 / 32

# ------------------------------
# Initialize default values (if any missing)
# ------------------------------
options, prams = initVes2D(options, prams)

# ------------------------------
# Get original area and length
# ------------------------------
_, area0, len0 = oc.geomProp(X)
print("area0: ", area0)
print("len0: ", len0)

with open(fileName, "wb") as fid:
    np.array([prams["N"], prams["nv"]]).flatten().astype("float64").tofile(fid)
    X.cpu().numpy().T.flatten().astype("float64").tofile(fid)


# ------------------------------
# Time stepping object
# ------------------------------

tt = TStepBiem(X, Xwalls, options, prams, rank, size, device)

if options["confined"]:
    tt.initialConfined()

# ------------------------------
# Initialize variables
# ------------------------------
sigma = torch.zeros(prams["N"], prams["nv"], dtype=torch.float64) if sigma is None else sigma
eta = torch.zeros(2 * prams["Nbd"], prams["nvbd"])
RS = torch.zeros(3, prams["nvbd"])

# ------------------------------
# Display setup
# ------------------------------
print(f"{prams['nv']} vesicle(s) in {options['farField']} flow, dt: {prams['dt']}")
print(f"Vesicle(s) discretized with {prams['N']} points")
print(f"we are using {X.dtype}")
if options["confined"]:
    print(f"Wall(s) discretized with {prams['Nbd']} points")

# ------------------------------
# Time loop
# ------------------------------
time_ = 0.0
modes = torch.concatenate(
    (torch.arange(0, prams["N"] // 2), torch.arange(-prams["N"] // 2, 0))
).to(X.device)  # .double()
if prams["nv"] % size != 0:
    raise ValueError(
        f"nv={prams['nv']} must be divisible by world_size={size} for distributed_tstep_biem_rewritten"
    )
print(prams)

print("GMRES max iter:", prams["gmresMaxIter"])
print("is cuda available:", torch.cuda.is_available())

print("X shape: ", X.shape)
print("sigma shape: ", sigma.shape)
print("eta shape: ", eta.shape)
print("RS shape: ", RS.shape)

t0 = time.time()

for step in tqdm(range(int(prams["T"] / prams["dt"]))):
    # Perform time step
    #print(X.dtype, sigma.dtype)
    Xnew, sigma, eta, RS, iter_, iflag = tt.time_step(X, sigma, eta, RS)
    sigma = torch.zeros(prams["N"], prams["nv"]) if sigma is None else sigma
    eta = torch.zeros(2 * prams["Nbd"], prams["nvbd"])
    RS = torch.zeros(3, prams["nvbd"])

    if options["reparameterization"]:
        # Redistribute arc-length
        XnewO = Xnew.clone()
        for _ in range(5):
            Xnew, allGood = oc.redistributeArcLength(Xnew, modes)
        X = oc.alignCenterAngle(XnewO, Xnew)
    else:
        X = Xnew

    # start.record()
    if options["correctShape"]:
        X = oc.correctAreaAndLengthAugLag(X.float(), area0, len0)
        # X = X.double()

    # X = filterShape(X, modeCut=10)

    # Update simulation time
    time_ += prams["dt"]

    # Display timestep info
    #print("*****************************************************************")
    #print(f"Time: {step} step, out of Tf: {prams['T']}")
    #print(f"GMRES took {iter_} matvecs, successful {not iflag}")
    #print("*****************************************************************")

    if rank == 0:
        output = np.concatenate(([time_], X.cpu().numpy().T.flatten())).astype("float64")
        with open(fileName, "ab") as fid:
            output.tofile(fid)
t1 = time.time()

if comm_info.rank == 0:
    print("Timed parareal solve:", t1 - t0)
