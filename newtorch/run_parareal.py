import torch
import sys

torch.set_default_dtype(torch.float32)
import numpy as np
from curve_batch_compile import Curve
from capsules import capsules
import time
from tstep_biem import TStepBiem
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from tools.filter import filterShape, interpft_vec
from parareal_algorithm import PararealSolver
from parareal_solvers import ParallelSolver, BIEMSolver

torch.set_default_dtype(torch.float64)


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
        "reparameterization": False,
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

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device("cuda")
    
    fileName = "./output_BIEM/parareal_output.bin"  # To save simulation data
    
    
    # Assume oc is your geometry utility class (like curve_py in MATLAB)
    oc = Curve()  # You need to define this with required methods
    
    # ------------------------------
    # Create geometry for confinement
    # ------------------------------
    prams = {}
    prams["Nbd"] = 32
    # prams['nvbd'] = 2
    prams["nvbd"] = 0
    
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
    selected_vesicle = [0]
    Xics = loadmat("../../npy-files/VF25_TG32Ves.mat").get("X")[:, selected_vesicle]
    Xics = Xics - Xics.mean()
    
    sigma = None
    X = torch.from_numpy(Xics).float().to(device)
    X = interpft_vec(X, 128).to(device)
    
    
    # ------------------------------
    # Simulation parameters and options
    # ------------------------------
    prams["N"] = X.shape[0] // 2
    prams["nv"] = X.shape[1]
    prams["dt"] = 1e-5
    prams["T"] = 1000 * prams["dt"]
    prams["kappa"] = 1.0
    prams["viscCont"] = torch.ones(prams["nv"])
    prams["gmresTol"] = 1e-10
    prams["areaLenTol"] = 1e-2
    prams["vortexSize"] = 2.5
    prams["chanWidth"] = 2.5
    prams["farFieldSpeed"] = 400
    
    prams["repStrength"] = 1e5
    prams["minDist"] = 1.0 / 32
    
    options = {
        "farField": "shear",
        "repulsion": False,
        "correctShape": True,
        "reparameterization": True,
        "usePreco": True,
        "matFreeWalls": False,
        "confined": False,
    }
    
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
    sigma = torch.zeros(prams["N"], prams["nv"]) if sigma is None else sigma
    print(prams)
    print(options)
    
    
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
    
    numCores = 5
    prams["T"] /= numCores
    coarse_prams = prams.copy()
    
    # Use a larger time step size for coarse solver
    coarse_prams["dt"]*=10
    
    coarseSolver = BIEMSolver(options, coarse_prams, Xwalls, X)
    parallelSolver = ParallelSolver(options, prams, Xwalls, X, sigma, numCores)
    
    
    pararealSolver = PararealSolver(
        parallelSolver=parallelSolver, coarseSolver=coarseSolver)
    X = pararealSolver.pararealSolve(
        initVesicles=X, sigma=sigma, numCores=numCores, endTime=prams["T"], pararealIter=2, file_name=fileName
    )
    parallelSolver.close()
    
    output = np.concatenate(([time_], X.cpu().numpy().T.flatten())).astype("float64")
