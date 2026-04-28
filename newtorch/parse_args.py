import argparse
from scipy.io import loadmat


#PARAM_DEFAULTS = {
#    "dt": 2e-6,
#    "T": 600,
#    "N": 32,
#    "nv": 120,
#    "pr": 1,
#    "farFieldSpeed": 400,
#    "chanWidth": 2.5,
#    "vortexSize": 2.5,
#    "coarse_dt": 5e-6,
#    "nProcs": 6,
#    # "input": "../../npy-files/shearIC.mat",
#    "input": "../../npy-files/VF25_TG128Ves.mat",
#    #"input": "../../npy-files/VF25_TG32Ves.mat",
#    #"output": "output_BIEM/speed.bin",
#    #"output": "output_BIEM/speed4.bin",
#    "output": None,
#    "nProcsVesNet": 6
#}

PARAM_DEFAULTS = {
    "dt": 5e-6,
    "T": 500,
    "N": 128,
    "nv": 32,
    "pr": 1,
    "farFieldSpeed": 400,
    "chanWidth": 2.5,
    "vortexSize": 2.5,
    "coarse_dt": 2e-5,
    "nProcs": 4,
    # "input": "../../npy-files/shearIC.mat",
    "input": "../../npy-files/VF25_TG128Ves.mat",
    #"input": "../../npy-files/VF25_TG32Ves.mat",
    #"output": "output_BIEM/speed.bin",
    "output": None,
    #"output": "output_BIEM/new_test.bin",
    "nProcsVesNet": 4
}

#PARAM_DEFAULTS = {
#    "dt": 5e-6,
#    "T": 1000,
#    "N": 128,
#    "nv": 1,
#    "pr": 1,
#    "farFieldSpeed": 400,
#    "chanWidth": 2.5,
#    "vortexSize": 2.5,
#    "coarse_dt": 1e-5,
#    "nProcs": 2,
#    #"input": "../../npy-files/shearIC.mat",
#    #"input": "../../npy-files/VF25_TG128Ves.mat",
#    "input": "../../npy-files/VF25_TG32Ves.mat",
#    #"output": "output_BIEM/speed.bin",
#    "output": "output_BIEM/2e5_one_ves.bin",
#    "nProcsVesNet": 1
#}

ARG_TO_PARAM = {
    "dt": "dt",
    "T": "T",
    "N": "N",
    "nv": "nv",
    "pr": "pr",
    "speed": "farFieldSpeed",
    "cw": "chanWidth",
    "vs": "vortexSize",
    "coarse_dt": "coarse_dt",
    "nProcs": "nProcs",
    "input": "input",
    "output": "output",
    "nProcsVesNet": "nProcsVesNet"
}


def parse_cli():
    parser = argparse.ArgumentParser(description="Vesicle BIEM simulation")

    # --- Simulation controls ---
    parser.add_argument("--dt", type=float, help="Time step size")
    parser.add_argument(
        "--T", type=int, help="Final time (in number of coarse time steps)"
    )
    parser.add_argument(
        "--wT", type=int, help="Windowed time (in number of time steps)"
    )
    parser.add_argument("--N", type=int, help="Number of discretization points")
    parser.add_argument("--nv", type=int, help="Number of vesicles")
    parser.add_argument("--pr", type=str, help="Number of parareal iterations")

    # --- Flow / options ---
    parser.add_argument(
        "--farField", type=str, choices=["shear", "taylorGreen"], help="Flow type"
    )
    parser.add_argument("--vs", type=float, help="Vortex size")
    parser.add_argument("--cw", type=float, help="Channel width")
    parser.add_argument("--speed", type=float, help="Far field speed")

    # --- IO / runtime ---
    parser.add_argument("--input", type=str, help="Input .mat file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument(
        "--use_vesnet", action="store_true", help="Use VesNet with parareal"
    )
    parser.add_argument(
        "--coarse_dt", type=float, help="Time step size of coarse solver"
    )
    parser.add_argument("--nProcs", type=int, help="Number of processes to use")
    parser.add_argument("--nProcsVesNet", type=int, help="Number of processes to use")

    return parser.parse_args()


def modify_options_params(args, options, params):
    # Apply scalar params from CLI when present, otherwise apply defaults
    for arg_name, param_name in ARG_TO_PARAM.items():
        value = getattr(args, arg_name)
        if value is not None:
            params[param_name] = value
        elif param_name in PARAM_DEFAULTS:
            params[param_name] = PARAM_DEFAULTS[param_name]

    # Derived default depends on T
    params["window_size"] = args.wT if args.wT is not None else min(5000, params["T"])

    # Flags
    params["use_vesnet"] = args.use_vesnet
    #params["use_vesnet"] = True

    # Options
    options["farField"] = args.farField if args.farField is not None else "taylorGreen"
    #options["farField"] = args.farField if args.farField is not None else "shear"

    # IO
    file_name = params["output"] 
    if params["farFieldSpeed"] == 2000:
        Xics = loadmat(params["input"]).get("Xic")[:, : params["nv"]]
        N = 128   
        Xics[:N, 0] -= 0.05
    else:
        Xics = loadmat(params["input"]).get("X")[:, : params["nv"]]

    return file_name, Xics
