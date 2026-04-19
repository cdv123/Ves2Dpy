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
