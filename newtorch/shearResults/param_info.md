# Ground Truth params

prams["N"] = X.shape[0] // 2
prams["nv"] = X.shape[1]
prams["dt"] = 1e-6
prams["T"] = 10000 * prams["dt"]
prams["kappa"] = 1.0
prams["viscCont"] = torch.ones(prams["nv"])
prams["gmresTol"] = 1e-10
prams["areaLenTol"] = 1e-2
prams["vortexSize"] = 2.5
prams["chanWidth"] = 2.5
prams["farFieldSpeed"] = 2000

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

