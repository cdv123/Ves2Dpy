# Params Used

## Ground Truth
prams["dt"] = 1e-6
prams["T"] = 10000 * prams["dt"]
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
    "farField": "taylorGreen",
    "repulsion": False,
    "correctShape": True,
    "reparameterization": False,
    "usePreco": True,
    "matFreeWalls": False,
    "confined": False,
}

## Coarse BIEM
same time horizion but dt = 1e-5
