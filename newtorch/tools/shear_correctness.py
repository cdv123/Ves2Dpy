import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from load_ves2d_file import load_ves2d_file

filename = "Verification/TwoVesicles/pararealVesNet/oneIter5e6_1e6.bin"
vesx, vesy, time, N, nv, xinit, yinit = load_ves2d_file(filename)

# vesx, vesy shapes assumed: (N, nv, ntime)
assert nv == 2

ntime = vesx.shape[2]
dist = np.empty(ntime)

for k in range(ntime):
    x1, y1 = vesx[:, 0, k], vesy[:, 0, k]
    x2, y2 = vesx[:, 1, k], vesy[:, 1, k]

    dx = x1[:, None] - x2[None, :]
    dy = y1[:, None] - y2[None, :]

    dist[k] = np.sqrt(dx**2 + dy**2).min()

plt.figure()
plt.plot(time, dist)
plt.xlabel("time")
plt.ylabel("minimum distance between vesicles")
plt.title("Vesicle separation over time")
plt.grid(True)
plt.show()
