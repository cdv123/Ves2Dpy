import numpy as np


def load_ves2d_file(filename):
    """
    Python equivalent of the MATLAB function loadShanVesFile.

    Parameters
    ----------
    filename : str
        Path to the binary file containing double-precision data.

    Returns
    -------
    vesx : ndarray, shape (N, nv, ntime)
    vesy : ndarray, shape (N, nv, ntime)
    time : ndarray, shape (ntime,)
    N : int
    nv : int
    xinit : ndarray, shape (N, nv)
    yinit : ndarray, shape (N, nv)
    """

    # Read all data as double precision
    val = np.fromfile(filename, dtype=np.float64)

    # First two entries
    N = int(val[0])
    nv = int(val[1])

    # Initial configuration
    n_init = 2 * N * nv
    Xinit = val[2 : 2 + n_init]

    xinit = np.zeros((N, nv))
    yinit = np.zeros((N, nv))

    istart = 0
    for iv in range(nv):
        iend = istart + N
        xinit[:, iv] = Xinit[istart:iend]
        istart = iend

        iend = istart + N
        yinit[:, iv] = Xinit[istart:iend]
        istart = iend

    # Remaining data: time series
    val = val[2 + n_init :]

    # Number of time steps
    stride = 2 * N * nv + 1
    ntime = val.size // stride

    vesx = np.zeros((N, nv, ntime))
    vesy = np.zeros((N, nv, ntime))
    time = np.zeros(ntime)

    istart = 0
    for it in range(ntime):
        time[it] = val[istart]
        istart += 1

        for iv in range(nv):
            iend = istart + N
            vesx[:, iv, it] = val[istart:iend]
            istart = iend

            iend = istart + N
            vesy[:, iv, it] = val[istart:iend]
            istart = iend

    return vesx, vesy, time, N, nv, xinit, yinit


def load_ves2d_file_singleX(filename):
    """
    Load Ves2D binary output file and return trajectory as X(t).

    Returns
    -------
    Xtraj : ndarray, shape (2N, nv, ntime)
        Full vesicle trajectory.
    time : ndarray, shape (ntime,)
        Time values.
    N : int
    nv : int
    Xinit : ndarray, shape (2N, nv)
        Initial configuration.
    """

    # ---- Read entire file ----
    val = np.fromfile(filename, dtype=np.float64)

    # ---- Header ----
    N = int(val[0])
    nv = int(val[1])

    blocksize = 2 * N * nv

    # ---- Initial configuration ----
    Xinit_flat = val[2 : 2 + blocksize]

    # Written as X.T.flatten(), so reshape as (nv, 2N) then transpose
    Xinit = Xinit_flat.reshape(nv, 2 * N).T  # shape (2N, nv)

    # ---- Remaining data ----
    val = val[2 + blocksize :]

    stride = 1 + blocksize
    ntime = val.size // stride

    if val.size % stride != 0:
        raise ValueError("File size does not match expected record structure")

    # ---- Allocate ----
    time = np.zeros(ntime)
    Xtraj = np.zeros((2 * N, nv, ntime))

    # ---- Parse records ----
    istart = 0
    for it in range(ntime):
        # Time
        time[it] = val[istart]
        istart += 1

        # X block
        X_flat = val[istart : istart + blocksize]
        istart += blocksize

        # Rebuild X: (2N, nv)
        Xtraj[:, :, it] = X_flat.reshape(nv, 2 * N).T

    return Xtraj, time, N, nv, Xinit
