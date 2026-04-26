import numpy as np
import matplotlib.pyplot as plt
from load_ves2d_file import load_ves2d_file


def count_vesicles_in_cell_single_file(filename, vsize=5.0, drop_first=False):
    """
    Compute the MATLAB-style 'number of vesicles in the cell' statistic
    for a single Ves2D simulation file.

    Parameters
    ----------
    filename : str
        Path to the binary simulation file.
    vsize : float, default=5.0
        Side length of the square cell. The cell is [0, vsize] x [0, vsize].
    drop_first : bool, default=False
        If True, discard the first time step, matching the MATLAB handling
        used for some files.

    Returns
    -------
    result : dict with keys
        time : ndarray, shape (ntime,) or (ntime-1,)
        nves_in_cell : ndarray, shape (ntime,) or (ntime-1,)
        frac_in_cell : ndarray
            Fraction of vesicles in the cell.
        percent_in_cell : ndarray
            Percentage of vesicles in the cell.
        cx : ndarray, shape (nv, ntime) or (nv, ntime-1)
            Vesicle x-centroids.
        cy : ndarray, shape (nv, ntime) or (nv, ntime-1)
            Vesicle y-centroids.
        nv : int
        N : int
    """
    vesx, vesy, time, N, nv, xinit, yinit = load_ves2d_file(filename)

    # vesx, vesy expected shape: (N, nv, ntime)
    # Mean over boundary points -> centroids, shape (nv, ntime)
    cx = np.mean(vesx, axis=0)
    cy = np.mean(vesy, axis=0)

    # Same condition as MATLAB:
    # abs(cx - vsize/2) <= vsize/2 and abs(cy - vsize/2) <= vsize/2
    # which is equivalent to 0 <= cx <= vsize and 0 <= cy <= vsize
    inside = (np.abs(cx - vsize / 2) <= vsize / 2) & (
        np.abs(cy - vsize / 2) <= vsize / 2
    )

    # Count vesicles inside for each time step
    nves_in_cell = np.sum(inside, axis=0)

    if drop_first:
        time = time[1:]
        cx = cx[:, 1:]
        cy = cy[:, 1:]
        nves_in_cell = nves_in_cell[1:]

    frac_in_cell = nves_in_cell / nv
    percent_in_cell = 100.0 * frac_in_cell

    return {
        "time": time,
        "nves_in_cell": nves_in_cell,
        "frac_in_cell": frac_in_cell,
        "percent_in_cell": percent_in_cell,
        "cx": cx,
        "cy": cy,
        "nv": nv,
        "N": N,
    }


fine_result = count_vesicles_in_cell_single_file(
    "taylorResults/N128biem32.bin", vsize=2.5
)
coarse_result = count_vesicles_in_cell_single_file(
    "taylorResults/N128coarseBiem32.bin", vsize=2.5
)
# parareal_result = count_vesicles_in_cell_single_file(
#    "taylorResults/pararealVesNetSameDt.bin", vsize=2.5
# )
# parareal_diff_dt_result = count_vesicles_in_cell_single_file(
#    "taylorResults/pararealVesNet6e51e5.bin", vsize=2.5
# )
fine_parareal = count_vesicles_in_cell_single_file(
    "taylorResults/pararealSameDtVesnet.bin", vsize=2.5
)
coarse_parareal = count_vesicles_in_cell_single_file(
    "taylorResults/parareal2e51e5Vesnet.bin", vsize=2.5
)

plt.figure(figsize=(7, 5))
plt.plot(
    fine_result["time"],
    fine_result["percent_in_cell"],
    linewidth=2,
    label="Fine boundary integral method",
)
plt.plot(
    coarse_result["time"],
    coarse_result["percent_in_cell"],
    linewidth=2,
    label="Coarse boundary integral method",
)
# plt.plot(
#    parareal_result["time"],
#    parareal_result["percent_in_cell"],
#    linewidth=2,
#    label="Parareal+VesNet with both coarse and fine dt = 1e-5",
# )
# plt.plot(
#    parareal_diff_dt_result["time"],
#    parareal_diff_dt_result["percent_in_cell"],
#    linewidth=2,
# )
plt.plot(
    fine_parareal["time"],
    fine_parareal["percent_in_cell"],
    label="Parareal with dt=1e-5",
    linewidth=2,
)
plt.plot(
    coarse_parareal["time"],
    coarse_parareal["percent_in_cell"],
    label="Parareal with fine dt=1e-5, coarse dt=2e-5",
    linewidth=2,
)

plt.xlabel("Time")
plt.ylabel("% of vesicles in the cell")
plt.legend()
plt.tight_layout()
plt.show()
