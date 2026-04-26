import numpy as np
import matplotlib.pyplot as plt
from load_ves2d_file import load_ves2d_file


def compute_start_indices(time_file, time_ref, fraction, s_file, s_ref):
    """
    fraction: e.g. 0.5 for halfway
    returns (start_file, start_ref)
    """
    t_start = fraction * min(time_file[-1], time_ref[-1])

    start_file = np.searchsorted(time_file, t_start)
    start_ref = np.searchsorted(time_ref, t_start)

    # snap to stride grid
    start_file = (start_file // s_file) * s_file
    start_ref = (start_ref // s_ref) * s_ref

    return start_file, start_ref


def vesicle_distance(vesx, vesy):
    """Minimum pairwise distance between two vesicles over time."""
    ntime = vesx.shape[2]
    dist = np.empty(ntime)

    for k in range(ntime):
        x1, y1 = vesx[:, 0, k], vesy[:, 0, k]
        x2, y2 = vesx[:, 1, k], vesy[:, 1, k]
        dx = x1[:, None] - x2[None, :]
        dy = y1[:, None] - y2[None, :]
        dist[k] = np.sqrt(dx**2 + dy**2).min()

    return dist


def compare_files(file_list, reference_file, strides, labels):
    """
    file_list: list of files to compare
    reference_file: ground truth
    strides: list of (file_stride, ref_stride)
    starts: list of (file_start, ref_start)
    """

    vesx_ref, vesy_ref, time_ref, *_ = load_ves2d_file(reference_file)
    dist_ref = vesicle_distance(vesx_ref, vesy_ref)

    plt.figure()

    for file, (s_file, s_ref), label in zip(file_list, strides, labels):
        vesx, vesy, time, *_ = load_ves2d_file(file)
        dist = vesicle_distance(vesx, vesy)
        start_file = len(time) // 2
        start_ref = len(time_ref) // 2

        start_file = 0
        start_ref = 0

        errors = []
        t_vals = []

        idx_file = range(start_file, len(time), s_file)
        idx_ref = range(start_ref, len(time_ref), s_ref)

        for i_f, i_r in zip(idx_file, idx_ref):
            errors.append(abs(dist[i_f] - dist_ref[i_r]))
            t_vals.append(time[i_f])

        plt.plot(t_vals, errors, label=label)

    plt.xlabel("Time")
    plt.ylabel("Distance error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


files = [
    "Verification/TwoVesicles/pararealVesNet/oneIter1e5_5e6FourCore.bin",
    "Verification/TwoVesicles/pararealVesNet/twoIter1e5_5e6FourCore.bin",
]

starts = [
    (0, 0),
    (0, 0),
]


strides = [
    (1, 5),
    (1, 5),
]

compare_files(
    files,
    "Verification/TwoVesicles/groundTruth.bin",
    strides,
    labels=[
        "One iter",
        "Two iter",
    ],
)
