import numpy as np
import inspect
import sys
import torch
import matplotlib.pyplot as plt
from load_ves2d_file import load_ves2d_file_singleX
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from curve_batch_compile import Curve


def compute_errors(parareal_file, ground_file, parareal_stride=1, ground_stride=1):
    parareal_x, parareal_time, _, _, _ = load_ves2d_file_singleX(parareal_file)
    ground_x, ground_time, _, _, _ = load_ves2d_file_singleX(ground_file)

    curve = Curve()

    par_indices = range(0, len(parareal_time), parareal_stride)
    gnd_indices = range(0, len(ground_time), ground_stride)

    center_diffs = []
    angle_errors = []
    t = []

    for par_i, gnd_i in zip(par_indices, gnd_indices):
        par_x = torch.from_numpy(parareal_x[:, :, par_i])
        gnd_x = torch.from_numpy(ground_x[:, :, gnd_i])

        c_par = curve.getPhysicalCenterNotCompiled_(par_x)
        c_gnd = curve.getPhysicalCenterNotCompiled_(gnd_x)
        center_diffs.append(torch.norm(c_par - c_gnd))

        theta_par = curve.getIncAngle(par_x)
        theta_gnd = curve.getIncAngle(gnd_x)

        dtheta = torch.abs(theta_par - theta_gnd)
        dtheta = torch.minimum(dtheta, torch.pi - dtheta)
        angle_errors.append(torch.mean(dtheta) * 180.0 / torch.pi)

        t.append(parareal_time[par_i])

    center_diffs = torch.stack(center_diffs).cpu().numpy()
    angle_errors = torch.stack(angle_errors).cpu().numpy()
    t = np.array(t)

    return t, center_diffs, angle_errors


def plot_errors(
    parareal_file,
    ground_file,
    label=None,
    ax_center=None,
    ax_angle=None,
    parareal_stride=1,
    ground_stride=1,
    labels=[],
):
    t, center_err, angle_err = compute_errors(
        parareal_file,
        ground_file,
        parareal_stride=parareal_stride,
        ground_stride=ground_stride,
    )

    if ax_center is None or ax_angle is None:
        fig, (ax_center, ax_angle) = plt.subplots(1, 2, figsize=(10, 4))

    ax_center.plot(t, center_err, label=label)
    ax_center.set_xlabel("Time")
    ax_center.set_ylabel("Center error")
    ax_center.grid(True)

    ax_angle.plot(t, angle_err, label=label)
    ax_angle.set_xlabel("Time")
    ax_angle.set_ylabel("Angle error (deg)")
    ax_angle.grid(True)
    print(label, angle_err[-1])
    print(label, center_err[-1])

    return ax_center, ax_angle


fig, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(10, 4))

plot_errors(
    "Verification/OneVesicle/pararealVesNet/oneIter2e5_1e5.bin",
    "Verification/OneVesicle/groundTruth.bin",
    label="one iteration",
    ax_center=ax_c,
    ax_angle=ax_a,
    ground_stride=10,
)

plot_errors(
    "Verification/OneVesicle/pararealVesNet/twoIter2e5_1e5.bin",
    "Verification/OneVesicle/groundTruth.bin",
    label="two iterations",
    ax_center=ax_c,
    ax_angle=ax_a,
    ground_stride=10,
)
plot_errors(
    "Verification/OneVesicle/pararealVesNet/threeIter2e5_1e5.bin",
    "Verification/OneVesicle/groundTruth.bin",
    label="three iterations",
    ax_center=ax_c,
    ax_angle=ax_a,
    ground_stride=10,
)
plot_errors(
    "Verification/OneVesicle/pararealVesNet/fourIter2e5_1e5.bin",
    "Verification/OneVesicle/groundTruth.bin",
    label="four iterations",
    ax_center=ax_c,
    ax_angle=ax_a,
    ground_stride=10,
)

ax_c.legend()
ax_a.legend()
ax_c.set_yscale("log")
ax_a.set_yscale("log")

ax_c.set_title("Centre Error vs Time")
ax_a.set_title("Inclination Angle Error vs Time")

plt.show()
