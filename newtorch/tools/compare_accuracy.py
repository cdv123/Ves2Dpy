import numpy as np
import torch
import matplotlib.pyplot as plt
from load_ves2d_file import load_ves2d_file_singleX
import os
import sys
import inspect


def get_percent_diff_metric(parareal_x, ground_x, metric):
    c_par = metric(parareal_x)
    c_gnd = metric(ground_x)

    diff = torch.norm(c_par - c_gnd)

    # percent_error = 100.0 * diff / (ref + 1e-12)

    return diff


def get_angle_error_metric(parareal_x, ground_x, metric):
    theta_par = metric(parareal_x)  # shape (nv,)
    theta_gnd = metric(ground_x)

    # wrapped angular difference in [0, pi/2]
    dtheta = torch.abs(theta_par - theta_gnd)
    dtheta = torch.minimum(dtheta, torch.pi - dtheta)

    # Return mean angular error (degrees)
    return torch.mean(dtheta) * 180.0 / torch.pi


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from curve_batch_compile import Curve

# --- Load data ---
filename = "output_BIEM/vesnet.bin"
parareal_x, parareal_time, N, nv, parareal_xinit = load_ves2d_file_singleX(filename)

filename = "output_BIEM/ground_truth.bin"
ground_x, ground_time, N, nv, ground_xinit = load_ves2d_file_singleX(filename)

parareal_ntime = parareal_time.size
ground_ntime = ground_time.size

curve = Curve()

center_diffs = []
angle_errors = []

for parareal_it, ground_it in zip(range(parareal_ntime), range(0, ground_ntime, 10)):
    ground_x_i = torch.from_numpy(ground_x[:, :, ground_it])
    parareal_x_i = torch.from_numpy(parareal_x[:, :, parareal_it])

    diff_center = get_percent_diff_metric(
        parareal_x_i, ground_x_i, curve.getPhysicalCenterNotCompiled_
    )

    angle_error_deg = get_angle_error_metric(
        parareal_x_i, ground_x_i, curve.getIncAngle
    )

    center_diffs.append(diff_center)
    angle_errors.append(angle_error_deg)

    # print(
    #     f"Step {parareal_it}: center error = {diff_center.item():.6f}: angle degree error = {angle_error_deg.item():.6f}"
    # )

center_diffs_np = torch.stack(center_diffs).cpu().numpy()
angle_errors_np = torch.stack(angle_errors).cpu().numpy()

t_np = parareal_time[: len(center_diffs_np)]  # time axis (same length)

# =========================
# Save raw errors to file
# =========================
out = np.column_stack((t_np, center_diffs_np, angle_errors_np))

np.savetxt(
    "parareal_vs_ground_errors_ground_truth.txt",
    out,
    header="time   center_error   angle_error_deg",
)

print("Saved error data to parareal_vs_ground_errors_ground_truth.txt")

# =========================
# Plot center error
# =========================
plt.figure()
plt.plot(t_np, center_diffs_np)
plt.xlabel("Time")
plt.ylabel("Center error (absolute)")
plt.title("Parareal vs BIE same time step: Center Error")
plt.grid(True)
plt.savefig("center_error_same_time_step.png", dpi=300)

# =========================
# Plot angle error
# =========================
plt.figure()
plt.plot(t_np, angle_errors_np)
plt.xlabel("Time")
plt.ylabel("Angle error (degrees)")
plt.title("Parareal vs BIE same time step: Inclination Angle Error")
plt.grid(True)
plt.savefig("angle_error_ground_truth.png", dpi=300)

plt.show()

print("Saved plots: center_error_ground_truth.png, angle_error_ground_truth.png")
