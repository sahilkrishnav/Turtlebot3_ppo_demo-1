import numpy as np

# Load data
data = np.load('mdp_data.npz')

states = data['state']         # (N, 3)
actions = data['action']       # (N, 2)
next_states = data['next_state']  # (N, 3)

delta_t = 0.1

# Estimate derivatives
dx = (next_states[:, 0] - states[:, 0]) / delta_t
dy = (next_states[:, 1] - states[:, 1]) / delta_t
dtheta = (next_states[:, 2] - states[:, 2]) / delta_t

# Prepare regression inputs
v = actions[:, 0]
omega = actions[:, 1]
theta = states[:, 2]

# Design regression matrices
A_dx = (v * np.cos(theta)).reshape(-1, 1)
A_dy = (v * np.sin(theta)).reshape(-1, 1)
A_dtheta = omega.reshape(-1, 1)

# Perform least squares estimation
k1, _, _, _ = np.linalg.lstsq(A_dx, dx, rcond=None)
k2, _, _, _ = np.linalg.lstsq(A_dy, dy, rcond=None)
k3, _, _, _ = np.linalg.lstsq(A_dtheta, dtheta, rcond=None)

# Print results clearly
print("Estimated parameters of B(x):")
print(f"k1 (scaling on cos(theta) * v): {k1[0]:.6f}")
print(f"k2 (scaling on sin(theta) * v): {k2[0]:.6f}")
print(f"k3 (scaling on omega): {k3[0]:.6f}")

# Predict
dx_pred = A_dx @ k1
dy_pred = A_dy @ k2
dtheta_pred = A_dtheta @ k3

# Errors
dx_error = np.abs(dx_pred - dx)
dy_error = np.abs(dy_pred - dy)
dtheta_error = np.abs(dtheta_pred - dtheta)

mse_dx = np.mean(dx_error ** 2)
mse_dy = np.mean(dy_error ** 2)
mse_dtheta = np.mean(dtheta_error ** 2)

print("\nL1 Errors:")
print(f"L1 dx: {dx_error:.8f}")
print(f"L1 dy: {dy_error:.8f}")
print(f"L1 dtheta: {dtheta_error:.8f}")

print("\nMean Squared Errors:")
print(f"MSE dx: {mse_dx:.8f}")
print(f"MSE dy: {mse_dy:.8f}")
print(f"MSE dtheta: {mse_dtheta:.8f}")
