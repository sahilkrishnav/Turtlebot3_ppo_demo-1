import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
N = 1000  # number of steps
delta_t = 0.1  # time step

# True B(x) parameters
k1_true = 1.2
k2_true = 0.8
k3_true = 0.9

# Noise level (simulate real-world sensor noise)
noise_std_position = 0.005  # ~5mm noise
noise_std_theta = 0.005     # ~0.005 rad (~0.3 degrees)

# Generate random actions
np.random.seed(42)
v = np.random.uniform(0.1, 0.3, N)
omega = np.random.uniform(-1.0, 1.0, N)

# Initialize states
states = np.zeros((N, 3))  # [x, y, theta]

# Simulate motion with noise
for t in range(1, N):
    theta = states[t - 1, 2]
    
    # Ideal update
    dx = k1_true * np.cos(theta) * v[t - 1] * delta_t
    dy = k2_true * np.sin(theta) * v[t - 1] * delta_t
    dtheta = k3_true * omega[t - 1] * delta_t

    # Add noise to state updates
    dx_noisy = dx + np.random.normal(0, noise_std_position)
    dy_noisy = dy + np.random.normal(0, noise_std_position)
    dtheta_noisy = dtheta + np.random.normal(0, noise_std_theta)

    states[t, 0] = states[t - 1, 0] + dx_noisy
    states[t, 1] = states[t - 1, 1] + dy_noisy
    states[t, 2] = states[t - 1, 2] + dtheta_noisy

# Prepare data like real experiment
state = states[:-1]
next_state = states[1:]
action = np.vstack([v[:-1], omega[:-1]]).T

# Optional: save pseudo data
np.savez('pseudo_data_with_noise.npz', state=state, action=action, next_state=next_state)

print("Pseudo data with noise generated.")

# === Estimation code ===

# Estimate derivatives
dx = (next_state[:, 0] - state[:, 0]) / delta_t
dy = (next_state[:, 1] - state[:, 1]) / delta_t
dtheta = (next_state[:, 2] - state[:, 2]) / delta_t

# Prepare regression inputs
theta = state[:, 2]

A_dx = (action[:, 0] * np.cos(theta)).reshape(-1, 1)
A_dy = (action[:, 0] * np.sin(theta)).reshape(-1, 1)
A_dtheta = action[:, 1].reshape(-1, 1)

# Least squares estimation
k1, _, _, _ = np.linalg.lstsq(A_dx, dx, rcond=None)
k2, _, _, _ = np.linalg.lstsq(A_dy, dy, rcond=None)
k3, _, _, _ = np.linalg.lstsq(A_dtheta, dtheta, rcond=None)

print("\nEstimated parameters of B(x):")
print(f"k1 (true {k1_true}): {k1[0]:.6f}")
print(f"k2 (true {k2_true}): {k2[0]:.6f}")
print(f"k3 (true {k3_true}): {k3[0]:.6f}")

# Prediction errors
dx_pred = A_dx @ k1
dy_pred = A_dy @ k2
dtheta_pred = A_dtheta @ k3

dx_error = dx_pred - dx
dy_error = dy_pred - dy
dtheta_error = dtheta_pred - dtheta

mse_dx = np.mean(dx_error ** 2)
mse_dy = np.mean(dy_error ** 2)
mse_dtheta = np.mean(dtheta_error ** 2)

print("\nMean Squared Errors:")
print(f"MSE dx: {mse_dx:.8f}")
print(f"MSE dy: {mse_dy:.8f}")
print(f"MSE dtheta: {mse_dtheta:.8f}")

# === Plot errors ===
time_axis = np.arange(len(dx)) * delta_t

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_axis, dx_error, label='dx error', color='red')
plt.title('dx Prediction Error')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_axis, dy_error, label='dy error', color='green')
plt.title('dy Prediction Error')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_axis, dtheta_error, label='dtheta error', color='blue')
plt.title('dtheta Prediction Error')
plt.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
