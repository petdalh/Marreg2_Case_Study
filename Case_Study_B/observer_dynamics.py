import numpy as np
import matplotlib.pyplot as plt
from wrap import wrap


M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])

D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])

T_b = np.diag([124, 124, 124])
L1 = np.diag([10, 10, 10])
L2 = np.diag([1, 1, 1])
L3 = np.diag([0.1, 0.1, 0.1])


def create_R(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])


# Initial guess of estimated variables
# eta_thilde = np.zeros((3, 1), dtype=float)
# nu_thilde = np.zeros((3, 1), dtype=float)
# bias_thilde = np.zeros((3, 1), dtype=float)
x = np.array([1,1,np.pi/4, 0.1, 0.1, 0, 0.1, -0.1, 0.01]).reshape(9,1)

# Generate eta for simulation
dt = 0.01
num_steps = 10000

history = np.zeros((num_steps, 9))

for i in range(num_steps):
    # Measurement noise
    w = 0.00001 * np.random.normal(size=(3, 1))

    # Create R
    psi = wrap(x[2, 0] + w[2, 0])
    R = create_R(psi)

    A = np.block(
        [[-L1, R, np.zeros((3, 3))],
         [
             -np.linalg.inv(M) @ L2 @ np.transpose(R), -np.linalg.inv(M) @ D,
             np.linalg.inv(M)
         ], [-L3 @ np.transpose(R),
             np.zeros((3, 3)), -np.linalg.inv(T_b)]])

    b = np.vstack([-L1, R, -L3 @ np.transpose(R)])

    x = (np.eye(9, 9) + dt * A) @ x + dt * b @ w

    history[i, :] = x.flatten()


# Create time vector
time = np.arange(0, num_steps) * dt

# Create a figure with subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('State History', fontsize=16)

# State variable names (adjust these to match your system's variables)
state_names = [
    'Position x', 'Position y', 'Heading ψ',
    'Velocity u', 'Velocity v', 'Velocity r',
    'Bias x', 'Bias y', 'Bias ψ'
]

# Plot each state variable
for i in range(9):
    row, col = i // 3, i % 3
    axs[row, col].plot(time, history[:, i])
    axs[row, col].set_title(state_names[i])
    axs[row, col].set_xlabel('Time (s)')
    axs[row, col].grid(True)

# Add a specific ylabel for each row
axs[0, 0].set_ylabel('Position/Angle')
axs[1, 0].set_ylabel('Velocity')
axs[2, 0].set_ylabel('Bias')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
plt.show()

# You can also create separate plots for related variables
# For example, plot positions together
plt.figure(figsize=(10, 6))
plt.plot(time, history[:, 0], label='Position x')
plt.plot(time, history[:, 1], label='Position y')
plt.plot(time, history[:, 2], label='Heading ψ')
plt.title('Position and Heading')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot velocities together
plt.figure(figsize=(10, 6))
plt.plot(time, history[:, 3], label='Velocity u')
plt.plot(time, history[:, 4], label='Velocity v')
plt.plot(time, history[:, 5], label='Velocity r')
plt.title('Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot biases together
plt.figure(figsize=(10, 6))
plt.plot(time, history[:, 6], label='Bias x')
plt.plot(time, history[:, 7], label='Bias y')
plt.plot(time, history[:, 8], label='Bias ψ')
plt.title('Biases')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Bonus: Visualize the 2D trajectory
plt.figure(figsize=(8, 8))
plt.plot(history[:, 0], history[:, 1])
plt.scatter(history[0, 0], history[0, 1], color='green', s=100, label='Start')
plt.scatter(history[-1, 0], history[-1, 1], color='red', s=100, label='End')
plt.title('2D Trajectory')
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.axis('equal')  # Equal scaling for x and y axes
plt.grid(True)
plt.legend()
plt.show()
