import numpy as np
import matplotlib.pyplot as plt

def wrap(angle):
    """Wraps angle to [-π, π)"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# System matrices
M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])

# Observer gains and bias time constant
T_b = np.diag([142, 142, 142])  
L1 = np.diag([10, 10, 10])
L2 = np.diag([1, 1, 1])
L3 = np.diag([0.1, 0.1, 0.1])

def create_R(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

# Initial conditions as specified in Task 4.3
eta_tilde = np.array([1, 1, np.pi/4]).reshape(3, 1)
nu_tilde = np.array([0.1, 0.1, 0]).reshape(3, 1)
b_tilde = np.array([0.1, -0.1, 0.01]).reshape(3, 1)

# Combine into state vector
x = np.vstack([eta_tilde, nu_tilde, b_tilde])

# Simulation parameters
dt = 0.01
num_steps = 20000  
noise_power = 0.00001

history = np.zeros((num_steps, 9))

for i in range(num_steps):
    # Current time
    t = i * dt
    
    # True yaw angle: ψ(t) = 0.1t as specified in Task 4.3
    psi_true = 0.1 * t
    
    # Measurement noise
    w = np.sqrt(noise_power) * np.random.normal(size=(3, 1))
    
    # Create rotation matrix with true yaw angle plus noise
    psi_noisy = psi_true + w[2, 0]
    R = create_R(psi_noisy)
    
    # Extract current state
    eta_tilde_curr = x[0:3]
    nu_tilde_curr = x[3:6]
    b_tilde_curr = x[6:9]
    
    # Compute error dynamics derivatives
    y_tilde = eta_tilde_curr + w  # Equation (17)
    
    # η̃˙ = R(ψ + w₃)ν̃ - L₁ỹ
    eta_tilde_dot = R @ nu_tilde_curr - L1 @ y_tilde
    
    # ν̃˙ = M⁻¹(-Dν̃ + b̃ - L₂R(ψ + w₃)ᵀỹ)
    nu_tilde_dot = np.linalg.inv(M) @ (-D @ nu_tilde_curr + b_tilde_curr - L2 @ R.T @ y_tilde)
    
    # b̃˙ = -T_b⁻¹b̃ - L₃R(ψ + w₃)ᵀỹ
    b_tilde_dot = -np.linalg.inv(T_b) @ b_tilde_curr - L3 @ R.T @ y_tilde
    
    # Forward Euler integration
    eta_tilde_curr += dt * eta_tilde_dot
    nu_tilde_curr += dt * nu_tilde_dot
    b_tilde_curr += dt * b_tilde_dot
    
    # Apply wrapping to yaw angle (third component of eta_tilde)
    eta_tilde_curr[2, 0] = wrap(eta_tilde_curr[2, 0])
    
    # Update state vector
    x = np.vstack([eta_tilde_curr, nu_tilde_curr, b_tilde_curr])
    
    # Store history
    history[i, :] = x.flatten()

# Create time vector
time = np.arange(0, num_steps) * dt

# Create a figure with subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('State History', fontsize=16)

# State variable names (adjust these to match your system's variables)
state_names = [
    'Position x error', 'Position y error', 'Heading ψ error',
    'Velocity u error', 'Velocity v error', 'Velocity r error',
    'Bias x error', 'Bias y error', 'Bias ψ error'
]

# Plot each state variable
for i in range(9):
    row, col = i // 3, i % 3
    axs[row, col].plot(time, history[:, i])
    axs[row, col].set_title(state_names[i])
    axs[row, col].set_xlabel('Time (s)')
    axs[row, col].grid(True)

# Add a specific ylabel for each row
axs[0, 0].set_ylabel('Position/Angle Error')
axs[1, 0].set_ylabel('Velocity Error')
axs[2, 0].set_ylabel('Bias Error')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
plt.show()


# Plot ERRORS for position and heading
plt.figure(figsize=(10, 6))
plt.plot(time, history[:, 0], label='Position x error')
plt.plot(time, history[:, 1], label='Position y error')
plt.plot(time, history[:, 2], label='Heading ψ error')
plt.title('Position and Heading Estimation Errors')
plt.xlabel('Time (s)')
plt.ylabel('Error Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot ERRORS for velocities
plt.figure(figsize=(10, 6))
plt.plot(time, history[:, 3], label='Velocity u error')
plt.plot(time, history[:, 4], label='Velocity v error')
plt.plot(time, history[:, 5], label='Velocity r error')
plt.title('Velocity Estimation Errors')
plt.xlabel('Time (s)')
plt.ylabel('Error Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot ERRORS for biases
plt.figure(figsize=(10, 6))
plt.plot(time, history[:, 6], label='Bias x error')
plt.plot(time, history[:, 7], label='Bias y error')
plt.plot(time, history[:, 8], label='Bias ψ error')
plt.title('Bias Estimation Errors')
plt.xlabel('Time (s)')
plt.ylabel('Error Value')
plt.legend()
plt.grid(True)
plt.show()

# 2D trajectory of POSITION ERROR
plt.figure(figsize=(8, 8))
plt.plot(history[:, 0], history[:, 1])
plt.scatter(history[0, 0], history[0, 1], color='green', s=100, label='Start')
plt.scatter(history[-1, 0], history[-1, 1], color='red', s=100, label='End')
plt.title('2D Position Error Trajectory')
plt.xlabel('Position x error')
plt.ylabel('Position y error')
plt.axis('equal')  # Equal scaling for x and y axes
plt.grid(True)
plt.legend()
plt.show()
