import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def dp_system_simulation():
    # 1. System Parameters (from your case study)
    M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
    D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
    M_inv = np.linalg.inv(M)
    
    # 2. PID Controller Parameters (replace with your tuned values)
    Kp = np.diag([8, 12, 5])   
    Ki = np.diag([0.5, 0.8, 0.3])  
    Kd = np.diag([18, 25, 10])   
    
    # 3. Reference and Initial Conditions
    eta_ref = np.array([10, 5, np.pi/4])  # Reference position [x, y, psi] in NE frame
    eta0 = np.array([0, 0, 0])            # Initial position
    nu0 = np.array([0, 0, 0])             # Initial velocity
    Xi0 = np.array([0, 0, 0])             # Initial integral state
    
    # 4. Simulation Parameters
    t_sim = 100.0  # Simulation time in seconds
    dt = 0.1       # Time step
    t = np.arange(0, t_sim, dt)
    
    # 5. Define the closed-loop system dynamics
    def system_dynamics(y, t, M, M_inv, D, Kp, Ki, Kd, eta_ref):
        # Unpack states
        Xi = y[0:3]    # Integral state
        eta = y[3:6]    # Position in NE frame [x, y, psi]
        nu = y[6:9]     # Velocity in body frame [u, v, r]
        
        # Get current yaw angle
        psi = eta[2]
        
        # Rotation matrix
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # Skew-symmetric matrix S
        S = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        
        # Pose error in body frame (Task 1.1)
        e = R.T @ (eta - eta_ref)
        
        # PID control law (Task 1.2)
        tau_pid = -Ki @ Xi - Kp @ e - Kd @ nu
        
        # System dynamics (from theory)
        eta_dot = R @ nu
        nu_dot = M_inv @ (-D @ nu + tau_pid)  # Assuming perfect bias estimation
        
        # Integral state dynamics
        Xi_dot = R.T @ (eta - eta_ref)
        
        # Combine derivatives
        y_dot = np.concatenate([Xi_dot, eta_dot, nu_dot])
        
        return y_dot
    
    # 6. Run simulation
    y0 = np.concatenate([Xi0, eta0, nu0])
    sol = odeint(system_dynamics, y0, t, args=(M, M_inv, D, Kp, Ki, Kd, eta_ref))
    
    # 7. Extract results
    Xi_history = sol[:, 0:3]
    eta_history = sol[:, 3:6]
    nu_history = sol[:, 6:9]
    
    # 8. Plot results
    plt.figure(figsize=(12, 8))
    
    # Position and heading
    plt.subplot(3, 1, 1)
    plt.plot(t, eta_history[:, 0], label='x [m]')
    plt.plot(t, eta_history[:, 1], label='y [m]')
    plt.plot(t, eta_history[:, 2], label='psi [rad]')
    plt.axhline(y=eta_ref[0], color='r', linestyle='--', label='x_ref')
    plt.axhline(y=eta_ref[1], color='g', linestyle='--', label='y_ref')
    plt.axhline(y=eta_ref[2], color='b', linestyle='--', label='psi_ref')
    plt.ylabel('Position/Heading')
    plt.legend()
    plt.grid(True)
    
    # Velocity
    plt.subplot(3, 1, 2)
    plt.plot(t, nu_history[:, 0], label='u [m/s]')
    plt.plot(t, nu_history[:, 1], label='v [m/s]')
    plt.plot(t, nu_history[:, 2], label='r [rad/s]')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    
    # Control forces
    # Calculate control forces for plotting
    tau_history = []
    for i in range(len(t)):
        psi = eta_history[i, 2]
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        e = R.T @ (eta_history[i] - eta_ref)
        tau = -Ki @ Xi_history[i] - Kp @ e - Kd @ nu_history[i]
        tau_history.append(tau)
    
    tau_history = np.array(tau_history)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, tau_history[:, 0], label='tau_x [N]')
    plt.plot(t, tau_history[:, 1], label='tau_y [N]')
    plt.plot(t, tau_history[:, 2], label='tau_psi [Nm]')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Forces')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 9. Plot trajectory in NE plane
    plt.figure(figsize=(8, 6))
    plt.plot(eta_history[:, 0], eta_history[:, 1], label='Vessel trajectory')
    plt.scatter(eta_ref[0], eta_ref[1], color='r', label='Reference position')
    
    # Plot initial and final orientation
    plot_orientation(eta_history[0], length=2, color='g', label='Start')
    plot_orientation(eta_history[-1], length=2, color='b', label='End')
    
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title('Vessel Trajectory in NE Frame')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_orientation(eta, length=1, color='k', label=None):
    """Plot vessel orientation as an arrow"""
    x, y, psi = eta
    dx = length * np.sin(psi)
    dy = length * np.cos(psi)
    plt.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc=color, ec=color, label=label)

if __name__ == "__main__":
    dp_system_simulation()