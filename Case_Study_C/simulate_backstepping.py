import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ShipConfig:    
    def __init__(self):
        # System parameters
        self.M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
        self.D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
        self.T_b = np.diag([124, 124, 124])
        self.M_inv = np.linalg.inv(self.M)
        self.T_b_inv = np.linalg.inv(self.T_b)

ship_config = ShipConfig()

# Guidance parameters
p0 = np.array([0, 0])
p1 = np.array([100, 50])
U_ref = 0.0
psi_ref = 0.0
mu = 1.0
eps = 1e-5

# Controller gains
K1_gain = 200000.0
K2_gain = 200000.0

# Initial conditions
eta0 = np.array([0, 0, 0])
nu0 = np.array([0, 0, 0])
bias0 = np.array([0, 0, 0])
s0 = 0.0
initial_state = np.concatenate([eta0, nu0, bias0, [s0]])

# Simulation parameters
t_sim = 100.0
dt = 0.1
t = np.arange(0, t_sim, dt)

def straight_line(s, p0, p1, psi_ref):
    pd = (1 - s) * p0 + s * p1
    eta_d = np.array([pd[0], pd[1], psi_ref])
    pd_s = p1 - p0
    eta_ds = np.array([pd_s[0], pd_s[1], 0.0])
    eta_ds2 = np.zeros(3)
    return eta_d, eta_ds, eta_ds2

def update_law(eta_hat, s):
    p_hat = eta_hat[:2]
    eta_d, eta_ds, _ = straight_line(s, p0, p1, psi_ref)
    norm_eta_ds = np.linalg.norm(eta_ds[:2]) + eps
    V1_s = -(p1 - p0).T @ (p_hat - eta_d[:2])
    w = -mu/norm_eta_ds * V1_s
    v_s = U_ref / np.linalg.norm(p1 - p0)
    return w, v_s, 0.0

def backstepping_controller(eta_hat, nu_hat, bias_hat, eta_d, eta_ds, eta_ds2, w, v_s, v_ss):
    K1 = np.diag([K1_gain]*3)
    K2 = np.diag([K2_gain]*3)
    
    psi = eta_hat[2]
    R_T = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    # Ensure all inputs are properly shaped (3,1) arrays
    eta_hat = eta_hat.reshape(3,1)
    nu_hat = nu_hat.reshape(3,1)
    bias_hat = bias_hat.reshape(3,1)
    eta_d = eta_d.reshape(3,1)
    eta_ds = eta_ds.reshape(3,1)
    eta_ds2 = eta_ds2.reshape(3,1)
    
    z1 = R_T @ (eta_hat - eta_d)
    s_dot = w + v_s
    z1_dot = nu_hat - R_T @ eta_ds * s_dot
    alpha_1 = -K1 @ z1 + R_T @ eta_ds * v_s
    alpha_1_dot = -K1@(nu_hat - R_T@eta_ds*(v_s+w))+K1@R_T@eta_ds + R_T@eta_ds2*v_s+R_T@eta_ds*v_ss
    
    z2 = nu_hat - alpha_1
    
    tau = -K2@z2 + ship_config.D@nu_hat-bias_hat+ship_config.M@alpha_1_dot
    
    
    
    return tau.flatten()

def vessel_dynamics(state, t):
    eta = state[:3]
    nu = state[3:6]
    bias = state[6:9]
    s = state[9]
    
    # Guidance calculations
    w, v_s, v_ss = update_law(eta, s)
    eta_d, eta_ds, eta_ds2 = straight_line(s, p0, p1, psi_ref)
    
    # Control calculation
    tau = backstepping_controller(eta.copy(), nu.copy(), bias.copy(),
                                eta_d.copy(), eta_ds.copy(), eta_ds2.copy(),
                                w, v_s, v_ss)
    
    # Dynamics calculations
    psi = eta[2]
    R = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    eta_dot = R @ nu
    nu_dot = ship_config.M_inv @ (tau - ship_config.D @ nu + bias)
    bias_dot = -ship_config.T_b_inv @ bias
    s_dot = w + v_s
    
    return np.concatenate([eta_dot, nu_dot, bias_dot, [s_dot]])

# Run simulation
solution = odeint(vessel_dynamics, initial_state, t)

# Extract results
eta = solution[:, :3]
nu = solution[:, 3:6]
bias = solution[:, 6:9]
s_values = solution[:, 9]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(eta[:, 0], eta[:, 1], label='Actual Path')
plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--', label='Desired Path')
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('Vessel Path Tracking')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(t, s_values, label='Path Parameter s')
plt.xlabel('Time (s)')
plt.ylabel('s')
plt.title('Path Parameter Evolution')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(t, nu[:, 0], label='Surge Velocity (u)')
plt.plot(t, nu[:, 1], label='Sway Velocity (v)')
plt.plot(t, nu[:, 2], label='Yaw Rate (r)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Components')
plt.legend()
plt.grid(True)

plt.show()