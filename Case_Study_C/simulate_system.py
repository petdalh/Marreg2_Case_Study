import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from StraightLineGuidance import StraightLineGuidance
from PathUpdateLaw import PathUpdateLaws

def dp_system_simulation():
    # 1. System Parameters
    M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
    D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
    M_inv = np.linalg.inv(M)
    
    # 2. PID Controller Parameters
    Kp = np.diag([8, 12, 5])   
    Ki = np.diag([0.5, 0.8, 0.3])  
    Kd = np.diag([18, 25, 10])   
    
    # 3. Path and Guidance Setup
    p0 = np.array([0, 0])       # Start point [x0, y0]
    p1 = np.array([100, 50])    # End point [x1, y1]
    U_ref = 1.0                 # Desired speed along path (m/s)
    psi_ref = np.pi/4           # Optional fixed heading (set to None for path-tangent)
    
    guidance = StraightLineGuidance(p0, p1, U_ref, psi_ref)
    
    # 4. Initial Conditions
    eta0 = np.array([0, 0, 0])  # Initial position [x, y, psi]
    nu0 = np.array([0, 0, 0])   # Initial velocity [u, v, r]
    Xi0 = np.array([0, 0, 0])   # Initial integral state
    
    # 5. Simulation Parameters
    t_sim = 100.0               # Simulation time in seconds
    dt = 0.1                    # Time step
    t = np.arange(0, t_sim, dt)

    
    
    # Add update law configuration
    update_law = PathUpdateLaws(
        update_law_type='normalized_gradient',  # Options: 'tracking', 'gradient', etc.
        mu=0.5,      # Start with 0 (tracking), then try 0.1, 0.5, 1.0
        lam=1.0,     # For filtered gradient
        epsilon=1e-3
    )
    
    # For testing with fixed position off the path
    test_fixed_position = False  # Set to True to test update laws with fixed position
    if test_fixed_position:
        fixed_p = np.array([30, 20])  # Position offset from path
    
    # Modified system dynamics
    def system_dynamics(y, t, M, M_inv, D, Kp, Ki, Kd, guidance, update_law):
        Xi = y[0:3]     # Integral state
        eta = y[3:6]    # Position [x, y, psi]
        nu = y[6:9]     # Velocity [u, v, r]
        s = y[9]        # Path parameter (new state)
        
        # For testing: override position with fixed offset
        if test_fixed_position:
            eta[:2] = fixed_p
            
        # Get guidance commands at current s
        pd, psi_d, U_s = guidance.update(s)
        
        # Current yaw angle
        psi = eta[2]
        
        # Rotation matrix
        R = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # Compute path error terms (Task 4.5)
        p_error = eta[:2] - pd
        eta_d_s = np.concatenate([guidance.pd_s, [0]])  # Path derivative
        
        # V₁ˢ(η,s) = -(p - p_d(s))ᵀ(p₁ - p₀) (for straight line)
        V1_s = -np.dot(guidance.pd_s, p_error)
        
        # Compute s_dot using selected update law
        s_dot = update_law.compute(V1_s, eta_d_s, U_s)
        
        # Pose error in body frame
        position_error = np.array([eta[0]-pd[0], eta[1]-pd[1], 0])
        e = R.T @ position_error
        
        # Heading error with angle wrapping
        e_psi = (psi - psi_d + np.pi) % (2*np.pi) - np.pi
        
        # Combined error vector (position and heading)
        e_full = np.array([e[0], e[1], e_psi])
        
        # PID control law
        tau_pid = -Ki @ Xi - Kp @ e_full - Kd @ nu
        
        # System dynamics
        eta_dot = R @ nu
        nu_dot = M_inv @ (-D @ nu + tau_pid)
        
        # Integral state dynamics
        Xi_dot = R.T @ position_error
        
        # Combine derivatives including s_dot
        y_dot = np.concatenate([Xi_dot, eta_dot, nu_dot, [s_dot]])
        
        return y_dot
    
    # Add s to initial state (last element)
    y0 = np.concatenate([Xi0, eta0, nu0, [0]])
    
    # Run simulation
    sol = odeint(system_dynamics, y0, t, 
                args=(M, M_inv, D, Kp, Ki, Kd, guidance, update_law))
    
    # Extract results including s
    Xi_history = sol[:, 0:3]
    eta_history = sol[:, 3:6]
    nu_history = sol[:, 6:9]
    s_history = sol[:, 9]
    
    # Generate path reference points actually followed
    pd_history = np.array([guidance.update(s)[0] for s in s_history])
    
    # Plotting (modified to show actual path followed)
    plt.figure(figsize=(12, 8))
    
    # Position plot
    plt.subplot(2, 1, 1)
    plt.plot(eta_history[:, 0], eta_history[:, 1], label='Vessel trajectory')
    plt.plot(pd_history[:, 0], pd_history[:, 1], 'r--', label='Reference point')
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k:', label='Desired path')
    
    if test_fixed_position:
        plt.scatter(fixed_p[0], fixed_p[1], color='k', marker='x', s=100, 
                   label='Fixed vessel position')
        # Plot orthogonal projection
        t = np.dot(fixed_p - p0, p1 - p0) / np.dot(p1 - p0, p1 - p0)
        t = np.clip(t, 0, 1)
        proj = p0 + t * (p1 - p0)
        plt.plot([fixed_p[0], proj[0]], [fixed_p[1], proj[1]], 'k-', 
                label='Orthogonal projection')
    
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title(f'Path Following with {update_law.update_law_type} law (μ={update_law.mu})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot path parameter evolution
    plt.subplot(2, 1, 2)
    plt.plot(t, s_history, label='Path parameter s(t)')
    plt.xlabel('Time [s]')
    plt.ylabel('s')
    plt.title('Evolution of Path Parameter')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_orientation(eta, length=1, color='k', label=None):
    """Plot vessel orientation as an arrow"""
    x, y, psi = eta
    dx = length * np.sin(psi)
    dy = length * np.cos(psi)
    plt.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc=color, ec=color, label=label)

if __name__ == "__main__":
    dp_system_simulation()