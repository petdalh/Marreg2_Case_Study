import numpy as np
from scipy.optimize import minimize

def thrust_allocation(tau_c):
    """
    tau_c: [Fx, Fy, Mz] commanded vector
    Returns:
       u_opt:  1D array of thruster magnitudes
       alpha_opt: 1D array of thruster angles
    """

    # Example system definitions (adapt to your needs)
    n_thrusters = 5
    max_thrust = np.array([125, 150, 125, 300, 300]) * 1000  # in N
    X = np.array([ 39.3,  35.6,  31.3, -28.5, -28.5])
    Y = np.array([  0.0,   0.0,   0.0,   5.0,  -5.0])
    fixed_angles = [np.pi/2, np.nan, np.pi/2, np.nan, np.nan]
    
    # Decision variables: [F1, F2, F3, F4, F5, alpha2, alpha4, alpha5]
    # (assuming thrusters 1 and 3 have fixed angles)
    # Provide an initial guess:
    x0 = np.zeros(8)
    
    # Define the B(alpha) * F mapping as a function of x
    # Here x[:5] are thrust magnitudes, x[5..7] are angles for thrusters 2, 4, 5
    def B_func(x):
        # Build angles
        alpha = np.array([
            fixed_angles[0],      # thruster 1
            x[5],                 # thruster 2
            fixed_angles[2],      # thruster 3
            x[6],                 # thruster 4
            x[7],                 # thruster 5
        ])
        # Build 3×5 matrix B
        # For each thruster i, the column is:
        # [ cos(alpha[i]); sin(alpha[i]); X[i]*sin(alpha[i]) - Y[i]*cos(alpha[i]) ]
        B = np.zeros((3, 5))
        for i in range(5):
            B[0,i] = np.cos(alpha[i])
            B[1,i] = np.sin(alpha[i])
            B[2,i] = X[i]*np.sin(alpha[i]) - Y[i]*np.cos(alpha[i])
        return B

    # Objective = || B(alpha)*F - tau_c ||^2 (or norm(...)^2)
    def objective(x):
        B = B_func(x)
        F = x[:5]  # thrust magnitudes
        residual = B @ F - tau_c
        return np.sum(residual**2)  # squared norm

    # Bounds on thrust: [-max_thrust, max_thrust] if bidirectional
    # or [0, max_thrust] if unidirectional
    thrust_bounds = [(-0.1*mt, mt) for mt in max_thrust]
    # Bounds on angles: [-pi, pi]
    angle_bounds = [(-np.pi, np.pi)]*3
    bounds = thrust_bounds + angle_bounds

    # Example constraints for angle slew limits, etc. (omitted here)
    # constraints = [{...}, {...}]

    # Solve via SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   # constraints=constraints,
                   options={'disp': True})
    x_opt = res.x

    # Extract solution
    u_opt = x_opt[:5]
    alpha_var = x_opt[5:]
    alpha_opt = np.array([
        fixed_angles[0],
        alpha_var[0],
        fixed_angles[2],
        alpha_var[1],
        alpha_var[2],
    ])

    return u_opt, alpha_opt

# Example usage:
if __name__ == "__main__":
    tau_cmd = np.array([1e5, 1e5, 5e4])  # [Fx, Fy, Mz] example
    u_opt, alpha_opt = thrust_allocation(tau_cmd)
    print("Optimal thrusts: ", u_opt)
    print("Optimal angles:  ", alpha_opt)
