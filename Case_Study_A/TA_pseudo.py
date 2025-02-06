#!/usr/bin/env python3

import numpy as np

def thruster_allocation(tau: np.ndarray) -> np.ndarray:
    """
    Allocates thruster inputs based on the commanded forces and moment in the vessel's body-fixed frame.

    Parameters
    ----------
    tau : np.ndarray
        A 3×1 vector [X, Y, N]^T representing surge (X), sway (Y), and yaw (N) commands.

    Returns
    -------
    np.ndarray
        A 5×1 vector containing the required inputs for each thruster, and thruster angles.
    """

    # Vessel parameters
    alpha = np.array((np.pi/2, np.pi/2), dtype=float).T
    x = np.array((0, 0, 0), dtype=float).T
    y = np.array((0, 0, 0), dtype=float).T
    K = np.array((1, 1, 1), dtype=float).T

    # Compute the matrix B_alpha which maps the forces and moments to the thruster inputs
    B_alpha = np.array([
        [np.cos(alpha[0]), np.cos(alpha[1]), 0],
        [np.sin(alpha[0]), np.sin(alpha[1]), 1],
        [x[0]*np.sin(alpha[0])-y[0]*np.cos(alpha[0]), x[1]*np.sin(alpha[1])-y[1]*np.cos(alpha[1]), x[2]]
    ])

    # Compute the weighted pseudoinverse of B_alpha
    B_w_pinv = np.linalg.pinv(B_alpha)
    # If we used W is not the identity matrix
    # W = np.eye(1/K**2)
    # BW = np.linalg.pinv(B_alpha.T @ W @ B_alpha) @ (B_alpha.T @ W)

    # Compute simplified F^* with F_d = 0
    F_star = B_w_pinv @ tau

    # Compute the thruster inputs u_thurster = k * F^*
    u_thrusters = K * F_star

    # Handeling negative values
    for i in (0, 1):  # Assuming thruster 2 is a fixed-direction thruster
        if u_thrusters[i] < 0:
            u_thrusters[i] = -u_thrusters[i]
            alpha[i] += np.pi  # reverse direction
    # If we cannot have neegative thrusters for the fixed-direction thruster
    # if u_thrusters[2] < 0: u_thrusters[2] = 0

    u = np.append(u_thrusters, alpha)
    return u

def main():
    # Example usage
    tau = np.array((1, -1, -0.5), dtype=float).T
    u = thruster_allocation(tau)
    print(u)

if __name__ == "__main__":
    main()
