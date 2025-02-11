#!/usr/bin/env python3

import numpy as np

def thruster_allocation(tau: np.ndarray) -> np.ndarray:
    """
    Weighted pseudoinverse thrust allocation with sign-flip logic for 2 VSP thrusters
    plus a single fixed-direction lateral thruster.

    Parameters
    ----------
    tau : np.ndarray
        3x1 vector [X, Y, N]^T representing surge, sway, and yaw commands.

    Returns
    -------
    np.ndarray
        5x1 vector [u1, u2, u3, alpha1, alpha2], where
        - u1, u2, u3 are thruster inputs (e.g. in [0,1] or [−1,1]),
        - alpha1, alpha2 are azimuth angles for thrusters 1 & 2.
    """

    # Thruster positions (x_i, y_i).
    x1, y1 = (1.0,  0.0)
    x2, y2 = (-1.0, 0.0)
    x3, y3 = (0.0,  0.0)   

    # Thruster gains (F_i = k_i * u_i)
    K = np.array([1.0, 1.0, 1.0], dtype=float)

    # Weight matrix W (here just the identity)
    W = np.eye(5)

    # Feed-forward f_d (size 5).  Often zero by default
    f_d = np.zeros(5)
    
    B = np.array([
        [1, 0, 1, 0, 0],  # surge
        [0, 1, 0, 1, 1],  # sway
        [-y1, x1, -y2, x2, x3]  # yaw
    ], dtype=float)

    
    # Compute the weighted pseudoinverse B_W^†
    W = np.eye(3)  # For now, just identity
    W_sqrt = np.sqrt(W)  # Or np.diag(np.sqrt(np.diag(W))) if needed
    B_w = W_sqrt @ B
    B_w_pinv = np.linalg.pinv(B)  


    # f* = B_W^† tau_cmd + (I - B_W^† B) f_d
    Q_w = np.eye(5) - B_w_pinv @ B
    f_star = B_w_pinv @ tau + Q_w @ f_d

    # Parse out X1*, Y1*, X2*, Y2*, Y3*
    X1_star, Y1_star, X2_star, Y2_star, Y3_star = f_star

    
    # Convert to thruster magnitudes/angles for thrusters 1 & 2
    F1_star = np.hypot(X1_star, Y1_star)
    F2_star = np.hypot(X2_star, Y2_star)

    alpha1_star = np.arctan2(Y1_star, X1_star)
    alpha2_star = np.arctan2(Y2_star, X2_star)

    # Third thruster
    F3_star = abs(Y3_star)

    # Convert to thruster inputs u_i = F_i / k_i
    u1_star = F1_star / K[0]
    u2_star = F2_star / K[1]
    u3_star = F3_star / K[2]

    # Clamp each thruster input to [0, 1]:
    u1_cmd = max(0.0, min(u1_star, 1.0))
    u2_cmd = max(0.0, min(u2_star, 1.0))
    u3_cmd = max(0.0, min(u3_star, 1.0))

    # Wrap angles to [-pi, pi], if needed
    def wrap_angle(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    alpha1_cmd = wrap_angle(alpha1_star)
    alpha2_cmd = wrap_angle(alpha2_star)

    # 5x1 vector: [u1, u2, u3, alpha1, alpha2]
    return np.array([u1_cmd, u2_cmd, u3_cmd, alpha1_cmd, alpha2_cmd])

def main():
    # Example usage
    tau_cmd = np.array([1, -1, -0.5])  # 3x1
    result = thruster_allocation(tau_cmd)
    print("Thruster allocation:", result)

if __name__ == "__main__":
    main()
