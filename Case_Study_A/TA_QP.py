#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

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
    x1, y1 = (-1.0, -1.0)  
    x2, y2 = (-1.0, 1.0)   
    x3, y3 = (1.0, 0.0)

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


def plot_vessel_with_thrusters(tau, thruster_allocation_result):
    """
    Plot the vessel with thrusters represented as arrows.
    
    Parameters
    ----------
    tau : np.ndarray
        3x1 vector [X, Y, N]^T representing surge, sway, and yaw commands.
    thruster_allocation_result : np.ndarray
        5x1 vector [u1, u2, u3, alpha1, alpha2]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract thruster values
    u1, u2, u3, alpha1, alpha2 = thruster_allocation_result
    
    # Vessel dimensions (arbitrary scale for visualization)
    vessel_length = 3.0
    vessel_width = 1.0
    
    # Thruster positions (match with allocation function)
    # Adjust positions for better visualization
    x1, y1 = (-0.8 * vessel_length / 2, -0.2 * vessel_width)  # Rear left
    x2, y2 = (-0.8 * vessel_length / 2, 0.2 * vessel_width)   # Rear right
    x3, y3 = (0.8 * vessel_length / 2, 0)                    # Front center
    
    # Draw vessel outline as a simple polygon
    vessel_outline = [
        (-vessel_length/2, -vessel_width/2),  # Rear left
        (vessel_length/2, -vessel_width/2),   # Front left
        (vessel_length/2 + 0.3, 0),           # Front point
        (vessel_length/2, vessel_width/2),    # Front right
        (-vessel_length/2, vessel_width/2),   # Rear right
    ]
    vessel = Polygon(vessel_outline, closed=True, facecolor='lightgray', edgecolor='black')
    ax.add_patch(vessel)
    
    # Arrow scale factor (adjust as needed for visibility)
    arrow_scale = 0.5
    
    # Draw azimuth thrusters (thrusters 1 & 2) as arrows
    arrow_length1 = u1 * arrow_scale
    dx1 = arrow_length1 * np.cos(alpha1)
    dy1 = arrow_length1 * np.sin(alpha1)
    ax.arrow(x1, y1, dx1, dy1, head_width=0.05, head_length=0.1, fc='blue', ec='blue', width=0.02)
    ax.add_patch(Circle((x1, y1), 0.05, color='blue'))
    
    arrow_length2 = u2 * arrow_scale
    dx2 = arrow_length2 * np.cos(alpha2)
    dy2 = arrow_length2 * np.sin(alpha2)
    ax.arrow(x2, y2, dx2, dy2, head_width=0.05, head_length=0.1, fc='blue', ec='blue', width=0.02)
    ax.add_patch(Circle((x2, y2), 0.05, color='blue'))
    
    # Draw the fixed lateral thruster (thruster 3) as a horizontal arrow
    arrow_length3 = u3 * arrow_scale
    # Fixed lateral thruster only acts in the Y direction
    ax.arrow(x3, y3, 0, arrow_length3, head_width=0.05, head_length=0.1, fc='red', ec='red', width=0.02)
    ax.add_patch(Circle((x3, y3), 0.05, color='red'))
    
    # Add command vector as a green arrow from center of vessel
    command_scale = 0.3
    center = (0, 0)
    ax.arrow(center[0], center[1], tau[0] * command_scale, tau[1] * command_scale, 
             head_width=0.05, head_length=0.1, fc='green', ec='green', width=0.02)
    
    # Add thruster information as text
    plt.text(-1.5, -1.0, f'Azimuth Thruster 1: Force={u1:.2f}, Angle={np.degrees(alpha1):.1f}°', fontsize=9)
    plt.text(-1.5, -1.2, f'Azimuth Thruster 2: Force={u2:.2f}, Angle={np.degrees(alpha2):.1f}°', fontsize=9)
    plt.text(-1.5, -1.4, f'Lateral Thruster 3: Force={u3:.2f}', fontsize=9)
    plt.text(-1.5, -1.6, f'Command: Surge={tau[0]:.2f}, Sway={tau[1]:.2f}, Yaw={tau[2]:.2f}', fontsize=9)
    
    # Set axis properties
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Vessel Thruster Allocation Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig

def main():
    # Example usage
    tau_cmd = np.array([1, 1, 0])  # 3x1
    result = thruster_allocation(tau_cmd)
    print("Thruster allocation:", result)
    
    # # Create visualization
    fig = plot_vessel_with_thrusters(tau_cmd, result)
    
    # Add interactive input for demonstration
    plt.figure(figsize=(10, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, 
             "To try different commands, change tau_cmd values in the main() function.",
             ha='center', va='center', fontsize=12)
    
    plt.show()

if __name__ == "__main__":
    main()