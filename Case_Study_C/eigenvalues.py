import numpy as np

D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])

M_inv = np.linalg.inv(M)

P_gain = 0.
I_gain = 0.05
D_gain = 2.0

Kp = np.diag([P_gain, P_gain, P_gain+10])
Ki = np.diag([I_gain, I_gain, I_gain])
Kd = np.diag([D_gain, D_gain, D_gain+2])

A = np.block([
    [np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
    [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)],
    [-M_inv@Ki, -M_inv@Kp, -M_inv@(D+Kd)]
])

eigenvalues = np.linalg.eigvals(A)

print(eigenvalues)

