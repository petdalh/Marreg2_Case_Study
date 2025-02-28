
#!/usr/bin/env python3

import numpy as np
from template_observer.wrap import wrap


def B(phi):
    return 

def luenberger(
        eta: np.ndarray,
        tau: np.ndarray,
        L1: np.ndarray,
        L2: np.ndarray,
        L3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    eta_hat = np.zeros((3, 1), dtype=float)
    nu_hat = np.zeros((3, 1), dtype=float)
    bias_hat = np.zeros((3, 1), dtype=float)

    M = np.array([
        [16, 0, 0],
        [0, 24, 0.53],
        [0, 0.53, 2.8]
    ])

    D = np.array([
        [0.66, 0, 0],
        [0, 1.3, 2.8], 
        [0, 0, 1.9]
    ])

    T_b = np.eye([143, 143, 143])

    A = np.block(
        [-L1, np.eye(3), np.zeros(3,3)],
        [-np.matmul(np.inv(M), L2), -np.matmul(np.inv(M), D), np.inv(M)],
        [-L3, np.zeros(3,3), -np.inv(T_b)]
    )

    np.vstack([
        L1, 
        np.dot(np.inv(M), L2),
        L3
    ])


    # Enter your code here

    return eta_hat, nu_hat, bias_hat