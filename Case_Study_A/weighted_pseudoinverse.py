import numpy as np

def weighted_pseudoinverse(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    W_sqrt = np.sqrt(W) 
    
    B_tilde = W_sqrt @ B
    
    B_tilde_pinv = np.linalg.pinv(B_tilde)
    
    W_sqrt_inv = np.linalg.inv(W_sqrt)
    B_w_pinv = W_sqrt_inv @ B_tilde_pinv
    
    return B_w_pinv