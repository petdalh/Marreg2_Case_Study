import numpy as np

def weighted_pseudoinverse(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Returns the weighted Moore-Penrose pseudoinverse of B, given a diagonal weighting W.

    Parameters
    ----------
    B : np.ndarray
        The matrix to invert (3×n, for instance).
    W : np.ndarray
        A diagonal weighting matrix (n×n).

    Returns
    -------
    np.ndarray
        The weighted pseudoinverse of B (n×3).
    """
    # 1. Compute W^(1/2). Here we assume W is diagonal, so just take sqrt of each diagonal element.
    W_sqrt = np.sqrt(W)   # or np.diag(np.sqrt(np.diag(W))) if needed
    
    # 2. Form the weighted matrix B_tilde
    B_tilde = W_sqrt @ B
    
    # 3. Compute the standard (unweighted) pseudoinverse of B_tilde
    B_tilde_pinv = np.linalg.pinv(B_tilde)
    
    # 4. Multiply by W^(-1/2) on the left to get the weighted pseudoinverse
    W_sqrt_inv = np.linalg.inv(W_sqrt)
    B_w_pinv = W_sqrt_inv @ B_tilde_pinv
    
    return B_w_pinv