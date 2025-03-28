import numpy as np

class StraightLineGuidance:
    def __init__(self, p0, p1, U_ref, psi_ref=None):
        """
        Straight-line path guidance system
        
        Parameters:
        -----------
        p0 : array_like
            Start point [x0, y0] in NE coordinates
        p1 : array_like
            End point [x1, y1] in NE coordinates
        U_ref : float
            Desired speed along path
        psi_ref : float, optional
            Fixed heading reference (if None, uses path tangent)
        """
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.U_ref = U_ref
        self.psi_ref = psi_ref
        
        # Pre-compute path parameters
        self.pd_s = self.p1 - self.p0  # Path derivative (constant for straight line)
        self.path_length = np.linalg.norm(self.pd_s)
        
        # For path-tangent heading
        if self.psi_ref is None:
            self.psi_d = np.arctan2(self.pd_s[1], self.pd_s[0])
    
    def update(self, s):
        """
        Compute guidance outputs at path parameter s ∈ [0,1]
        
        Returns:
        --------
        pd : ndarray
            Desired position [xd, yd] at s
        psi_d : float
            Desired heading at s
        U_s : float
            Desired speed at s
        """
        # Task 2.1: Straight-line path parameterization
        pd = (1 - s) * self.p0 + s * self.p1
        
        # Task 2.3: Heading calculation
        if self.psi_ref is not None:
            psi_d = self.psi_ref  # Use fixed heading if specified
        else:
            psi_d = self.psi_d  # Use path-tangent heading
            
        # Task 2.4: Speed profile
        U_s = self.U_ref / self.path_length if self.path_length > 0 else 0
        
        return pd, psi_d, U_s
    
    def get_path_derivatives(self):
        """
        Return path derivatives (Task 2.1)
        
        Returns:
        --------
        pd_s : ndarray
            First derivative ∂p/∂s
        pd_s2 : ndarray
            Second derivative ∂²p/∂s²
        pd_s3 : ndarray
            Third derivative ∂³p/∂s³
        """
        # For straight line:
        pd_s = self.pd_s
        pd_s2 = np.zeros_like(pd_s)
        pd_s3 = np.zeros_like(pd_s)
        
        return pd_s, pd_s2, pd_s3
    
    def get_heading_derivatives(self):
        """
        Return heading derivatives (Task 2.3)
        
        Returns:
        --------
        psi_d_s : float
            First derivative ∂ψ/∂s
        psi_d_s2 : float
            Second derivative ∂²ψ/∂s²
        """
        # For straight line (constant heading):
        psi_d_s = 0.0
        psi_d_s2 = 0.0
        
        return psi_d_s, psi_d_s2