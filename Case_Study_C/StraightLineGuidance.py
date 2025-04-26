import numpy as np

class StraightLineGuidance:
    def __init__(self, p0, p1, U_ref, psi_ref=None):
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
        # For straight line:
        pd_s = self.pd_s
        pd_s2 = np.zeros_like(pd_s)
        pd_s3 = np.zeros_like(pd_s)
        
        return pd_s, pd_s2, pd_s3
    
    def get_heading_derivatives(self):
        # For straight line (constant heading):
        psi_d_s = 0.0
        psi_d_s2 = 0.0
        
        return psi_d_s, psi_d_s2