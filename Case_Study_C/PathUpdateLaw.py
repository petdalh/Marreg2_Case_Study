import numpy as np

class PathUpdateLaws:
    def __init__(self, update_law_type='tracking', mu=0.0, lam=1.0, epsilon=1e-3):
        """
        Parameters:
        -----------
        update_law_type : str
            One of ['tracking', 'gradient', 'normalized_gradient', 'filtered_gradient']
        mu : float
            Gain parameter for gradient-based laws (μ ≥ 0)
        lam : float
            Filter gain for filtered gradient law (λ > 0)
        epsilon : float
            Small regularization constant (0 < ε ≪ 1)
        """
        self.update_law_type = update_law_type
        self.mu = mu
        self.lam = lam
        self.epsilon = epsilon
        self.omega_s = 0.0  # State for filtered gradient law
        
    def compute(self, V1_s, eta_d_s, v_s):
        """
        Compute ω_s based on selected update law
        
        Parameters:
        -----------
        V1_s : float
            Path error term V₁ˢ(η,s) = -(η-η_d(s))ᵀη_dˢ(s)
        eta_d_s : ndarray
            Path derivative η_dˢ(s)
        v_s : float
            Nominal speed profile
        
        Returns:
        --------
        s_dot : float
            Derivative of path parameter ṡ = v_s + ω_s
        """
        if self.update_law_type == 'tracking':
            omega_s = 0.0
            
        elif self.update_law_type == 'gradient':
            omega_s = -self.mu * V1_s
            
        elif self.update_law_type == 'normalized_gradient':
            eta_d_s_norm = np.linalg.norm(eta_d_s) + self.epsilon
            omega_s = -(self.mu / eta_d_s_norm) * V1_s
            
        elif self.update_law_type == 'filtered_gradient':
            # Dynamic equation for ω_s
            omega_s_dot = -self.lam * (self.omega_s + self.mu * V1_s)
            self.omega_s += omega_s_dot * self.dt  # Simple Euler integration
            omega_s = self.omega_s
            
        else:
            raise ValueError(f"Unknown update law type: {self.update_law_type}")
            
        return v_s + omega_s