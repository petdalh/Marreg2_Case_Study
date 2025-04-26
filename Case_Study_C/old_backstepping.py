 def backstepping_controller(observer, reference, K1_gain, K2_gain, config=ship_config) -> np.ndarray:



# Getting the states from the observer

eta_hat = np.array(observer.eta).reshape(3, 1)

nu_hat = np.array(observer.nu).reshape(3, 1)

bias_hat = np.array(observer.bias).reshape(3, 1)



# Getting the states from the reference

eta_d = np.array(reference.eta_d).reshape(3, 1)

eta_ds = np.array(reference.eta_ds).reshape(3, 1)

eta_ds2 = np.array(reference.eta_ds2).reshape(3, 1)



w = reference.w

v_s = reference.v_s

v_ss = reference.v_ss



K1 = np.diag([K1_gain, K1_gain, K1_gain])*0.1

K2 = np.diag([K2_gain, K2_gain, K2_gain])*1



psi = eta_hat[2, 0]

R_T = np.array([

[np.cos(psi), np.sin(psi), 0],

[-np.sin(psi), np.cos(psi), 0],

[0, 0, 1]

])



z1 = R_T @ (eta_hat - eta_d)

s_dot = w + v_s

z1_dot = nu_hat - R_T @ eta_ds * s_dot

alpha_1 = -K1 @ z1 + R_T @ eta_ds * v_s

alpha_1_dot = -K1@(nu_hat - R_T@eta_ds*(v_s+w))+K1@R_T@eta_ds + R_T@eta_ds2*v_s+R_T@eta_ds*v_ss


z2 = nu_hat - alpha_1


tau = -K2@z2 + ship_config.D@nu_hat-bias_hat+ship_config.M@alpha_1_dot



return tau