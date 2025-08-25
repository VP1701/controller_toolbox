import numpy as np
import scipy.linalg
from scipy.optimize import minimize

def d2c(A_d, B_d, C_d, D_d, dt):
    """Function for transforming discrete time state space system with state space matrices A_d, B_d, C_d, D_d and a smapling time of dt
    to a continuous time state space system with matrices A_c, B_c, C_c, D_c.

    Args:
        -A_d ()
        -B_d
        -C_d
        -D_d

    """

    n = A_d.shape[0]

    # Check that matrix dimension are compatible
    assert A_d.shape == (n, n), "A_d must be square"
    assert B_d.shape[0] == n, "B_d must have same number of rows as A"
    assert C_d.shape[1] == n, "C_d must have same number of columns as A"
    assert D_d.shape[0] == C_d.shape[0], "D_d must have same number of rows as C"

    I = np.eye(n)
    M = 150

    # Eigenvalues of A_d
    eigvals_discrete = np.linalg.eigvals(A_d)

    # Initial guess: A_c ≈ (A_d - I) / dt first order taylor series
    A_c0 = (A_d - I) / dt
    A_c0_flat = A_c0.flatten()

    # 4th-order Runge-Kutta step for d(phi)/dt = A_c * phi
    def rk4_step(phi, A_c, h):
        k1 = h * A_c @ phi
        k2 = h * A_c @ (phi + 0.5 * k1)
        k3 = h * A_c @ (phi + 0.5 * k2)
        k4 = h * A_c @ (phi + k3)

        return phi + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Compute phi(dt) ≈ A using M Runge-Kutta steps
    def compute_phi_dt(A_c, dt, M):
        h = dt / M 
        phi = I.copy()
        for _ in range(M):
            phi = rk4_step(phi, A_c, h)
        return phi

    # Minimize error between A and phi(dt) to find A_c
    def objective(A_c_flat):
        A_c = A_c_flat.reshape((n, n))
        phi_dt = compute_phi_dt(A_c, dt, M)
        error = np.sum((A_d - phi_dt)**2)
        return error



    if np.any(np.abs(eigvals_discrete) < 1e-10) or np.any(np.real(eigvals_discrete) < - 1e-10):
        print(f"A matrix has eigenvalues which are zero or they have negative real parts")
        print(f"Numerically solving for A_c")
        # Optimize to get A_c
        result = minimize(objective, A_c0_flat, method='L-BFGS-B')
        A_c = result.x.reshape((n ,n))
        print(f"Result from optimization: {result}")
    else:
        print(f"A matrix eigenvalues are nonzero and have positive real parts")
        print(f"Solving A_c with matrix logarithm")
        A_c = scipy.linalg.logm(A_d) / dt


    # Compute gamma =
    h = dt / M
    phi = I.copy()
    gamma = 0
    for _ in range(M):
        phi_next = rk4_step(phi, A_c, h)
        gamma += ((phi + phi_next) / 2) * h # Integral aproximation using trapezoid rule
        phi = phi_next 

    # Solve gamma @ B_c = B for B_c
    B_c, _, _, _ = np.linalg.lstsq(gamma, B_d, rcond=None)
    
    # Output matrices are unchanged
    C_c = C_d 
    D_c = D_d 

    return A_c, B_c, C_c, D_c 