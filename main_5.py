import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Model Definition (based on the thesis)
# ==============================================================================

def F(x, params):
    """Logistic growth function: F(x) = rx(1 - x/K)"""
    return params['r'] * x * (1 - x / params['K'])

def r_func(t, params):
    """Dissipation function: r(t) = exp(-ρt)"""
    return np.exp(-params['rho'] * t)
    
def r_prime(t, params):
    """Derivative of dissipation function: r'(t) = -ρ*exp(-ρt)"""
    return -params['rho'] * np.exp(-params['rho'] * t)

def switching_function(p, x, t, params):
    """
    Switching function σ(t) from page 172.
    σ(t) = p(t)x(t) + (c - πx(t))e^(-δt)
    The control u(t) depends on the sign of σ(t).
    The thesis aims to *minimize* J, where J is negative profit.
    So we want to make J as small (large negative) as possible.
    J = ∫ (c - πx)u dt.
    To minimize this, if (c - πx) is negative, u should be E. If positive, u=0.
    The Hamiltonian term for u is [p(t)x(t) + (c - πx(t))e^(-δt)]*(-u).
    So, σ(t) = p(t)x(t) + (c - πx(t))e^(-δt).
    To minimize H, if σ(t) is positive, u=E. If σ(t) is negative, u=0.
    Let's stick to the conclusion on page 164:
    u(t) = E if σ(t) < 0
    u(t) = 0 if σ(t) > 0
    where σ is the function d'échange.
    σ(t) = p(t)x(t) + (c-πx(t))e^(-δt)
    """
    return p * x + (params['c'] - params['pi'] * x) * np.exp(-params['delta'] * t)

# ==============================================================================
# 2. Numerical Solver
# ==============================================================================

def solve_biomass_model(params, N=1000, max_iter=100, tol=1e-5, alpha=0.1):
    """
    Solves the integro-differential optimal control problem for biomass harvesting.

    Args:
        params (dict): Dictionary of model parameters.
        N (int): Number of time steps for discretization.
        max_iter (int): Maximum number of iterations for the solver.
        tol (float): Convergence tolerance.
        alpha (float): Step size for control update (for stability).

    Returns:
        dict: A dictionary containing the results (t, x, p, u, sigma).
    """
    T = params['T']
    dt = T / N
    t = np.linspace(0, T, N + 1)

    # --- Initialize arrays for state, adjoint, and control ---
    x = np.zeros(N + 1)
    p = np.zeros(N + 1)
    u = np.full(N + 1, params['E'] / 2)  # Initial guess for control

    x[0] = params['x0']

    print("Starting forward-backward sweep iterations...")
    for iteration in range(max_iter):
        u_old = u.copy()

        # --- (a) Forward Sweep: Solve for x(t) given u(t) ---
        # Equation: x'(t) = F(x(t)) - u(t)x(t) + ∫[0,t] r'(t-s)F(x(s))ds
        for i in range(N):
            # Numerical integration for the memory term (convolution)
            # using the trapezoidal rule
            integral_s = t[:i+1]
            integrand_x = r_prime(t[i] - integral_s, params) * F(x[:i+1], params)
            integral_val_x = np.trapz(integrand_x, integral_s) if i > 0 else 0

            # Euler forward step
            x_dot = F(x[i], params) - u[i] * x[i] + integral_val_x
            x[i+1] = x[i] + x_dot * dt
            # Ensure biomass is non-negative
            if x[i+1] < 0:
                x[i+1] = 0

        # --- (b) Backward Sweep: Solve for p(t) given x(t) and u(t) ---
        # Terminal condition p(T) = 0 is already set in initialization
        # Equation (3.1): -p'(t) = p(t){F'(x(t)) - u(t) + ∫[0,t]r'(t-s)F(x(s))ds} + πu(t)e^(-δt)
        # We rewrite as p'(t) = -p(t){...} - πu(t)e^(-δt)
        p[N] = 0 # Transversality condition (3.2)
        for i in range(N, 0, -1):
            # The integral term is the same as in the forward sweep
            integral_s = t[:i+1]
            integrand_p = r_prime(t[i] - integral_s, params) * F(x[:i+1], params)
            integral_val_p = np.trapz(integrand_p, integral_s) if i > 0 else 0

            # Euler backward step
            F_prime_x = params['r'] * (1 - 2 * x[i] / params['K'])
            p_dot = -p[i] * (F_prime_x - u[i] + integral_val_p) \
                    - params['pi'] * u[i] * np.exp(-params['delta'] * t[i])
            
            p[i-1] = p[i] - p_dot * dt

        # --- (c) Update Control u(t) using the switching function ---
        sigma = switching_function(p, x, t, params)
        
        # Determine the new control based on the sign of sigma
        u_new = np.zeros(N + 1)
        u_new[sigma < 0] = params['E'] # Harvest at max effort
        u_new[sigma > 0] = 0           # No harvesting

        # For stability, update the control using a convex combination
        u = alpha * u_new + (1 - alpha) * u_old

        # --- (d) Check for convergence ---
        change = np.sqrt(np.sum((u - u_old)**2)) / (N+1)
        print(f"Iteration {iteration + 1}/{max_iter}, Change in u: {change:.6f}")
        if change < tol:
            print(f"\nConvergence reached after {iteration + 1} iterations.")
            break
    
    if iteration == max_iter - 1:
        print("\nWarning: Maximum number of iterations reached without convergence.")

    # Final calculation of the switching function with the converged control
    sigma = switching_function(p, x, t, params)
    
    return {'t': t, 'x': x, 'p': p, 'u': u, 'sigma': sigma}


# ==============================================================================
# 3. Running the Simulation and Plotting Results
# ==============================================================================
if __name__ == '__main__':
    # --- Define Model Parameters ---
    # These parameters are chosen to create an interesting scenario.
    # The relationship between r, delta, and the costs/prices is crucial.
    model_params = {
        'x0': 500,      # Initial biomass
        'r': 5,      # Intrinsic growth rate
        'K': 5000,      # Carrying capacity
        'rho': 5,    # Dissipation rate (ρ from r(t)=exp(-ρt))
        'pi': 10,     # Price per unit of biomass (π)
        'c': 2,     # Cost per unit of effort
        'delta': 0.05,  # Discount rate (δ)
        'E': 5,      # Maximum harvesting effort
        'T': 20,       # Time horizon
    }

    # --- Solve the model ---
    results = solve_biomass_model(
        model_params, 
        N=500, 
        max_iter=150, 
        tol=1e-6,
        alpha=0.05 # Smaller alpha for better stability
    )

    # --- Plotting ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Optimal Biomass Harvesting with Dissipation (Memory Effect)', fontsize=16)

    # Plot 1: Biomass (State x(t))
    axs[0].plot(results['t'], results['x'], lw=2, label='Biomass x(t)')
    axs[0].axhline(model_params['K'], color='gray', linestyle='--', label=f'Carrying Capacity K={model_params["K"]}')
    axs[0].set_ylabel('Biomass')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Harvesting Effort (Control u(t))
    axs[1].plot(results['t'], results['u'], lw=2, color='red', label='Harvesting Effort u(t)')
    axs[1].set_ylim(-0.05, model_params['E'] + 0.05)
    axs[1].set_ylabel('Effort')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Adjoint Variable (p(t))
    axs[2].plot(results['t'], results['p'], lw=2, color='green', label='Adjoint Variable p(t)')
    axs[2].set_ylabel('Adjoint (Shadow Price)')
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: Switching Function (σ(t))
    axs[3].plot(results['t'], results['sigma'], lw=2, color='purple', label='Switching Function σ(t)')
    axs[3].axhline(0, color='black', linestyle='--', lw=1)
    axs[3].set_ylabel('Switching Function')
    axs[3].set_xlabel('Time (t)')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()