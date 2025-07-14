import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Paramètres du modèle (exemples)
r = 0.75      # Taux de croissance intrinsèque
K = 1000      # Capacité de charge
p = 10        # Prix par unité de biomasse
c = 2         # Coût unitaire de l'effort
δ = 0.05      # Taux d'actualisation
T = 30        # Horizon temporel
x0 = 500      # Biomasse initiale
γ = 0.01      # Coefficient de dissipation
u_max = 2   # Effort maximal

# Discrétisation temporelle
N = 120
t_eval = np.linspace(0, T, N)
dt = t_eval[1] - t_eval[0]

def volterra_eq(t, u_func, x_func):
    """Équation intégrale de Volterra avec dissipation cumulative"""
    def integrand(s):
        dissipation = γ * quad(lambda τ: x_func(τ), 0, s)[0]
        return r * x_func(s) * (1 - x_func(s)/K) - u_func(s) - dissipation
    return x0 + quad(integrand, 0, t)[0]

def profit(u_func, x_func):
    """Profit actualisé avec terme quadratique"""
    def integrand(t):
        return np.exp(-δ * t) * ( c  - p * x_func(t)) * u_func(t)
    # Intégration du profit actualisé
    return quad(integrand, 0, T)[0]

def objective(u_params):
    """Fonction objectif pour l'optimisation"""
    u_func = lambda t: np.interp(t, t_eval, u_params)
    
    # Résolution de l'équation intégrale
    x_vals = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        if i == 0:
            x_vals[i] = x0
        else:
            x_interp = lambda s: np.interp(s, t_eval[:i], x_vals[:i])
            x_vals[i] = volterra_eq(t, u_func, x_interp)
    
    return -profit(u_func, lambda t: np.interp(t, t_eval, x_vals))

# Optimisation du contrôle
initial_guess = np.full(N, u_max/2)
bounds = [(0, u_max) for _ in range(N)]

result = minimize(
    objective,
    initial_guess,
    bounds=bounds,
    method='L-BFGS-B',
    options={'maxiter': 100, 'ftol': 1e-6}
)

# Extraction des résultats
u_opt = np.clip(result.x, 0, u_max)
x_opt = np.zeros(N)
for i, t in enumerate(t_eval):
    if i == 0:
        x_opt[i] = x0
    else:
        x_interp = lambda s: np.interp(s, t_eval[:i], x_opt[:i])
        x_opt[i] = volterra_eq(t, lambda s: np.interp(s, t_eval, u_opt), x_interp)

optimal_profit = -result.fun

# Visualisation
plt.figure(figsize=(12, 9))

plt.subplot(2, 1, 1)
plt.plot(t_eval, x_opt, 'b-', linewidth=2)
plt.ylabel('Biomasse (x)')
plt.title(f'Dynamique optimale de la biomasse (Profit = {optimal_profit:.2f})')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_eval, u_opt, 'r-', linewidth=2)
plt.xlabel('Temps')
plt.ylabel('Effort de récolte (u)')
plt.title('Contrôle optimal de récolte')
plt.grid(True)

plt.tight_layout()
plt.show()
# Ce code implémente un modèle de gestion de la biomasse avec optimisation du contrôle de récolte
# en utilisant l'équation intégrale de Volterra et la méthode de minimisation.
print("Optimisation terminée.")
print("Profit optimal =", optimal_profit)
print("Contrôle optimal u =", u_opt)