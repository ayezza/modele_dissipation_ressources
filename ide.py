import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# Paramètres
r, K = 0.5, 10000
B0 = [8000]
t_span = (0, 100)

# Le "noyau de mémoire" G(s). Ici une simple exponentielle décroissante.
def memory_kernel(s):
    return np.exp(-s)

# Fonction de l'IDE. C'est la partie complexe.
def ide_model(t, B, sol):
    # Si t=0, l'intégrale est nulle
    if t == 0:
        integral_term = 0
    else:
        # L'intégrale de G(s) * B(t-s) ds de s=0 à t
        # On utilise l'objet solution 'sol' pour obtenir les valeurs passées de B
        integrand = lambda s: memory_kernel(s) * sol(t - s)[0]
        integral_term, _ = quad(integrand, 0, t)

    dB_dt = r * B[0] * (1 - (1 / K) * integral_term)
    return [dB_dt]

# Pour résoudre, on a besoin d'une approche par étapes, car la fonction 
# dépend de la solution elle-même. C'est un défi.
# Une approche simplifiée (mais moins précise) serait de discrétiser le temps 
# et de calculer l'intégrale comme une somme à chaque pas.

# Voici un exemple avec une boucle manuelle (méthode d'Euler, pour l'illustration)
dt = 0.1
times = np.arange(t_span[0], t_span[1], dt)
biomass_history = [B0[0]]

for i, t in enumerate(times[1:]):
    # Calcul de l'intégrale comme une somme discrète sur l'historique
    integral_term = 0
    for j in range(len(biomass_history)):
        s = (i-j) * dt
        integral_term += memory_kernel(s) * biomass_history[j] * dt
    
    B_current = biomass_history[-1]
    dB_dt = r * B_current * (1 - (1/K) * integral_term)
    B_new = B_current + dB_dt * dt
    biomass_history.append(B_new)

# Affichage
plt.figure(figsize=(10, 6))
plt.plot(times, biomass_history)
plt.title("Simulation d'une IDE (approche discrétisée)")
plt.xlabel('Temps')
plt.ylabel('Biomasse')
plt.grid(True)
plt.show()