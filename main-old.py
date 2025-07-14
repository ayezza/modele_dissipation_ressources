import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp

class ModeleDissipation:
    def __init__(self, r, K, E, p):
        """
        Initialise les paramètres du modèle.
        :param r: Taux de croissance naturel
        :param K: Niveau de saturation (biomasse maximale)
        :param E: Effort d'exploitation maximal
        :param p: Paramètre de dissipation
        """
        self.r = r
        self.K = K
        self.E = E
        self.p = p

    def croissance_logistique(self, t, x):
        """
        Fonction de croissance logistique avec dissipation.
        :param t: Temps
        :param x: Taille de la population
        :return: Taux de variation de la population
        """
        dissipation = np.exp(-self.p * t)  # Fraction de dissipation
        croissance_naturelle = self.r * x * (1 - x / self.K) # Taux de croissance logistique
        exploitation = self.E * x
        return dissipation * croissance_naturelle - exploitation

    def resoudre(self, x0, temps, points=5000):
        """
        Résout l'équation différentielle pour le modèle.
        :param x0: Taille initiale de la population
        :param temps: Intervalle de temps (début, fin)
        :param points: Nombre de points pour l'évaluation
        :return: Solution de l'équation différentielle
        solution de l'équation de RICCATI:
        x(t) = x0*self.K /(x0+ (self.K-x0)*math.exp(-self.r * t))
        """
        
        t_eval = np.linspace(temps[0], temps[1], points)
        solution = solve_ivp(self.croissance_logistique, temps, [x0], method='RK45', t_eval=t_eval)
        return solution


class ModeleIntegral:
    def __init__(self, kernel, f):
        """
        Initialise les paramètres du modèle intégral.
        :param kernel: Fonction noyau K(t, s)
        :param f: Fonction inhomogène f(t)
        """
        self.kernel = kernel
        self.f = f

    def resoudre(self, t, n_iter=10):
        """
        Résout une équation intégrale de Volterra du second type :
        g(t) = f(t) + ∫[0, t] K(t, s) g(s) ds
        :param t: Points de temps pour l'évaluation
        :param n_iter: Nombre d'itérations pour l'approximation
        :return: Solution g(t)
        """
        g = np.zeros_like(t)  # Initial guess for g(t)
        for _ in range(n_iter):
            g_new = np.zeros_like(t)
            for i, ti in enumerate(t):
                integral, _ = quad(lambda s: self.kernel(ti, s) * g[np.searchsorted(t, s)], 0, ti)
                g_new[i] = self.f(ti) + integral
            g = g_new
        return g


def main():
    # Définir les paramètres pour le modèle différentiel
    r = 50.0  # Taux de croissance naturel
    K = 5000  # Niveau de saturation (biomasse maximale)
    E = 3  # Effort d'exploitation maximal
    p = 0.01  # Paramètre de dissipation

    # CAS D'UN MODÈLE DIFFÉRENTIEL
    #//////////////////////////////////////////////////////////////////////

    # Initialiser le modèle différentiel
    modele_diff = ModeleDissipation(r, K, E, p)

    # Conditions initiales pour le modèle différentiel
    x0 = 500  # Taille initiale de la population
    temps = [0, 50]  # Intervalle de temps (en années)

    # Résoudre l'équation différentielle
    solution_diff = modele_diff.resoudre(x0, temps)

    # Afficher les résultats du modèle différentiel
    plt.plot(solution_diff.t, solution_diff.y[0], label="Population (Cas équation différentielle)")
    plt.xlabel("Temps (années)")
    plt.ylabel("Population")    
    plt.title("Modèle Différentiel de Dissipation")
    plt.legend()
    plt.grid()
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.tight_layout()
    plt.show()
    


    # CAS D'UN MODÈLE INTEGRAL
    #//////////////////////////////////////////////////////////////////////

    # Définir les paramètres pour le modèle intégral
    kernel = lambda t, s: np.exp(-(t - s))  # Exemple de noyau ( r(t-s) )
    # f = lambda t: np.sin(t)  # Exemple de fonction inhomogène
    f= lambda t: t # Exemple de fonction inhomogène (peut être modifié)

    # Initialiser le modèle intégral
    modele_int = ModeleIntegral(kernel, f)

    # Points de temps pour le modèle intégral
    t = np.linspace(0, 50, 250)   # Intervalle de temps (0 à 100)

    # Résoudre l'équation intégrale
    solution_int = modele_int.resoudre(t)

    # Afficher les résultats du modèle intégral
    plt.plot(t, solution_int, label="Solution (cas modèle Intégral)")

    # Finaliser le graphique
    plt.xlabel("Temps")
    plt.ylabel("Population")
    plt.title("Modèle Intégral")
    plt.grid()
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()