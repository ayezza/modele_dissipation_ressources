import uuid  # used to generate unique run IDs for output files
import os
# -*- coding: utf-8 -*- 
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  



class BiomassModel:
    def __init__(self, params):
        # Initialize the model with parameters
        if not isinstance(params, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self.params = params
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.abspath('output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.output_dir = os.path.abspath(self.output_dir)
        self.validate_params(params)
        
        self.run_id = str(uuid.uuid4())[:8]  # Generate short unique ID for this run
        self.validate_params(params)

    def get_output_filename(self, base_name, extension):
        """Generate unique filename with run ID"""
        return os.path.join(self.output_dir, f"{base_name}_{self.run_id}.{extension}")
        
    def validate_params(self, params):
        # Ensure all required parameters are present
        required_params = ['T', 'N', 'x0', 'r_growth', 'K', 'dissipation_rate', 'interest_rate',
                           'unit_cost', 'unit_price', 'E', 'model_type']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

   
    def simulate_dynamics(self, u):
        """
        Simulates biomass dynamics using either:
        1. Volterra integral equation (with memory effect)
        2. Differential equation (standard logistic growth with harvesting)
        """
        T = self.params['T']
        N = self.params['N']
        dt = T / N
        t = np.linspace(0, T, N+1)
        x = np.zeros(N+1)
        x[0] = self.params['x0']

        # Choose simulation type based on params
        model_type = self.params.get('model_type', 'volterra')  # default to volterra if not specified

        if model_type == 'volterra':
            # Volterra integral equation (existing code)
            for i in range(1, N+1):
                summation = 0.0
                for j in range(i):
                    # Logistic growth with dissipation
                    growth = self.params['r_growth'] * x[j] * (1 - x[j]/self.params['K'])
                    # the retention factor for dissipation applies only in the Volterra model to include memory effect
                    retention = np.exp(-self.params['dissipation_rate'] * (t[i] - t[j]))
                    summation += (retention * growth - u[j] * x[j]) * dt
                    if summation < 0:
                        summation = 0.0
                x[i] = self.params['x0'] + summation

        elif model_type == 'differential':
            # Standard differential equation (dx/dt = rx(1-x/K) - ux)
            for i in range(N):
                growth = self.params['r_growth'] * x[i] * (1 - x[i]/self.params['K'])
                harvest = u[i] * x[i]
                x[i+1] = x[i] + (growth - harvest) * dt

        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'volterra' or 'differential'")

        # Prevent negative or nan biomass for both models
        x[~np.isfinite(x)] = 0.0
        x[x < 0] = 0.0

        return x, t
    
    def profit(self, u_func, x_func): 
        """Profit actualisé avec terme quadratique"""
        """
        Calcule le profit actualisé en intégrant la fonction de profit instantané
        sur l'intervalle [0, T] avec le contrôle u(t) et la biomasse x(t).
        
        Le profit instantané est défini par :
        profit(t) = e^{-δt} * (p - c) * x(t) * u(t)
        
        où c est le coût unitaire de l'effort, p est le prix par unité de biomasse,
        δ est le taux d'actualisation.
        """
        T = self.params['T']
        N = self.params['N']
        dt = T / N
        t = np.linspace(0, T, N+1)
        profit = 0.0
        for i in range(N):
            revenue = self.params['unit_price'] * u_func[i] * x_func[i]
            cost = self.params['unit_cost'] * u_func[i] * x_func[i]
            profit += np.exp(-self.params['interest_rate'] * t[i]) * (revenue - cost) * dt
        return profit

    
    def profit_instant(self, u, x, t):
        """Profit instantané"""
        dt = self.params['T'] / self.params['N']
        """        Calcule le profit instantané à chaque pas de temps.
        Le profit instantané est défini par :
        profit(t) = e^{-δt} * (p - c) * x(t) * u(t)
        """
        revenue = self.params['unit_price'] * u * x
        cost = self.params['unit_cost'] * u * x
 
        profit_instant = np.exp(-self.params['interest_rate'] * t) * (revenue - cost) * dt
        
        return profit_instant
    

    def objective(self, u):
        
        """
        Fonction objectif à minimiser (négatif du profit)
        Le profit est défini par l'intégrale discrétisée
        J = ∫0^T e^{-δt} (c - p * x(t))u(t) dt.
        Si la trajectoire x présente des valeurs négatives, une pénalité importante est ajoutée.
        """
        x, t = self.simulate_dynamics(u)
        profit = 0.0
        for i in range(self.params['N']):
            #  profit += np.exp(-self.params['interest_rate'] * t[i]) * u[i] * x[i] * dt
            profit += self.profit_instant(u[i], x[i], t[i]) 
        # Pénalité si la biomasse devient négative
        if np.any(x < 0):
            penalty = 1e6 * np.sum(np.abs(x[x < 0]))
            profit -= penalty
        # Nous maximisons le profit, d'où l'opposé
        return -profit



    def optimize(self):
        # ... (move your optimization code here, using self.ob jective)
        N = self.params['N']
        # Initialisation du contrôle (par exemple, effort constant à la moitié de la borne maximale)
        u0 = np.full(N, self.params['E'] / 2)  # Effort initial faible
        
        # Définir les bornes pour chaque contrôle : u[i] ∈ [0, E]
        # ensure that u does not exceed the maximum effort E and that E does not exceed 1
        bounds = [(0, min(self.params['E'], 1.0))]
        # bounds = [(0, self.params['E'])] * N  # Original line
        
        
        # On utilise la méthode SLSQP du solveur scipy.optimize.minimize
        res = minimize(self.objective, u0, bounds=bounds, method='SLSQP',
                    options={'maxiter': 1000, 'disp': True})
        
        # Récupération du contrôle optimal et calcul de la trajectoire correspondante
        u_opt = res.x
        x_opt, t = self.simulate_dynamics(u_opt)
        # Calcul du profit cumulé
        # dt = self.params['T'] / self.params['N'] 
        # profit = sum(np.exp(-self.params['interest_rate'] * t[i]) * u_opt[i] * x_opt[i] * dt for i in range(N))
        profit = sum( self.profit_instant(u_opt[i], x_opt[i], t)   for i in range(N))
        
        return u_opt, x_opt, t, profit
    
    def plot_results(self, t, x_opt, u_opt):
        fig = plt.figure(figsize=(14, 15))
        gs = gridspec.GridSpec(5, 2, width_ratios=[5, 1])  # 5 lignes, 2 colonnes (90%/10%)

        # 1. Biomasse
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, x_opt, 'b-', label='Biomasse x(t)')
        ax1.set_xticks(np.arange(0, self.params['T'] + 1, 1))  # Ajuster les ticks de l'axe x
        ax1.set_xlabel('Temps t')
        ax1.set_ylabel('Biomasse')
        ax1.set_title('Trajectoire de la biomasse')
        ax1.grid()
        ax1.legend()

        # 2. Contrôle optimal
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.step(t[:-1], u_opt, 'r-', where='post', label='Contrôle u(t)')
        ax2.set_xticks(np.arange(0, self.params['T'] + 1, 1))  # Ajuster les ticks de l'axe x
        ax2.set_xlabel('Temps t')
        ax2.set_ylabel('Effort de récolte u(t)')
        ax2.legend()
        ax2.set_ylim(0, self.params['E'] * 1.1)
        ax2.set_title('Contrôle optimal')
        ax2.grid()


        # 3. Extracted biomass
        extracted_biomass = u_opt * x_opt[:-1]  # Calculate extracted biomass
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(t[:-1], extracted_biomass, 'm-', label='Biomasse extraite')
        ax3.set_xticks(np.arange(0, self.params['T'] + 1, 1))
        ax3.set_xlabel('Temps t')
        ax3.set_ylabel('Biomasse extraite')
        ax3.set_title('Biomasse extraite au cours du temps')
        ax3.grid()
        ax3.legend()


        # 4. Profit instantané
        dt = self.params['T'] / self.params['N']
        profit_instant = np.exp(-self.params['interest_rate'] * t[:-1]) * u_opt * x_opt[:-1] * dt
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(t[:-1], profit_instant, 'g-', label='Profit instantané')
        ax4.set_xticks(np.arange(0, self.params['T'] + 1, 1))  # Ajuster les ticks de l'axe x
        ax4.set_xlabel('Temps t')
        ax4.set_ylabel('Profit instantané')
        ax4.set_title('Profit instantané')
        ax4.grid()
        ax4.legend()

        # 5. Profit cumulé
        profit_cum = np.zeros_like(t)
        for i in range(self.params['N']):
            profit_cum[i+1] = profit_cum[i] + np.exp(-self.params['interest_rate'] * t[i]) * u_opt[i] * x_opt[i] * dt
        ax5 = fig.add_subplot(gs[4, 0])
        ax5.plot(t, profit_cum, '-', label='Profit cumulé')
        ax5.set_xticks(np.arange(0, self.params['T'] + 1, 1))  # Ajuster les ticks de l'axe x
        ax5.set_xlabel('Temps t')
        ax5.set_ylabel('Profit cumulé')
        ax5.set_title('Profit cumulé')
        ax5.grid()
        ax5.legend()

        # Colonne de droite pour les paramètres
        ax_params = fig.add_subplot(gs[0, 1])  # Toutes les lignes, 2ème colonne
        ax_params.axis('off')
        ax_params.set_title("Parameters:", loc='left', fontsize=12, pad=12)  # boîte de texte pour les paramètres
        params_text = "\n".join([f"{k}: {v}" for k, v in self.params.items()])
        params_text = params_text + "\n" + "\n".join([f"Time step: {dt:.3f}",])
        ax_params.text(0, 1, params_text, fontsize=10, va='top', ha='left', 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.5', alpha=0.95))

        plt.suptitle('Optimisation du contrôle de la biomasse avec IDE', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save with unique ID
        plot_file = self.get_output_filename("biomass_optimization_results", "png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        
        # Préparation des données pour le CSV
        df = pd.DataFrame({
            "model_type": self.params['model_type'],
            "T": self.params['T'],                     # Valeur unique
            "N": self.params['N'],                     # Valeur unique
            "x0": self.params['x0'],                   # Valeur unique
            "r_growth": self.params['r_growth'],       # Valeur unique
            "K": self.params['K'],                     # Valeur unique
            "dissipation_rate": self.params['dissipation_rate'],  # Valeur unique
            "interest_rate": self.params['interest_rate'],  # Valeur unique
            "unit_cost": self.params['unit_cost'],     # Valeur unique
            "unit_price": self.params['unit_price'],   # Valeur unique
            "E": self.params['E'],                     # Valeur unique
            "time_step": t[:-1],                # N valeurs
            "x_opt": x_opt[:-1],                # N valeurs
            "u_opt": u_opt,                     # N valeurs
            "profit_instant": profit_instant,   # N valeurs
            "profit_cum": profit_cum[:-1]       # N valeurs
        })

        # Sauvegarde dans un fichier CSV with unique ID
        csv_file = self.get_output_filename("biomass_optimization_timeseries", "csv")
        df.to_csv(csv_file, index=False)
            
    
    def generate_detailed_results(self, t, x_opt, u_opt):
        """Generate detailed results for each time period"""
        
        # Initialize lists to store computed values
        dt = self.params['T'] / self.params['N']
        growth_rates = []
        extracted = []
        revenues = []
        costs = []
        profits_instant = []
        profits_cum = 0
        profits_cumulative = []
        
        # Calculate values for each period
        for i in range(len(t)-1):
            # Growth calculation
            if self.params['model_type'] == 'differential':
                growth = self.params['r_growth'] * x_opt[i] * (1 - x_opt[i]/self.params['K'])
            else:  # volterra
                growth = self.params['r_growth'] * x_opt[i] * (1 - x_opt[i]/self.params['K']) * \
                        np.exp(-self.params['dissipation_rate'] * dt)
            
            # Extraction
            extraction = u_opt[i] * x_opt[i]
            
            # Financial calculations
            revenue = self.params['unit_price'] * extraction
            cost = self.params['unit_cost'] * u_opt[i]
            profit = np.exp(-self.params['interest_rate'] * t[i]) * (revenue - cost)
            
            # Update cumulative profit
            profits_cum += profit * dt
            
            # Store values
            growth_rates.append(growth)
            extracted.append(extraction)
            revenues.append(revenue)
            costs.append(cost)
            profits_instant.append(profit)
            profits_cumulative.append(profits_cum)
        
        # Create DataFrame
        df_detailed = pd.DataFrame({
            'Model_Type': self.params['model_type'],
            'T': self.params['T'],
            'N': self.params['N'],
            'x0': self.params['x0'],
            'r_growth': self.params['r_growth'],
            'K': self.params['K'],
            'dissipation_rate': self.params['dissipation_rate'],
            'interest_rate': self.params['interest_rate'],
            'unit_cost': self.params['unit_cost'],
            'unit_price': self.params['unit_price'],
            'E': self.params['E'],
            'Time_Step': dt,
            'Growth_Rate': growth_rates,
            'Control_Effort': u_opt,
            'period': np.arange(len(t)-1),
            'Time': t[:-1],
            'Biomass': x_opt[:-1],
            'Extracted_Biomass': extracted,
            'Revenue': revenues,
            'Cost': costs,
            'Instant_Profit': profits_instant,
            'Cumulative_Profit': profits_cumulative
        })
        
        # Add model parameters as a new sheet
        params_df = pd.DataFrame([self.params])
        
        # Save to Excel with multiple sheets
        excel_file = self.get_output_filename("detailed_results", "xlsx")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_detailed.to_excel(writer, sheet_name='Detailed_Results', index=False)
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        return df_detailed
    
    
def main(params):
    # define the model and run the optimization
    model = BiomassModel(params)
    u_opt, x_opt, t, profit = model.optimize()
    # Plot the results
    model.plot_results(t, x_opt, u_opt)
    # Generate and save detailed results
    model.generate_detailed_results(t, x_opt, u_opt)
    # Print the results
    print("Profit optimal =", profit)
    print(f"Results saved to {os.path.join(model.output_dir, 'detailed_results.xlsx')}")
    print("Contrôle optimal u =", u_opt) 
    print("Trajectoire de la biomasse x =", x_opt)   


if __name__ == '__main__':
    # Load parameters from JSON file
    if not os.path.exists('params.json'):
        raise FileNotFoundError("params.json file not found. Please create it with the required parameters.")
    
    try:
        # Load parameters from JSON file
        with open('params.json', 'r') as f:
            params = json.load(f)
            print("Parameters loaded:", params) 
    except json.JSONDecodeError as e:   
        raise ValueError(f"Error decoding JSON from params.json: {e}")
    except FileNotFoundError as e:
        raise ValueError(f"params.json file not found: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")
    
    # Validate parameters
    if not isinstance(params, dict):
        raise ValueError("Parameters should be a dictionary loaded from params.json.")
            
    # Run the main function with the loaded parameters
    main(params)

