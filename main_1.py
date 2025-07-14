import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# Import garbage collector to manage memory 
import gc


def run_model(k_factor, initial_population, n_years, initial_effort, P, r, kernel=None):
    """
    Run the model for :
    k_factor: factor to adjust the carrying capacity K.
    initial_population: initial population.
    n_years: number of years, 
    initial_effort: initial effort, 
    P: price per unit harvested, 
    r: intrinsic growth rate.
    Returns the time, population, optimal effort, optimal revenue for corresponding carrying capacity K.
    """
    # K is the carrying capacity of the population
    K = initial_population * k_factor  # Capacité de charge max paramétrique
    q = 0.1  # Coefficient de capturabilité (catchability coefficient)
    # Define the maximum effort 
    max_effort = 1.0  # Maximum effort allowed, can be adjusted
    # Define the initial effort
    initial_effort = initial_effort if initial_effort <= max_effort else max_effort
    # Ensure n_years is an integer  
    n_years = int(n_years)
    # Ensure initial_population is a positive integer
    initial_population = int(initial_population) if initial_population > 0 else 1 
    # Ensure r is a positive float
    r = float(r) if r > 0 else 1e-6  # Avoid division by zero in growth rate
    # Ensure P is a positive float
    P = float(P) if P > 0 else 1.0  # Price per unit harvested, default to 1.0 if not positive
    # Ensure kernel is a valid string or None
    if kernel is not None and not isinstance(kernel, str):
        raise ValueError("Kernel must be a string or None. Valid options are 'logistic_growth_kernel', 'sustainable_yield_kernel', 'capture_effort_kernel'.")
    # Ensure kernel is one of the valid options
    valid_kernels = ["logistic_growth_kernel", "sustainable_yield_kernel", "capture_effort_kernel"]
    if kernel is not None and kernel not in valid_kernels:
        raise ValueError(f"Kernel must be one of {valid_kernels}. Provided: {kernel}")  
      
    
    
    # Define the logistic growth model with harvesting (AUTHOR: P. F.Verhuls)
    def logistic_growth_kernel(x, u):
        """
        Logistic growth model with harvesting.
        x: population size at time t.
        r: intrinsic growth rate of the population.
        K: carrying capacity of the population.
        u: harvesting effort at time t.
        """
        return r * x * (1 - x / K) - u * x
    
    
    # Schaefer, 1954 generalized model for capture effort
    # Capture effort model
    def capture_effort_kernel(x, u):
        """
        Capture effort model.
        x: population size at time t.
        u: harvesting effort at time t, ex?, est l'effort de pêche (ex: nombre de jours-bateaux, nombre de filets).
        q est le coefficient de capturabilité (l'efficacité d'une unité d'effort de pêche).
        """
        return q * u * x
    
    
    # Schaefer, 1954 generalized Sustainable Yield model for capture effort
    # B_eq (Biomasse d'équilbre)  = K * (1 - (q * u) / r)
    # càd: x = K * (1 - (q * u) / r)
    def sustainable_yield_kernel(q, u):
        """
        Sustainable yield model.
        x: population size at time t.
        u: harvesting effort at time t.
        """
        if u == 0:
            return K * (1 - (q * u) / r)
        else:
            # Calculate the sustainable yield based on the effort and carrying capacity
            # B_eq = K * (1 - (q * u) / r)
            # Rearranging gives us the yield at equilibrium 
            return  q * K * u * (1 - (q*u)/r)
    
    # Set default kernel if not provided
    if kernel is None or kernel == "logistic_growth_kernel":
        selected_kernel = logistic_growth_kernel
    elif kernel == "sustainable_yield_kernel":
        # For sustainable yield, we need to define q (catchability coefficient)
        q = 0.1  # Example value for q, can be adjusted
        r = r if r != 0 else 1e-6  # Avoid division by zero
        print(f"q: {q},  r: {r}, K: {K}")
        selected_kernel = sustainable_yield_kernel
    elif kernel == "capture_effort_kernel":
        selected_kernel = capture_effort_kernel
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    

    
    def dynamique(t, x, u, kernel=selected_kernel):
        u_t = np.interp(t, np.linspace(0, len(u) - 1, len(u)), u)
        return kernel(x, u_t)  #  r * x * (1 - x / K) - u_t * x

    def solve_system(u, t_span, x0, kernel=selected_kernel):
        def dyn(t, x): return dynamique(t, x, u, kernel=kernel)
        solution = solve_ivp(dyn, t_span, [x0], t_eval=np.linspace(t_span[0], t_span[1], len(u)))
        return solution.t, solution.y[0]


    # objective function maximizing total revenue
    # P is the price per unit of population harvested
    # u is the effort vector
    # population is the resulting population vector after harvesting
    # r is the intrinsic growth rate of the population

    # Define the objective function to maximize total revenue
    # from harvesting the population over the years
    def fonction_objective(u, initial_population=initial_population, kernel=selected_kernel):
        t_span = [0, len(u)-1]  # Adjust time span to match length of u
        x0 = initial_population
        t_eval = np.linspace(t_span[0], t_span[1], len(u))  # Ensure same length as u
        _, population = solve_system(u, t_span, x0, kernel=kernel)
        
        # Ensure arrays have the same length
        if len(population) != len(u):
            # Interpolate population to match u length if necessary
            t = np.linspace(0, t_span[1], len(population))
            population = np.interp(np.linspace(0, t_span[1], len(u)), t, population)
        
        return -np.sum(P * u * population)

    
    # When creating the bounds for the optimizer:
    bounds = [(0, max_effort) for _ in range(n_years + 1)]
    u0 = np.random.uniform(0, max_effort, n_years + 1)
    u0[0] = initial_effort

    result = minimize(
        fonction_objective, 
        u0, 
        args=(initial_population, selected_kernel),  # logistic_growth_kernel or sustainable_yield_kernel etc.
        bounds=bounds, 
        method='SLSQP'
    )
    optimal_u = result.x
    time, optimal_population = solve_system(optimal_u, [0, n_years], initial_population)
    optimal_revenue = -result.fun

    return time, optimal_population, optimal_u, optimal_revenue, K

if __name__ == "__main__":
     # Base output directories
    base_output_dir = "data/output"
    
    # Paramètres fixes du modèle
    initial_effort = 0.1 # initial effort
    max_effort = 2.5 # maximum effort
    initial_population = 500 # initial population count
    # Price and growth rate
    P = 10 # price per unit harvested
    # Kernel name for the model
    kernel_name="logistic_growth_kernel"  # Default kernel name
    # Ask user for kernel choice
    kernel_choice = input(f"Choisissez le noyau (logistic_growth_kernel, sustainable_yield_kernel, capture_effort_kernel): ").strip()
    if kernel_choice in ["logistic_growth_kernel", "sustainable_yield_kernel", "capture_effort_kernel"]:
        kernel_name = kernel_choice   
    
    if kernel_name=="logistic_growth_kernel":
        # r is the intrinsic growth rate of the population
        # let's test from 10% to 100% of growth rate
        r0 = 0.1  # initial intrinsic growth rate of the population 
        r_max = 1  # maximum growth rate of the population
        r_step = .02  # step size for growth rate
    elif kernel_name == "sustainable_yield_kernel":
        # r is the intrinsic growth rate of the population
        # let's test from 10% to 100% of growth rate
        r0 = 1  # initial intrinsic growth rate of the population
        r_max = 10  # maximum growth rate of the population 
        r_step = 0.5  # step size for growth rate
    elif kernel_name == "capture_effort_kernel":
        # r is the intrinsic growth rate of the population
        # let's test from 10% to 100% of growth rate
        r0 = 1  # initial intrinsic growth rate of the population
        r_max = 10  # maximum growth rate of the population    
        r_step = 0.5  # step size for growth rate
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}. Please choose from 'logistic_growth_kernel', 'sustainable_yield_kernel', 'capture_effort_kernel'.")
    
    # Create an array of growth rates from r0 to r_max  
    r_values = np.arange(r0, r_max, r_step)
    print(f"Valeurs de r utilisées pour l'analyse: {r_values}")
    

     
    # Pour chaque valeur de r
    for r in r_values:
        # Créer un sous-dossier pour chaque valeur de r
        r_dir = f"r_{r:.2f}"
        base_r_dir = os.path.join(base_output_dir, r_dir)
        
        print(f"\nAnalyse pour r = {r:.2f} (taux de croissance {r*100:.2f}%)")
    
         # Array of different time horizons to test
        initial_period = 10  # Initial period in years
        max_periods_number = 100  # Maximum period in years
        step_size = 10  # Step size in years
        # Horizons from initial_period to initial_period years in steps of 100
        year_horizons = np.arange(initial_period, max_periods_number+1, step_size) 

        print(f"\nAnalyse des horizons temporels: {year_horizons} années")
        # Pour chaque horizon temporel
        for n_years in year_horizons:
            # Modifier les chemins de sortie pour inclure r
            year_dir = f"{n_years}_years"
            output_csv_dir = os.path.join(base_r_dir, year_dir, "CSV_files")
            output_graphs_dir = os.path.join(base_r_dir, year_dir, "graphs")
            
            # Create directories if they don't exist
            for directory in [output_csv_dir, output_graphs_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)

            # Clear memory at the start of each iteration
            plt.close('all')
            gc.collect()
            
            # Plage de valeurs pour k
            k_values = np.arange(1, 10.1, .5)
            
            # Listes pour stocker les résultats
            revenues = []
            populations_final = []
            efforts_final = []
            K_values = []
            
            # Exécution du modèle pour chaque valeur de k
            # Pour chaque valeur de k
            for k in k_values:
                # Run model and collect results
                time, pop, u, rev, K = run_model(k, initial_population, n_years, initial_effort, P, r, kernel=kernel_name)
                
                # Store results
                revenues.append(rev)
                populations_final.append(pop[-1])
                efforts_final.append(u[-1])
                K_values.append(K)

                # Ensure consistent array lengths
                time_points = np.linspace(0, n_years, len(pop))
                if len(time_points) != len(u):
                    # Interpolate effort to match population length
                    u = np.interp(time_points, np.linspace(0, n_years, len(u)), u)

                # Create and save detailed plot
                fig, ax1 = plt.subplots(figsize=(10, 6))
                line1 = ax1.plot(time_points, pop, 'b-', label='Population', linewidth=1)
                ax1.set_xlabel('Temps (années)')
                ax1.set_ylabel('Population', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0, color='blue')

                ax2 = ax1.twinx()
                line2 = ax2.plot(time_points, u, 'r-', label='Effort', linewidth=1)
                ax2.set_ylabel('Effort', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0, color='red')

                # Add legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
                plt.title(f'Évolution pour r={r:.2f}, k={k:.1f} (K={K:.0f}), {n_years} années, \
                Effort initial={initial_effort:.2f}, P (Unit Price)={P:.2f}\nUsed kernel={kernel_name}')
                
                # Save detailed plot
                plt.savefig(os.path.join(output_graphs_dir, 
                           f'evolution_r_{r:.2f}_k_{k:.1f}_{n_years}years.png'),
                           dpi=300, bbox_inches='tight')
                plt.close(fig)

                # Save detailed data
                detailed_df = pd.DataFrame({
                    'time': time,
                    'population': pop,
                    'effort': u
                })
                detailed_df['kernel'] = kernel_name
                detailed_df.to_csv(os.path.join(output_csv_dir, 
                                 f'detailed_results_r_{r:.2f}_k_{k:.1f}_{n_years}years.csv'),
                                 index=False)

            # Create and save summary plot after k loop
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
            ax1.plot(k_values, revenues, 'b-', label='Revenu')
            ax1.set_ylabel('Revenu total')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.legend()

            ax2.plot(k_values, populations_final, 'g-', label='Population finale')
            ax2.set_ylabel('Population')
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax2.legend()

            ax3.plot(k_values, efforts_final, 'r-', label='Effort final')
            ax3.set_xlabel('Facteur k')
            ax3.set_ylabel('Effort')
            ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax3.legend()

            plt.suptitle(f'Résumé pour r={r:.2f}, {n_years} années - \
                          Effort initial={initial_effort:.2f}, P (Unit Price)={P:.2f}\nUsed kernel={kernel_name}')
            plt.tight_layout()
            
            # Save summary plot
            plt.savefig(os.path.join(output_graphs_dir, 
                       f'summary_plots_r_{r:.2f}_{n_years}years.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save summary data
            summary_df = pd.DataFrame({
                'k_value': k_values,
                'revenue': revenues,
                'population_finale': populations_final,
                'effort_final': efforts_final,
                'capacity_charge': K_values
            })
            summary_df['kernel'] = kernel_name
            # Save summary results to CSV
            summary_df.to_csv(os.path.join(output_csv_dir, 
                            f'summary_results_r_{r:.2f}_{n_years}years.csv'),
                            index=False)
            
            print(f"Graphiques de résumé enregistrés dans {os.path.join(output_graphs_dir, f'summary_plots_{n_years}years.png')}")


            # Affichage des résultats optimaux
            best_k_index = np.argmax(revenues)
            print(f"Meilleure valeur de k: {k_values[best_k_index]:.2f}")
            print(f"Revenu maximal: {revenues[best_k_index]:.2f}")
            print(f"Population finale correspondante: {populations_final[best_k_index]:.0f}")
            print(f"Effort final correspondant: {efforts_final[best_k_index]:.2f}")
            print(f"Capacité de charge correspondante (K): {K_values[best_k_index]:.0f}")
            print(f"Ratio en % meilleure taille population vs capacité de charge: {100*populations_final[best_k_index]/K_values[best_k_index]:.2f}%")
            print(f"Ratio en % moyenne de la population vs capacité de charge: {100*np.mean(populations_final)/K_values[best_k_index]:.2f}%")

            # write best results to CSV
            best_results_df = pd.DataFrame({
                'best_k': [k_values[best_k_index]],
                'best_K': [K_values[best_k_index]],
                'best_revenue': [revenues[best_k_index]],
                'best_population_finale': [populations_final[best_k_index]],
                'best_effort_final': [efforts_final[best_k_index]],
                'best_ratio_population_K': [100 * populations_final[best_k_index] / K_values[best_k_index]],
                'best_ratio_mean_population_K': [100 * np.mean(populations_final) / K_values[best_k_index]]
            })
            best_results_df['kernel'] = kernel_name
            best_results_df['r'] = r  # Add r to the best results DataFrame
            # Add n_years to the best results DataFrame 
            best_results_df['n_years'] = n_years
            # Save best results to CSV
            best_results_df.to_csv(os.path.join(output_csv_dir, 
                                    f'best_results_r_{r:.2f}_{n_years}years.csv'), 
                                    index=False)
            print(f"Meilleurs résultats enregistrés dans {os.path.join(output_csv_dir, f'best_results_r_{r:.2f}_{n_years}years.csv')}")
            print(f"Analyse terminée pour {n_years} années, r={r:.2f}")
            
            # plot best_ratio_population_K and best_ratio_mean_population_K
            plt.figure(figsize=(10, 6))
            
            # sns.lineplot(data=best_results_df, x='best_k', y='best_ratio_population_K', label='Population finale / K (%)')
            # sns.lineplot(data=best_results_df, x='best_k', y='best_ratio_mean_population_K', label='Population moyenne / K (%)')
# 
            # plt.xlabel('best_k (Facteur k)')
            # plt.ylabel('Ratio (%)')
            # plt.title('Ratios Population/K et Population Moyenne/K en fonction de k pour r={r:.2f}, {n_years} années')
            # plt.ylim(0, 100)  # Optional: restrict y-axis to 0-100%
            # plt.legend()
            # plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
            # plt.tight_layout()
            # plt.savefig(os.path.join(output_graphs_dir, 
            #            f'ratios_population_K_r_{r:.2f}_{n_years}years.png'),
            #            dpi=300, bbox_inches='tight')
            # plt.show()
            
            plt.plot(best_results_df['best_k'], 
                      best_results_df['best_ratio_population_K'], 
                      label='Ratio Population/K', color='blue')
            plt.plot(best_results_df['best_k'], 
                     best_results_df['best_ratio_mean_population_K'],
                     label='Mean Ratio Population/K', color='orange')
            plt.xlabel('Years (k factor)')
            plt.ylabel('Ratio (%)')
            plt.title(f'Ratios Population/K pour r={r:.2f}, {n_years} années')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
            plt.savefig(os.path.join(output_graphs_dir, 
                        f'ratios_population_K_r_{r:.2f}_{n_years}years.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Graphiques de ratios enregistrés dans {os.path.join(output_graphs_dir, f'ratios_population_K_r_{r:.2f}_{n_years}years.png')}")

    
    # Save optimal solution for this time horizon
    results_df = pd.DataFrame({
        'k': k_values,
        'K': K_values,
        'revenue': revenues,
        'population_finale': populations_final,
        'effort_final': efforts_final,
        'ratio_population_K': [pop/K * 100 for pop, K in zip(populations_final, K_values)]
    })
    results_df['kernel'] = kernel_name
    results_df['r'] = r  # Add r to the results DataFrame
    # Modifier le nom du fichier des solutions optimales pour inclure r
    results_df.to_csv(os.path.join(output_csv_dir, 
                        f'optimal_solution_r_{r:.1f}_{n_years}years.csv'), 
                        index=False)
       
    print(f"Analyse terminée pour {n_years} années")
    
    
    
# This code is a simulation of a fish population model with corporate exploitation, using matplotlib for visualization.