# Biomass Optimization Model

This project implements a numerical optimization model for biomass management, considering both Volterra integral equations (with memory effects) and differential equations approaches.

The Volerra integral equation case is formulated based on my doctorate thesis in 1991, fourth application (pp. 159-169): 
[Thèse de doctorat Yezza Abdelwahab](https://umontreal.scholaris.ca/bitstreams/f3f9d2c1-d190-4aef-997d-1e51bc87eb4a/download) - Ph.D Yezza Abdelwahab, 1991.



## Description

The model optimizes the harvesting effort to maximize profit while maintaining sustainable biomass levels. It supports two different modeling approaches:
- Volterra integral equation (with memory/dissipation effects)
- Standard differential equation (logistic growth with harvesting)

### 🎯 Key Features

- Biomass dynamics simulation using either Volterra or differential equations
- Profit optimization with harvesting constraints
- Detailed visualization of results (5 different plots)
- Export of results to CSV and Excel files
- Configurable parameters via JSON file

## 🛠️ Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install numpy pandas scipy matplotlib openpyxl
```

## 📦 Configuration

Create a `params.json` file with the following parameters that can be modified from which the main program will extract the parameters:

```json
{
    "T": 20,                  // Time horizon generally in years
    "N": 240,                // Number of time steps for simulation
    "x0": 500.0,             // Initial biomass
    "r_growth": 5.0,         // Growth rate
    "K": 5000.0,             // Carrying capacity
    "dissipation_rate": 0.05,// Memory effect parameter (applies to Volterra model only)
    "interest_rate": 0.05,   // Economic discount rate
    "unit_cost": 2.0,          // Cost per unit effort
    "unit_price": 10.0,        // Price per unit biomass
    "E": 1.0,                  // Maximum harvesting effort
    "model_type": "volterra" // Model type: "volterra" or "differential"
}
```

## 🚀 Usage

Run the model:
```bash
python main.py
```

The program will:
1. Load parameters from `params.json`
2. Optimize the harvesting strategy depending of the model type (Volterra or Differential)
3. Generate visualizations and save them as png images in the `output` directory
4. Save results to CSV and Excel files in the `output` directory

## 📋 Output

### Visualizations
These graphs are generated in the same figure:
- Biomass trajectory
- Control effort
- Extracted biomass
- Instantaneous profit
- Cumulative profit

### Files Generated
All output files are prefixed by a unique GUID:

- `biomass_optimization_results_GUID.png`: Combined visualization of all results
- `biomass_optimization_timeseries_GUID.csv`: Time series data
- `detailed_results_GUID.xlsx`: Detailed results with multiple sheets including:
  - **Detailed_Results:** Time series of all variables
  - **Parameters:** Model configuration

## Examples of output files

### Graphs (Volterra Model and Differential Model)
<img src="https://github.com/ayezza/modele_dissipation_ressources/blob/main/output/biomass_optimization_results_02c582ef.png" alt="Biomass Optimization Results" width="1200" height="800">
<img src="https://github.com/ayezza/modele_dissipation_ressources/blob/main/output/biomass_optimization_results_28c03176.png" alt="Biomass Optimization Results" width="1200" height="800">



### CSV 
<img src="https://github.com/ayezza/modele_dissipation_ressources/blob/main/output/csv_output.png" alt="CSV Output" width="1200" height="800">

### Excel 
<img src="https://github.com/ayezza/modele_dissipation_ressources/blob/main/output/excel_output.png" alt="Excel Output" width="1200" height="800">



## Mathematical Model

### Objective Function
Maximize the discounted profit:
```math
J = ∫0^T e^{-δt} (c - p * x(t))u(t) dt
```

### Constraints
- 0 ≤ u(t) ≤ E (Harvesting effort bounds)
- x(t) ≥ 0 (Non-negative biomass)

### Dynamics
#### Volterra Model
```math
x(t) = x₀ + ∫₀ᵗ[rx(s)(1-x(s)/K)e⁻ᵖ⁽ᵗ⁻ˢ⁾ - u(s)x(s)]ds
```

#### Differential Model
```math
dx/dt = rx(1-x/K) - ux
```

## 📁 Project Structure

```
modele_dissipation_ressources/
├── main.py              # Main application file with BiomassModel class
├── params.json          # Configuration parameters
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── output/            # Generated output directory
    ├── biomass_optimization_results_GUID.png    # Visualization plots
    ├── biomass_optimization_timeseries_GUID.csv # Time series data
    └── detailed_results_GUID.xlsx              # Detailed analysis results
```

### Key Files Description

- `main.py`: Contains the core `BiomassModel` class implementing:
  - Biomass dynamics simulation (Volterra/Differential)
  - Optimization algorithms
  - Visualization methods
  - Results generation

- `params.json`: Configuration file containing:
  - Model parameters
  - Simulation settings
  - Economic parameters

- `output/`: Directory containing generated results:
  - PNG files for visualizations
  - CSV files for time series data
  - Excel files with detailed analysis



## 📜 License

This project is open source and available under the MIT License.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.