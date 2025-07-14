# Biomass Optimization Model

This project implements a numerical optimization model for biomass management, considering both Volterra integral equations (with memory effects) and differential equations approaches.

The Volerra integral equation case is formulated based on my doctorate thesis in 1991, fourth application (pp. 159-169): 
[Thèse de doctorat Yezza Abdelwahab]
(https://umontreal.scholaris.ca/items/74617e11-091c-4e14-9fd7-7ef01b852eb4) - Ph.D Yezza Abdelwahab, 1991



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

Create a `params.json` file with the following parameters:

```json
{
    "T": 20,                  // Time horizon
    "N": 240,                // Number of time steps
    "x0": 500.0,             // Initial biomass
    "r_growth": 5.0,         // Growth rate
    "K": 5000.0,             // Carrying capacity
    "dissipation_rate": 0.01,// Memory effect parameter (applies to Volterra model only)
    "interest_rate": 0.05,   // Economic discount rate
    "unit_cost": 2,          // Cost per unit effort
    "unit_price": 10,        // Price per unit biomass
    "E": 5,                  // Maximum harvesting effort
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
2. Optimize the harvesting strategy
3. Generate visualizations
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
- `biomass_optimization_results.png`: Combined visualization of all results
- `biomass_optimization_timeseries.csv`: Time series data
- `detailed_results.xlsx`: Detailed results with multiple sheets including:
  - **Detailed_Results:** Time series of all variables
  - **Parameters:** Model configuration

## Mathematical Model

### Objective Function
Maximize the discounted profit:
```math
J(x,u) = ∫₀ᵀ e⁻ᵟᵗ(p·x(t)·u(t) - c·u(t))dt
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
    ├── biomass_optimization_results.png    # Visualization plots
    ├── biomass_optimization_timeseries.csv # Time series data
    └── detailed_results.xlsx              # Detailed analysis results
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