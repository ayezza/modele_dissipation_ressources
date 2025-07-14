# Biomass Optimization Model

This project implements a numerical optimization model for biomass management, considering both Volterra integral equations (with memory effects) and differential equations approaches.

The Volerra integral equation case is formulated based on my doctorate thesis in 1991, fourth application (pp. 159-169): 
[Thèse de doctorat Yezza Abdelwahab]
(https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/78279918/d95e5505-1b4f-478e-b54e-def156526be4/Yezza_Abdelwahab_1991_these.pdf?AWSAccessKeyId=ASIA2F3EMEYE2HROMF6D&Signature=%2FynGyjdRH6KUAKBLFLXxGk9U3Zg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBcaCXVzLWVhc3QtMSJGMEQCIC9%2BnekCn3f2MjuBKa6XbYl3yG0tHzm4wyifh1Xm3%2B%2FYAiAWXOOFx6cp4KALikV18H7OvuoT%2BHvuC4ZptcQXAxuneSrxBAgwEAEaDDY5OTc1MzMwOTcwNSIMi9kdq1RFwu%2BHu5zoKs4Ej3MrAGRooGVIpCqztu3ib99Il7KBUEpRRGKFGZySb8AAHTE5Gn1GV27DyYSxX5d2i4wVYQehSL7ZXsP4CY3gWjUldTAW55jVSb2badjzN1JioSFMeBQctunCuasYDvqQghh5bMTq14NeVj%2BmeIdEEn53hfjanxTjW%2BMLPG2oFLKvolqvyF5BjB7OLaY58A9Ny9vLS2Qa2jmI4%2F77L9fDO587qGPgJprjGWGUD6LLPPKNodOhaSe3fhJU%2B7r48jDpS48evfjG8RWk6Nvyk%2B9Qn%2B09uLY7qfJ1tB795JSzuUrlTocyQ0bJ2MBS4e0IUzqZ39Lsi5IuaHDgCBkn1MAGuPMX%2FHGOWf3CcYQlJKZi9Z3yff4D6EjEQbOTwCsUYZQeUUJO79PNKTz4EnffP4%2BhI4Wy%2BNHxt4AVAwx6kkX7Uun2JWXUQkN01sTnSodKwrggXwmsE8dZzetXIlUJoWmuAytIPZssmEkTdjiiGbjn80s64UkMCSKhWI%2FGMziGA24x0Bnisdj2Ga3jpAJ3prmihLHQi5zGwR27wM4J8xN8x0vDIMplhWml039SJ0n2e8TqbBdei6M6S%2FQCd5RA4xsIYEKM4lJ2%2F7J2N3KIq4YJRW%2ByHgM1ACV6a3iTnzeCf77AtvPYy1t%2BbIm238zfRzlrlMdYJXIJF0kyGUX4UrW5JhWka7Re2IBcaJF9thj9IdCcRW5g6PruSDwEyM2CDJmRhQF7DvJO2hEZTUgSIkfVj%2FhHdqLR6vqPE022ViZTs0bS46VQJxqB%2FGhPcnEUtagw1bDUwwY6mwHgb3gc1IHg4afF0a1Ucz41scgma1C1%2B%2FsPwGl6GsAYDkt3ZMnbMfoo%2Fz3oWwbZBqblQ7l6ueoAiOnHDhlDc%2B%2ByGd047A27q%2BjZSeooMajiwxb0wOxAtJ%2BLzlS5%2BnxpXXZsRCBS%2B45qDFsz%2FCVylhEXmXJOGqXaFuwNu0HuuE2RK8JEopjHym1YhpYjQtcyrrQjamgrEdxM7pSkZA%3D%3D&Expires=1752504600) - Ph.D Yezza Abdelwahab, 1991



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