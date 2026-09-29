# Code Documentation

This directory contains all implementation code and analysis scripts used in the paper.

## Methodology vs. reproduction code (real-data analysis)

The files in this directory fall into two categories:

- **Methodology code (used in the real-data analysis)** — the implementation of the Warp-U sampler, bridge
  estimators, and the exoplanet model. These are the core algorithms
  described in the paper:
  - `functions/` — the primary methodology library (adaptive Warp-U MCMC,
    GWL / PT utilities, exoplanet model definitions).
  - `functions_used/warpu code/` — additional bridge-sampling and EM
    routines still sourced by the reproduction scripts.

- **Reproduction code (real-data analysis)** — scripts that consume the methodology to
  regenerate the paper's outputs. These are what
  `scripts/reproduce_realdata_analysis.sh` runs end-to-end:
  - `main/figure6/` — the pipeline for Figure 6 and Table 2 (Warp-U / PT
    samples, numerical-integration reference, RMSE summary, and plot).
  - `scripts/` — shell drivers (the main driver plus the two
    `run_parallel_*` scripts that launch the 50 repeats).
  - `environment/` — `renv` initialisation / verification helpers.

## Directory Structure

### environment/
Environment configuration files:
- `init_renv.R`: R environment initialization
- `verify_renv.R`: R environment verification


### functions/
Core function implementations:
- `warpu/`: Warp-U MCMC core implementation
  - `Functions_AdaptiveWarpU.R`: Adaptive Warp-U algorithm
  
- `mcmc/`: MCMC utility functions
  - `GWLD.R`: General MCMC implementation
  - `GWLD_skewT.R`: MCMC implementation for skew-t distribution
  - `PL_functions_used.R`: Parallel tempering implementation functions
  
- `model/`: Model definitions
  - `log_like.R`: Likelihood functions
  - `log_post.R`: Posterior distributions
  - `planet_model.R`: Planet model

### main/
Main analysis scripts:
- `figure6/`: Figure 6 related analysis
  - `warpU/`: Warp-U MCMC analysis scripts
  - `PT/`: Parallel Tempering analysis scripts
  - `Integral_results.R`: Numerical integration for target density
  - `RVdata_illustration_plot_density.R`: RV data visualization
  - `Comparison_estimation_real_results_summary.R`: RMSE summary generation

### scripts/
Utility scripts:
- `reproduce_realdata_analysis.sh`: Main reproduction script for real data analysis
  - Automates all steps in sequence
  - Handles environment initialization
  - Manages parallel computations
  - Monitors execution progress
  - Expected runtime: >30 hours total
  - Key steps:
    1. Environment initialization
    2. Warp-U MCMC sampling 
    3. Temperature tuning (if needed)
    4. PT sampling
    5. Numerical integration
    6. RV illustration
    7. Parallel normalizing constant estimation
    8. RMSE summary generation
- `run_parallel_warpU.sh`: Warp-U parallel execution (50 repeats)
- `run_parallel_PT.sh`: Parallel Tempering parallel execution (50 repeats)

## Reproduce paper results:
To reproduce the paper results, please refer to the main README.md in the project root directory. The main reproduction script `reproduce_realdata_analysis.sh` will execute all necessary steps in the correct order with proper error handling and progress monitoring. 