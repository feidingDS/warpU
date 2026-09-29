# Reproducing the results of the JASA article

*Channeling Multimodality Through a Unimodalizing Transport: Warp-U Sampler and Stochastic Bridge Sampling Estimator* (Ding, He, Jones & Meng, *Journal of the American Statistical Association*, 2026, [doi:10.1080/01621459.2026.2691322](https://doi.org/10.1080/01621459.2026.2691322)).

This document describes how to reproduce the simulation study and the real-data analysis in the paper. The work introduces a novel MCMC method for Bayesian Sampling and evidence estimation, with applications to exoplanet detection using radial velocity data.



## Repository Structure


The folder `./simulation` for the simulation section has the following structure:

```
./simulation

  ├── estimator
  ├── neuralode 
  ├── params
  └── vifitting
```


The folder `./realdata` for the real data analysis has the following structure:

```
./realdata

  ├── code/                       # Source code
  │   ├── environment/           # Environment configuration
  │   ├── functions/            # Core functions
  │   │   ├── warpu/           # Warp-U MCMC implementation
  │   │   ├── mcmc/            # MCMC utilities
  │   │   └── model/           # Model definitions
  │   ├── main/                # Main analysis scripts
  │   └── scripts/             # Utility scripts
  ├── data/                    # Data files
  ├── manuscript/             # Manuscript materials
  │   ├── figures/           # Final figures
  │   └── tables/            # Final tables
  └── output/                 # Generated outputs
      ├── results/           # Analysis results
      ├── figures/           # Generated figures
      ├── tables/           # Generated tables
      └── logs/             # Execution logs
```

## Expected Runtime

Hardware: CPU AMD 9950X, 128GB Memory.

- Environment Setup: ~30 minutes
- Simulation Study: ~15 hours
- Real Data Analysis, approximately broken down as:
  - Step 2 — Warp-U MCMC samples (`get_Adaptive_WarpU_samples_real.R`): ~10 hours
  - Step 3a — PT temperature tuning (`tune_PL_real.R`): ~15 hours
  - Step 3b — PT sampling (`get_sample_real_PL.R`): ~10 hours
  - Step 4 — Numerical integral reference (`Integral_results.R`): ~25 hours
  - Step 5 — Figure 6 plot (`RVdata_illustration_plot_density.R`): a few minutes
  - Step 6 — 50 parallel log-evidence repeats for Warp-U and PT (`run_parallel_warpU.sh`, `run_parallel_PT.sh`): varies with core count
  - Step 7 — RMSE summary for Table 2 (`Comparison_estimation_real_results_summary.R`): under a minute

These are rough estimates on the hardware above; actual runtimes will vary with core count and BLAS settings.



## Requirements

The system environment where the package was tested:

- Ubuntu
- R (== 4.4.2)
- Python (==3.11.7)


### Python Package Dependencies

Python package dependencies are specified in the file `environment.yml`. Run the following

```
conda env create -f environment.yml
conda activate warpu
```

The above will create the `warpu` environment and install the following packages with the tested version

- pip=23.3.1
- Numpy=1.26.4
- Cython=3.0.10
- threadpoolctl=2.2.0
- Scipy=1.10.1
- PyTorch=2.5.1
- torchdiffeq=0.2.4 (https://github.com/rtqichen/torchdiffeq)
- rpy2=3.5.11
- matplotlib=3.8.0
- tqdm
- seaborn
- POT
- ipywidgets

### R Package Dependencies


Package environment: This project uses `renv` for R package management. All required packages for real data analysis can be installed by going into the `realdata` directory

```bash
cd ./realdata
R
```

Then running inside R that

```R
#install.packages("renv")
renv::restore()
```

The initialization script will automatically install all required R packages and create a snapshot of the environment.


- Data manipulation and visualization:
  - `ggplot2`, `reshape2`, `ggpubr`, `gridExtra`
  - `MASS`, `Rfast`, `RiskPortfolios`

- MCMC and statistical computing:
  - `mvtnorm`, `mnormt`, `mclust`, `transport`
  - `FNN`, `e1071`, `sn`, `rstan`
  - `LaplacesDemon`, `bayestestR`

- Parallel computing:
  - `doRNG`, `doParallel`
  - `future`, `furrr`

- Additional utilities:
  - `rTensor`, `ggsci`, `numbers`
  - `microbenchmark`, `cubature`


## Steps to Reproduce for Simulation Study

In this section, "./" represents the current directory containing this readme file (i.e. the project root directory).

First install the developed python package:

```bash
conda activate warpu
cd ./pywarpu
pip install .
```

To execute the code for simulation, run the following commands:

```bash
conda activate warpu
cd ./simulation
bash exec_all.sh
```

After running the above script, the results will be stored in the folder `./simulation/results/`. 


Finally, run the jupyter notebook file `./simulation/main.ipynb` to reproduce the figures in the paper.


## Steps to Reproduce for Real Data Analysis


The reproduction procedures are divided into several parts. Results are stored as R data files (".RData") and can be used to generate figures and tables.


### Data description and access

The real data analysis uses the dataset described in Section 4 of the paper: simulated
radial-velocity (RV) observations of a star from the **Extremely Precise Radial Velocities
(EPRV3) Evidence Challenge** (Nelson et al., 2020), used in the context of RV exoplanet
detection. Each observation provides the time of measurement and its measurement error
(standard deviation). The processed posterior setup (data plus likelihood/prior closures)
is provided as `realdata/data/dataset1_one_planet_posterior.RData` and is the primary input
to all sampling scripts. All data required to reproduce the manuscript results are bundled
in `realdata/data/`; see [`realdata/data/README.md`](./realdata/data/README.md) for per-file
descriptions.


### Methodology vs. reproduction code (real-data analysis)

The files in this directory `realdata/code/` fall into two categories:

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


### Figure / Table → script mapping

The real-data section of the manuscript produces **Figure 6** and **Table 2**:

| Manuscript artifact | Generating script | Output file |
|---|---|---|
| Figure 6 (RV signal + density comparison) | `realdata/code/main/figure6/RVdata_illustration_plot_density.R` | `realdata/output/figures/p_density_final.pdf` |
| Table 2 (RMSE of log-evidence estimators) | `realdata/code/main/figure6/Comparison_estimation_real_results_summary.R` | results are saved and then displayed in `realdata/manuscript/reproduce_results.html` |


### Figure 6 and Table 2: Real Data Analysis


Make sure the R environment is properly initialized and change from the project root directory  to `./realdata/`.


```bash
cd ./realdata
# Add execution permission to the script
chmod +x code/scripts/reproduce_realdata_analysis.sh

# Run the script from the project root directory
./code/scripts/reproduce_realdata_analysis.sh
```

The script will automatically:

- Create necessary output directories
- Initialize the R environment with required packages
- Generate Warp-U MCMC samples
- Run Parallel Tempering (PT) analysis:
  - Temperature ladder tuning (if needed)
  - PT sampling
- Perform numerical integration for target density approximation
- Generate RV illustration and density plots
- Compute normalizing constant estimates (parallel computation)
- Generate RMSE summary

All outputs will be organized as follows:

- Logs: `output/logs/`
- Results: `output/results/`
- Figures: `output/figures/`

Key output files:

- `output/results/AdaptiveWarpU_Real.RData`: MCMC samples
- `output/results/PL_samples_Real.RData`: PT samples
- `output/results/int_results_gridfrom_Real.RData`: Numerical integration results for target density
- `output/figures/p_density_final.pdf`: Final density comparison plot

Note: 
- All R scripts are executed within the renv environment to ensure package version consistency
- Progress and any errors will be logged in the `output/logs` directory with timestamps


### Generating Experiment Report

After completing all analysis steps and ensuring the environment is properly set up, you can generate a comprehensive experiment report by the following R command:

```R
# Inside R
setwd("manuscript")
# Generate HTML report (recommended for interactive viewing)
rmarkdown::render('reproduce_results.Rmd', output_format = "html_document")
```

This will generate a file `manuscript/reproduce_results.html`. This report includes:

- Figure 6: RV signal and marginal posterior plots
- Table 2: RMSE comparison results

Note: 
- If you encounter any environment-related errors, please ensure the renv environment is properly restored using `renv::restore()`

