# Data Directory

## Data Files

The following data files are used in the analysis:

### Data Files
1. `dataset1_one_planet_posterior.RData`
   - Description: Raw data for one-planet model
   - Usage: Primary dataset for Bayesian evidence analysis

2. `stanData.RData`
   - Description: Stan model for real data analysis
   - Usage: Bayesian evidence analysis

3. `prior_bounds.RData`
   - Description: Prior bounds for model parameters
   - Usage: Used in MCMC sampling to define parameter space

4. `AdaptiveWarpU_Real_plot.RData`
   - Description: Warp-U MCMC sampling results used in rmarkdown to generate plot
   - Usage: Used for density estimation comparison and visualization

5. `PL_samples_Real_plot.RData`
   - Description: Parallel Tempering sampling results used in rmarkdown to generate plot
   - Usage: Used for comparing different sampling methods

6. `real_tmperature_get.RData`
   - Description: Optimized temperature ladder for Parallel Tempering
   - Usage: Used in PT sampling for efficient chain mixing

7. `int_results_gridfrom_Real.RData`
   - Description: Numerical integration results for density estimation
   - Usage: Used as reference for true density comparison

8. `one_planets_full_prior_fit_only_3chains_3000grid_3000iters.RData`
   - Description: Full prior fit results with 3 chains
   - Usage: Model fitting analysis

9. `AdaptiveWarpU_Real.RData`
   - Description: Warp-U data for exoplanet model
   - Usage: Used for normalizing constant estimation

10. `PL_samples_Real.RData`
   - Description: Parallel Tempering data for exoplanet model
   - Usage: Used for normalizing constant estimation

11. `PT_samples_Real_cost_large_Comparison_combined.RData`
    - Description: Combined PT sampling results to generate table
    - Usage: Performance analysis of PT method

12. `warpU_samples_Real_cost_large_Comparison_combined.RData`
    - Description: Combined Warp-U sampling results to generate table
    - Usage: Performance analysis of Warp-U method

## Data Format Details

- All .RData files can be loaded in R using the `load()` function

## Data Access

All data files are included in this repository and are publicly available. The data represents simulated and real astronomical observations that are not subject to any confidentiality restrictions.

## Data Usage

For examples of how to load and use these datasets, see:
- `code/main/figure6/warpU/get_Adaptive_WarpU_samples_real.R` 
- `code/main/figure6/PT/get_sample_real_PL.R`

