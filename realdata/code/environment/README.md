# Environment Configuration

This project uses the following programming languages and environment management tools:

## R Environment
- R Version: 4.4.2
- Environment Management Tool: renv (>= 0.17.0)
- Package Management File: renv.lock

### Dependencies
All packages are from official CRAN repository:

#### Core Analysis Packages:
- FNN (>= 1.1.3)
- mvtnorm (>= 1.1-3)
- mnormt (>= 2.0.2)
- numDeriv (>= 2016.8-1.1)
- stats4 (>= 4.0.0)
- MASS (>= 7.3-54)
- ggplot2 (>= 3.4.0)
- mclust (>= 5.4.10)
- transport (>= 0.12-2)
- e1071 (>= 1.7-9)
- sn (>= 2.1.0)
- RiskPortfolios (>= 2.1.6)

#### Document Generation Packages:
- knitr (>= 1.42)
- rmarkdown (>= 2.20)
- kableExtra (>= 1.3.4)
- gridExtra (>= 2.3)

## Setup Instructions

1. Install R (>= 4.0.0)
   ```bash
   # macOS (using Homebrew)
   brew install r
   
   # Ubuntu/Debian
   sudo apt-get install r-base
   ```

2. Install renv package:
   ```r
   install.packages("renv")
   ```

3. Initialize environment:
   ```r
   source("code/environment/init_renv.R")
   ```

4. Verify environment:
   ```r
   source("code/environment/verify_renv.R")
   ```

## Troubleshooting

### rmarkdown Installation Issues
If rmarkdown installation fails, try:
1. Using binary installation:
   ```r
   install.packages("rmarkdown", type="binary")
   ```
2. Check system dependencies:
   - macOS: Ensure XQuartz is installed
   - Linux: Install pandoc and related dependencies
     ```bash
     sudo apt-get install pandoc pandoc-citeproc
     ```

### Other Issues
- Check `renv/sessionInfo.txt` for detailed information if package installation fails
- Use `renv::diagnose()` to check environment issues
- To reset environment, use `renv::restore(clean=TRUE)`

## Notes
- All package versions are locked in renv.lock
- RStudio is recommended as IDE
- First-time setup may take some time to download and install dependencies 