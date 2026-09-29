# Verify renv environment setup
cat("Starting renv environment verification...\n")

# Check if renv is installed
if (!require("renv", quietly = TRUE)) {
  stop("renv package is not installed! Please run init_renv.R first")
}

# Check if in renv project
if (!renv::status()$synchronized) {
  cat("Warning: renv environment is not synchronized, may need to run renv::restore()\n")
}

# Verify if all required packages are installed
required_packages <- c(
  "FNN",
  "mvtnorm",
  "mnormt",
  "numDeriv",
  "stats4",
  "MASS",
  "ggplot2",
  "mclust",
  "transport",
  "e1071",
  "sn",
  "RiskPortfolios"
)

missing_packages <- character()
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  cat("The following packages are not installed:\n")
  cat(paste("  -", missing_packages, collapse = "\n"), "\n")
  cat("Please run init_renv.R to install missing packages\n")
} else {
  cat("All required packages are correctly installed!\n")
}

# Print environment information
cat("\nEnvironment Information:\n")
print(sessionInfo())

cat("\nVerification complete!\n") 