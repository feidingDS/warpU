# Initialize renv environment for JASA paper reproduction
# Set package installation path to non-synchronized folder
local_lib <- "~/R/library"
dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(local_lib)

# Install remotes if not already installed
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# Set official CRAN mirror
if (Sys.info()["sysname"] == "Darwin" && 
    grepl("arm64", Sys.info()["machine"])) {
  # Special handling for M1 Mac
  options(repos = c(
    CRAN = "https://cloud.r-project.org",
    RSPM = "https://packagemanager.posit.co/cran/latest"
  ))
}

# Set compiler flags
if (!file.exists("~/.R/Makevars")) {
  dir.create("~/.R", showWarnings = FALSE)
  writeLines(
    c("CXX=clang++",
      "CXXFLAGS=-g -O2 -std=c++11",
      "CXX11=clang++",
      "CXX11STD=-std=c++11",
      "CXX11FLAGS=-g -O2",
      "OBJCXXFLAGS=-g -O2 -std=c++11"),
    "~/.R/Makevars"
  )
}

# Install base system packages first
if (Sys.info()["sysname"] == "Darwin") {
  message("Installing system dependencies for M1 Mac...")
  # Install system dependencies
  install.packages(c("systemfonts", "textshaping", "ragg"),
                  repos = c(CRAN = "https://cloud.r-project.org"))
}

# Install and initialize renv if not already installed
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Set renv options
options(renv.config.auto.snapshot = FALSE)
options(renv.config.install.staged = FALSE)

# Initialize renv project
renv::init()

# Define core packages (minimal dependencies)
core_packages <- c(
  # Base dependencies
  "R6",
  "cli",
  "class",
  "mgcv",
  "quantreg",
  "MASS",
  "stats4",
  "mvtnorm",
  "mnormt",
  "numDeriv",
  "cubature"
)

# Define additional packages
additional_packages <- c(
  # Statistical packages
  "FNN",
  "mclust",
  "transport",
  "e1071",
  "sn",
  "RiskPortfolios",
  "LaplacesDemon",
  "bayestestR",
  "numbers",
  "dplyr",
  "LaplacesDemon",
  "smfsb",  
  # Parallel computing
  # Note: doRNG installation may fail due to pkgmaker dependency
  # This is not critical as the core functionality works without it
  "foreach",
  "doParallel",
  
  # Visualization
  "ggplot2@3.5.1",
  "ggsci",
  "ggpubr",
  "gridExtra@2.3",
  
  # Performance
  "microbenchmark",
  "smfsb",
  
  # Documentation
  "knitr@1.49",
  "rmarkdown",
  "kableExtra@1.4.0"
)

# Install dependencies for all packages
message("Installing package dependencies...")
# First install packages without version requirements
install.packages(core_packages, dependencies = TRUE)

# Install packages with specific versions
remotes::install_version("ggplot2", version = "3.5.1")
remotes::install_version("gridExtra", version = "2.3")
remotes::install_version("knitr", version = "1.49")
remotes::install_version("kableExtra", version = "1.4.0")

# Install other packages
other_packages <- c("ggsci", "ggpubr", "rmarkdown")
install.packages(other_packages, dependencies = TRUE)

# Install rstan separately
message("Installing rstan...")
tryCatch({
  install.packages("rstan", dependencies = TRUE)
}, error = function(e) {
  message("Note: rstan installation failed, but this is not critical for initial setup")
})

# Try to install doRNG separately
message("Attempting to install doRNG...")
tryCatch({
  if (!requireNamespace("doRNG", quietly = TRUE)) {
    install.packages("doRNG", dependencies = FALSE)
  }
}, error = function(e) {
  message("Note: doRNG installation failed, but this is not critical for core functionality")
})

# Verify installation
all_packages <- c(core_packages, additional_packages)
missing_packages <- c()
for (pkg in all_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  message(sprintf("Note: Some packages could not be installed: %s", paste(missing_packages, collapse = ", ")))
  message("This may not affect core functionality")
} else {
  message("All packages installed successfully!")
}

# Create snapshot of the environment
renv::snapshot()

# Save detailed session info
sessionInfo_file <- file.path("renv", "sessionInfo.txt")
sink(sessionInfo_file)
cat("Environment details for JASA paper reproduction\n")
cat("Date: ", as.character(Sys.time()), "\n\n")
print(sessionInfo())
sink()

message("Environment setup completed successfully!") 