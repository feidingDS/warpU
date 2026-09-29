#!/bin/bash

# Set directory paths (relative to workspace root)
LOG_DIR="output/logs"
RESULTS_DIR="output/results"
FIGURES_DIR="output/figures"


# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

# Set log file
MAIN_LOG="$LOG_DIR/realdata_reproduction_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Error checking function
check_error() {
    if [ $? -ne 0 ]; then
        log "WARNING: $1 failed but continuing with next steps"
        return 1
    fi
    return 0
}

# Function to calculate runtime
get_runtime() {
    local start_time=$1
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))
    local hours=$((runtime / 3600))
    local minutes=$(( (runtime % 3600) / 60 ))
    local seconds=$((runtime % 60))
    echo "${hours}h ${minutes}m ${seconds}s"
}

# File existence check with timeout
check_file_with_timeout() {
    local file_path=$1
    local timeout=$2  # timeout in seconds
    local start_time=$(date +%s)
    local current_time
    
    while true; do
        if [ -f "$file_path" ]; then
            return 0
        fi
        
        current_time=$(date +%s)
        if [ $((current_time - start_time)) -gt "$timeout" ]; then
            local runtime=$(get_runtime $start_time)
            log "WARNING: Timeout after ${runtime} waiting for file $file_path - the process of this step is still running, continuing with next steps"
            return 1
        fi
        
        sleep 10  # Check every 10 seconds
    done
}

# Function to run R script in renv environment
run_r_script() {
    local script_path=$1
    local log_file=$2
    local error_message=$3
    local output_file=$4
    local timeout=${5:-3600}  # Default timeout 1 hour
    
    local start_time=$(date +%s)
    log "Starting: $error_message"
    
    Rscript -e "renv::run('$script_path')" > "$log_file" 2>&1 &
    local r_pid=$!
    
    # Wait for R process to complete
    wait $r_pid
    if ! check_error "$error_message"; then
        local runtime=$(get_runtime $start_time)
        log "Process ran for ${runtime} before failing"
        return 1
    fi
    
    # If output file is specified, wait for it
    if [ ! -z "$output_file" ]; then
        log "Waiting for output file: $output_file"
        if ! check_file_with_timeout "$output_file" "$timeout"; then
            local runtime=$(get_runtime $start_time)
            log "Process ran for ${runtime} but output file was not generated"
            return 1
        fi
        log "Output file generated: $output_file"
    fi
    
    local runtime=$(get_runtime $start_time)
    log "Successfully completed in ${runtime}"
    return 0
}

# Start execution
log "Starting real data analysis reproduction..."

# Track overall success
OVERALL_SUCCESS=true
OVERALL_START_TIME=$(date +%s)

# 1. Initialize R environment
log "Step 1: Initializing R environment"
Rscript -e "renv::restore()" > "$LOG_DIR/renv_restore.log" 2>&1
if ! check_error "R environment initialization"; then
    OVERALL_SUCCESS=false
fi

# 2. Generate Warp-U MCMC samples
log "Step 2: Generating Warp-U MCMC samples"
if ! run_r_script "code/main/figure6/warpU/get_Adaptive_WarpU_samples_real.R" \
            "$LOG_DIR/get_Adaptive_WarpU_samples_real.log" \
            "Warp-U MCMC sample generation" \
            "$RESULTS_DIR/AdaptiveWarpU_Real.RData" \
            36000; then  # 10 hours timeout
    OVERALL_SUCCESS=false
fi

# 3. Parallel Tempering (PT)
log "Step 3: Running Parallel Tempering"
# Check if temperature tuning is needed
if [ ! -f "data/real_tmperature_get.RData" ]; then
    log "Starting temperature ladder tuning (estimated >15 hours)..."
    if ! run_r_script "code/main/figure6/PT/tune_PL_real.R" \
                "$LOG_DIR/tune_PL_real.log" \
                "Temperature ladder tuning" \
                "data/real_tmperature_get.RData" \
                54000; then  # 15 hours timeout
        OVERALL_SUCCESS=false
    fi
else
    log "Using existing temperature ladder data"
fi



log "Running PT sampling..."
if ! run_r_script "code/main/figure6/PT/get_sample_real_PL.R" \
            "$LOG_DIR/get_sample_real_PL.log" \
            "PT sampling" \
            "$RESULTS_DIR/PL_samples_Real.RData" \
            36000; then  # 10 hours timeout
    OVERALL_SUCCESS=false
fi

# 4. Approximate target density
log "Step 4: Approximating target density"
if ! run_r_script "code/main/figure6/Integral_results.R" \
            "$LOG_DIR/Integral_results.log" \
            "Target density approximation" \
            "$RESULTS_DIR/int_results_gridfrom_Real.RData" \
            90000; then  # 25 hours timeout
    OVERALL_SUCCESS=false
fi

# 5. Generate RV illustration and density plots
log "Step 5: Generating RV illustration and density plots"
if ! run_r_script "code/main/figure6/RVdata_illustration_plot_density.R" \
            "$LOG_DIR/RVdata_illustration_plot_density.log" \
            "RV illustration and density plot generation" \
            "$FIGURES_DIR/p_density_final.pdf" \
            1800; then  # 30 minutes timeout
    OVERALL_SUCCESS=false
fi

# 6. Normalizing constant estimation (parallel computation)
log "Step 6: Computing normalizing constant estimates"

# Warp-U parallel computation
log "Running Warp-U parallel computation..."
warp_u_start_time=$(date +%s)
bash code/scripts/run_parallel_warpU.sh > "$LOG_DIR/output_warpU_real.log" 2>&1
runtime=$(get_runtime $warp_u_start_time)
log "Warp-U parallel computation finished in ${runtime}"

# PT parallel computation
log "Running PT parallel computation..."
pt_start_time=$(date +%s)
bash code/scripts/run_parallel_PT.sh > "$LOG_DIR/output_PT_real.log" 2>&1
runtime=$(get_runtime $pt_start_time)
log "PT parallel computation finished in ${runtime}"

# Wait for parallel computation results
log "Waiting for parallel computation results..."
while true; do
    if [ -f "$RESULTS_DIR/warpU_samples_Real_cost_large_Comparison_combined.RData" ] && \
       [ -f "$RESULTS_DIR/PT_samples_Real_cost_large_Comparison_combined.RData" ]; then
        log "All parallel computation results are ready"
        break
    fi
    log "Still waiting for parallel computation results... (checking every 5 minutes)"
    sleep 300  # Wait for 5 minutes before checking again
done

# 7. Generate RMSE summary
log "Step 7: Generating RMSE summary"
if ! run_r_script "code/main/figure6/Comparison_estimation_real_results_summary.R" \
            "$LOG_DIR/Comparison_estimation_real_results_summary.log" \
            "RMSE summary generation"; then
    OVERALL_SUCCESS=false
fi

# Final status report
TOTAL_RUNTIME=$(get_runtime $OVERALL_START_TIME)
log "Real data analysis reproduction completed in ${TOTAL_RUNTIME}"
if [ "$OVERALL_SUCCESS" = true ]; then
    log "All steps completed successfully!"
else
    log "Some steps failed or timed out - check logs for details"
fi
log "All logs are saved in: $LOG_DIR"
log "Results are saved in: $RESULTS_DIR"
log "Figures are saved in: $FIGURES_DIR" 