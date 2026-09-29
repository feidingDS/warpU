## ============================================================================
## Combines the 50 per-repeat Warp-U result files written by
## Comparison_estimation_real_results_warpU_parallel.R into a single RData
## file consumed by Comparison_estimation_real_results_summary.R (Table 2).
##
## Input:
##   output/results/warpU_results_*.RData
##
## Output:
##   output/results/warpU_samples_Real_cost_large_Comparison_combined.RData
## ============================================================================

RMSE_orig_mat = NULL
RMSE_WarpBs_mat = NULL
RMSE_StochBs_mat = NULL
matrix_orig = NULL
matrix_Warp = NULL
matrix_Stoc = NULL

# Collect per-repeat result files
result_files = list.files(path = "output/results",
                         pattern = "warpU_results_.*\\.RData",
                         full.names = TRUE)

# Stack per-repeat squared errors and estimates across all repeats
for(file in result_files) {
    load(file)
    RMSE_orig_mat = rbind(RMSE_orig_mat, RMSE_orig)
    RMSE_WarpBs_mat = rbind(RMSE_WarpBs_mat, RMSE_WarpBs)
    RMSE_StochBs_mat = rbind(RMSE_StochBs_mat, RMSE_StochBs)

    if(exists("Est_orig")) {
        matrix_orig = rbind(matrix_orig, Est_orig)
        matrix_Warp = rbind(matrix_Warp, Est_WarpBs)
        matrix_Stoc = rbind(matrix_Stoc, Est_StochBs)
    }
}

# Aggregate across repeats: RMSE (*_mat1) and SD (*_mat2)
RMSE_orig_mat1 = sqrt(colMeans(RMSE_orig_mat))
RMSE_WarpBs_mat1 = sqrt(colMeans(RMSE_WarpBs_mat))
RMSE_StochBs_mat1 = sqrt(colMeans(RMSE_StochBs_mat))
RMSE_orig_mat2 = apply(RMSE_orig_mat, 2, sd)
RMSE_WarpBs_mat2 = apply(RMSE_WarpBs_mat, 2, sd)
RMSE_StochBs_mat2 = apply(RMSE_StochBs_mat, 2, sd)

save(list = ls(),
     file = 'output/results/warpU_samples_Real_cost_large_Comparison_combined.RData')
