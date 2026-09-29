## ============================================================================
## Generates the RMSE numbers reported in Table 2 of the paper.
##
## For each of the two sampler families (Parallel Tempering and Warp-U MCMC),
## the script loads the output of 50 repeated runs and computes the
## RMSE (with standard error) of three log-evidence estimators:
##   BS  -- Bridge Sampling       (RMSE_orig_*)
##   WB  -- Warp Bridge Sampling  (RMSE_WarpBs_*)
##   SWB -- Stochastic Warp Bridge Sampling (RMSE_StochBs_*)
##
## Inputs (under output/results/):
##   PT_samples_Real_cost_large_Comparison_combined.RData
##   warpU_samples_Real_cost_large_Comparison_combined.RData
## Both files are produced by code/scripts/run_parallel_PT.sh and
## code/scripts/run_parallel_warpU.sh respectively.
##
## Output:
##   RMSE_data_all is printed to console / log; the values are then displayed
##   in realdata/manuscript/reproduce_results.html (Table 2).
## ============================================================================

rm(list = ls())
# Libraries
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(ggplot2)
library(ggpubr)


num_level = 1
time_num_used = 1
## source the related functions
density_fun_used = "real_exo"
narrow_flag = FALSE
density_num = 123
set.seed(23)
## source the GWL algorithm
if(density_fun_used == "skew_t"){
  source('code/functions/mcmc/GWLD_skewT.R')
}else{
  source('code/functions/mcmc/GWLD.R')
}
source('code/functions/model/Stochastics_functions.R')
source('code/functions_used/warpu code/EMhd_diagonal.R')
source('code/functions_used/warpu code/Bridge_Sampling_funs_eff.R') 
if(density_fun_used == "skew_t"){
  source('code/functions/model/skew_t_density.R')
}
if(density_fun_used == "skew_t"){
  true_Z = 1
}else if(density_fun_used == 'mixture_Normal'){
  true_Z = 2*pi
}else if(density_fun_used == 'real_exo'){
  true_Z = -193.71
}
RepeatNum = 50
### suppose we have already got the GWL samples.
K_prop = com_num = 10
if(density_fun_used == "skew_t"){
  Dim = 10
}else if(density_fun_used == 'mixture_Normal'){
  Dim = 2
  p1 = function(x,log=F){
    nn = dim(x)[1]
    return_results = c()
    for (i in 1:nn) {
      return_results = c(return_results, q_density(x[i,]))
      
    }
   
    if(log){
      return(log(return_results))
    }else{
      return(return_results)
    }
  }
}else if(density_fun_used == 'real_exo'){
  Dim = 7
  load('data/dataset1_one_planet_posterior.RData')
  if(narrow_flag == FALSE){
    source('code/functions/model/full_priors.R')
    Pmin <- rep(1.25,3)
    Pmax <- rep(10^4,3)
  }
  source("code/functions/model/log_like.R")
  source("code/functions/model/log_post.R")
  source("code/functions/model/cov_fun.R")
  source("code/functions/model/cov_make.R")
  source("code/functions/model/planet_model.R")
  source('code/functions/warpu/Functions_AdaptiveWarpU.R')
  true_Z = -193.71
  matrix_orig = c()
  matrix_Warp = c()
  matrix_Stoc = c()
}

p1


x_seq = c(0)

# --- Section 1: Parallel Tempering results ---------------------------------
load('output/results/PT_samples_Real_cost_large_Comparison_combined.RData')
RepeatNum = 50
RMSE_orig_data = data.frame(x_seq = x_seq, RMSE =  RMSE_orig_mat1, se = RMSE_orig_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2)))
RMSE_WarpBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_WarpBs_mat1, se = RMSE_WarpBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2)))
RMSE_StochBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_StochBs_mat1, se = RMSE_StochBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2)))
RMSE_orig_data$curveID = 'BS'
RMSE_WarpBs_data$curveID = 'WB'
RMSE_StochBs_data$curveID = 'SWB'
RMSE_data_all = rbind(RMSE_orig_data,RMSE_WarpBs_data,RMSE_StochBs_data)




RMSE_data_all$curveID <- factor(RMSE_data_all$curveID , levels = c("BS", "WB", "SWB"))
RMSE_data_all

# --- Section 2: Warp-U MCMC results ----------------------------------------
load('output/results/warpU_samples_Real_cost_large_Comparison_combined.RData')
RepeatNum = 50
RMSE_orig_data = data.frame(x_seq = x_seq, RMSE =  RMSE_orig_mat1, se = RMSE_orig_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2)))
RMSE_WarpBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_WarpBs_mat1, se = RMSE_WarpBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2)))
RMSE_StochBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_StochBs_mat1, se = RMSE_StochBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2)))
RMSE_orig_data$curveID = 'BS'
RMSE_WarpBs_data$curveID = 'WB'
RMSE_StochBs_data$curveID = 'SWB'
RMSE_data_all = rbind(RMSE_orig_data,RMSE_WarpBs_data,RMSE_StochBs_data)




RMSE_data_all$curveID <- factor(RMSE_data_all$curveID , levels = c("BS", "WB", "SWB"))
RMSE_data_all
