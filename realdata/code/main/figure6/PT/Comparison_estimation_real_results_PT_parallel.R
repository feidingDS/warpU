## ============================================================================
## Worker script for Monte-Carlo of the log-evidence comparison,
## using samples produced by Parallel Tempering.
##
## This script computes three
## estimators of the log-evidence and records their squared error vs the
## quasi-true value true_Z:
##   BS  -- Bridge Sampling       (Est_orig / RMSE_orig)
##   WB  -- Warp Bridge Sampling  (Est_WarpBs / RMSE_WarpBs)
##   SWB -- Stochastic Warp Bridge Sampling (Est_StochBs / RMSE_StochBs)
##
## Inputs:
##   data/PL_samples_Real.RData            -- PT samples
##   data/dataset1_one_planet_posterior.RData
##
## Output:
##   output/results/PT_results_<current_repeat>.RData
##   (later combined by code/main/figure6/PT/merge_results_PT.R)
## ============================================================================

cat("Processing repeat number:", current_repeat, "\n")
set.seed(123)
# Libraries
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(ggplot2)

num_level = 1

# load data
load('data/PL_samples_Real.RData')

# method settings
time_num_used = 1
method_used = 'PL'
density_fun_used = 'real_exo'
narrow_flag = FALSE
density_num = 123

# source related functions
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
  Dim = 10
}else if(density_fun_used == 'mixture_Normal'){
  true_Z = 2*pi
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
  true_Z = -193.71
  Dim = 7
  load('data/dataset1_one_planet_posterior.RData')
  if(narrow_flag == FALSE){
    source('code/functions/model/full_priors.R')
    Nndex = 1:1000
    Pmin <- rep(1.25,3)
    Pmax <- rep(10^4,3)
  }
  source("code/functions/model/log_like.R")
  source("code/functions/model/log_post.R")
  source("code/functions/model/cov_fun.R")
  source("code/functions/model/cov_make.R")
  source("code/functions/model/planet_model.R")
  source('code/functions/warpu/Functions_AdaptiveWarpU.R')
}

RepeatNum = 1
K_prop = com_num = 10

if(method_used == "PL"){
  sample_results = t(PL_list[[1]][ 1 , , Nndex])
}else if(method_used == 'Adaptive_warpU'){
  Index_set = seq(from = 1, to= N, by = 1)
  sample_results = results_sample$w_all[Index_set,]
}

if(method_used == 'gwl'){
  number_last_stage_target_evaluation = n_store[[11]]
}else if(method_used  == 'PL'){
  number_last_stage_target_evaluation = dim(sample_results)[1]
  sampleNum_by_GWL = dim(sample_results)[1]
}else if (method_used == 'Adaptive_warpU'){
  number_last_stage_target_evaluation = dim(sample_results)[1]
  sampleNum_by_GWL = dim(sample_results)[1]
}

if(density_fun_used == 'real_exo'){
  Cost_est = 47500
  N1_in = number_last_stage_target_evaluation
  number_last_stage_target_evaluation = Cost_est
}

RMSE_orig = c()
RMSE_WarpBs = c()
RMSE_StochBs = c()
Est_orig = c()
Est_WarpBs = c()
Est_StochBs = c()
set.seed(current_repeat + 23)
scale_grid = c(0)
number_grid = floor((10^scale_grid * number_last_stage_target_evaluation)/(2*com_num-1))

time_com_begin = proc.time()

for (num_sample in (2*number_grid)) {
  sample_used = num_sample
  print('timebegin')
  time_beggin = proc.time()
  print(time_beggin)
  
  if(method_used == 'gwl'){
    all_GWLSample_used = GetSamples_from_LastStage(sample_size = sample_used, 
                                                  estimate_result_GWL = est_result,
                                                  david_data = 1)
  }else{
    index_used = c(1:sampleNum_by_GWL,1:sampleNum_by_GWL)
    all_GWLSample_used = sample_results[index_used,]
  }
  
  print('timeused')
  print(proc.time()-time_beggin)
  
  # divide the data into two parts
  fit_mat = all_GWLSample_used
  simulation_sample_size = dim(fit_mat)[1] 
  n = simulation_sample_size
  first_half = 1:floor(n/2)
  second_half = setdiff(1:n,first_half)
  data = fit_mat
  data_p1_first_half = data[first_half,]
  data_p1_second_half = data[second_half,]
  
  # First run EM
  sample_size = min(K_prop*500,floor(simulation_sample_size/2))
  index1 = sample(x=first_half,size=sample_size)
  x = data[index1,]
  tem_em = EM_rep(x,K_prop,reps=1,niters=1000)
  total_iter_runs <- tem_em$EM_lists[[1]]$iter+tem_em$EM_lists[[2]]$iter
  proposal_evaluations <- sample_size*2*K_prop*(total_iter_runs-1)
  prop_prop = tem_em$weight
  valid_index = which(prop_prop!=0)
  
  # get the parameters from EM 
  prop_prop = prop_prop[valid_index]
  K_prop_original <- K_prop
  K_prop = length(prop_prop)
  mu_prop = matrix(matrix(tem_em$mu,K_prop_original,Dim)[valid_index,],K_prop,Dim)
  sigma2_prop = matrix(matrix(tem_em$sigma2,K_prop_original,Dim)[valid_index,],K_prop,Dim)
  
  if(density_fun_used == 'real_exo'){
    data_p1_second_half = sample_results
    data_tp1 = rtp1(sample_results)
  }else{
    data_tp1 = rtp1(data_p1_second_half)
  }
  
  proposal_evaluations <- proposal_evaluations+dim(data_p1_second_half)[1]*K_prop*(K_prop+1)
  
  if(density_fun_used == 'real_exo'){
    if(method_used == 'Adaptive_warpU'){
      N2 = floor(n/2*(2*com_num-1)/com_num)
    }else{
      N2 = floor((n/2*(2*com_num-1) - (com_num-1)*N1_in)/com_num)
    }
  }else{
    if(method_used == 'Adaptive_warpU'){
      N2 = floor(n/2*(2*com_num-1)/com_num)
    }else{
      N2 = floor(n/2)
    }
  }
  
  data_normal = rmvnorm(N2,mean=rep(0,Dim),sigma=diag(1,Dim))
  
  proposal_evaluations <- proposal_evaluations+(dim(data_p1_second_half)[1]+N2)*K_prop^2 
  target_evaluations <- (dim(data_p1_second_half)[1]+N2)*K_prop
  standard_normal_evaluations <- (dim(data_p1_second_half)[1]+N2)*2
  
  # BS estimator
  data_phimix = rphi_mix(floor(n/2*(2*com_num-1)))
  log_bs_Z_hat_original_Bs <- Opt_BS_hd(x1=data_p1_second_half,
                                       x2=data_phimix, 
                                       q1=p1,
                                       q2=phi_mix,
                                       r0=10)
  
  print('this is the estimator of original Bridge method ')
  exp(log_bs_Z_hat_original_Bs)
  
  if(density_fun_used == 'real_exo'){
    RMSE_orig = c(RMSE_orig, ((log10(exp(log_bs_Z_hat_original_Bs)) - true_Z)^2))
    Est_orig = c(Est_orig, log10(exp(log_bs_Z_hat_original_Bs)))
  }else{
    RMSE_orig = c(RMSE_orig, ((exp(log_bs_Z_hat_original_Bs) - true_Z)^2))
  }
  
  # WB estimator
  log_bs_Z_hat_WarpU_Bs <- Opt_BS_hd(x1=data_tp1,
                                     x2=data_normal, 
                                     q1=tp1,
                                     q2=p2_warp0,
                                     r0=10)
  
  print('this is the estimator of WarpU Bridge method ')
  exp(log_bs_Z_hat_WarpU_Bs)
  
  if(density_fun_used == 'real_exo'){
    RMSE_WarpBs = c(RMSE_WarpBs, ((log10(exp(log_bs_Z_hat_WarpU_Bs)) - true_Z)^2))
    Est_WarpBs = c(Est_WarpBs, log10(exp(log_bs_Z_hat_WarpU_Bs)))
  }else{
    RMSE_WarpBs = c(RMSE_WarpBs, ((exp(log_bs_Z_hat_WarpU_Bs) - true_Z)^2))
  }
  
  # SWB estimator
  if(density_fun_used == 'real_exo'){
    Opt_Stochas_BS_splitNorm_same = function(w_in){
      tildeW_k = rtp1_split(w_in)
      split_norm = list()
      length(split_norm) = K_prop
      for (kk in 1:K_prop) {
        if(ceiling(length(tildeW_k[[kk]])) > 0){
          split_norm[[kk]] = rmvnorm((floor(Cost_est/N1_in))*ceiling(length(tildeW_k[[kk]])),
                                    mean=rep(0,Dim),
                                    sigma=diag(1,Dim))
        }
      }
      return(Opt_BS_hd_Sto_splitNorm(split_data = tildeW_k,split_norm = split_norm))
    }
  }else{
    Opt_Stochas_BS_splitNorm_same = function(w_in){
      tildeW_k = rtp1_split(w_in)
      split_norm = list()
      length(split_norm) = K_prop
      for (kk in 1:K_prop) {
        if(ceiling(length(tildeW_k[[kk]])) > 0){
          split_norm[[kk]] = rmvnorm(((2*com_num-1)*ceiling(length(tildeW_k[[kk]]))),
                                    mean=rep(0,Dim),
                                    sigma=diag(1,Dim))
        }
      }
      return(Opt_BS_hd_Sto_splitNorm(split_data = tildeW_k,split_norm = split_norm))
    }
  }

  for(try_num in 1:100){
    try({
      bs_Z_hat_StochasticWarp_Bs = Opt_Stochas_BS_splitNorm_same(data_p1_second_half)
      break
    }, silent = FALSE)
  }
  
  if (try_num == 100){
    if(density_fun_used == 'real_exo'){
      Opt_Stochas_BS_splitNorm_same = function(w_in){
        tildeW_k = rtp1_split(w_in)
        split_norm = list()
        length(split_norm) = K_prop
        for (kk in 1:K_prop) {
          if(ceiling(length(tildeW_k[[kk]])) > 0){
            split_norm[[kk]] = rmvnorm((floor(Cost_est/N1_in))*ceiling(length(tildeW_k[[kk]])),
                                      mean=rep(0,Dim),
                                      sigma=diag(1,Dim))
          }
        }
        return(Opt_BS_hd_Sto_splitNorm(split_data = tildeW_k,split_norm = split_norm))
      }
    }else{
      Opt_Stochas_BS_splitNorm_same = function(w_in){
        tildeW_k = rtp1_split(w_in)
        split_norm = list()
        length(split_norm) = K_prop
        for (kk in 1:K_prop) {
          if(ceiling(length(tildeW_k[[kk]])) > 0){
            split_norm[[kk]] = rmvnorm(((2*com_num-1)*ceiling(length(tildeW_k[[kk]]))),
                                      mean=rep(0,Dim),
                                      sigma=diag(1,Dim))
          }
        }
        return(Opt_BS_hd_Sto_splitNorm_Error(split_data = tildeW_k,split_norm = split_norm))
      }
    }
    
    bs_Z_hat_StochasticWarp_Bs = Opt_Stochas_BS_splitNorm_same(data_p1_second_half)
    if(density_fun_used == 'real_exo'){
      RMSE_StochBs = c(RMSE_StochBs, ((log10((bs_Z_hat_StochasticWarp_Bs)) - true_Z)^2))  
      Est_StochBs = c(Est_StochBs,log10((bs_Z_hat_StochasticWarp_Bs)))
    }else{
      RMSE_StochBs = c(RMSE_StochBs, (((bs_Z_hat_StochasticWarp_Bs) - true_Z)^2))  
    }
  }else{ 
    print('this is the estimator of stochastics WarpU Bridge method ')
    if(density_fun_used == 'real_exo'){
      RMSE_StochBs = c(RMSE_StochBs, ((log10((bs_Z_hat_StochasticWarp_Bs)) - true_Z)^2))  
      Est_StochBs = c(Est_StochBs,log10((bs_Z_hat_StochasticWarp_Bs)))
    }else{
      RMSE_StochBs = c(RMSE_StochBs, (((bs_Z_hat_StochasticWarp_Bs) - true_Z)^2))  
    }
  }
}

time_com_used = proc.time() - time_com_begin 

# Save results for this repeat
save(list = c("RMSE_orig", "RMSE_WarpBs", "RMSE_StochBs", "Est_orig", "Est_WarpBs", "Est_StochBs"), 
     file = paste0('output/results/PT_results_', current_repeat, '.RData'))