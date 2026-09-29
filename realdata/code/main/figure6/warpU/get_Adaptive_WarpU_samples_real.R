## ============================================================================
## Runs the adaptive Warp-U MCMC sampler on the exoplanet posterior and saves
## all sampling state for downstream use (Figure 6 right panel).
##
## Inputs (under realdata/data/):
##   dataset1_one_planet_posterior.RData  -- posterior setup (p1, data_now, ...)
##   prior_bounds.RData                   -- prior box bounds
##   stanData.RData                       -- Stan HMC samples used to set the
##                                           proposal scale sigma_I and star_w
##
## Output:
##   output/results/AdaptiveWarpU_Real.RData
##
## Pipeline role:
##   Step 2 of code/scripts/reproduce_realdata_analysis.sh. The resulting
##   samples are consumed by RVdata_illustration_plot_density.R to draw the
##   Warp-U MCMC density in Figure 6.
## ============================================================================

rm(list = ls())

set.seed(23)

library(FNN)
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(numDeriv)
library(ggplot2)
library(mclust)
library(transport)
library(e1071)
library(sn)
library(RiskPortfolios)
# source the function
density_fun_used = "real_exo"
restrict_prior_flag = FALSE

## source the GWL algorithm
if(density_fun_used == "skew_t"){
  source('code/functions/mcmc/GWLD_skewT.R')
}else if(density_fun_used == "real_exo"){
  load('data/dataset1_one_planet_posterior.RData')
  source("code/functions/model/log_like.R")
  source("code/functions/model/log_post.R")
  source("code/functions/model/cov_fun.R")
  source("code/functions/model/cov_make.R")
  load('data/prior_bounds.RData')
  source("code/functions/model/planet_model.R")
  if(restrict_prior_flag == TRUE){
    source("code/functions/model/restricted_priors.R")
  }else{
    source('code/functions/model/full_priors.R')
  }
}else{
  source('code/functions/mcmc/GWLD.R')
}
  
source('code/functions/warpu/Functions_AdaptiveWarpU.R')
if(density_fun_used == "skew_t"){
  source('code/functions/model/skew_t_density.R')
}



N = 2500#52000
last_stage_num_factor = 1
sigma_I_11_factor = 1
#
if(density_fun_used == "skew_t"){
  alpha = 0.5
  sigma2 = 2.38^2/(10)
  stage_num =20
  sample_range = rbind(rep(-20,10), rep(20,10))
  sigma_I = diag(10)
  com_num = 25
  star_w = c(-3.448158,-9.456676, 7.11798, -0.7514566, -4.918324, 8.630202, 2.897756, -2.959098, 6.310617, 7.785749)
}else if(density_fun_used == "real_exo"){
  com_num = 10
  stage_num = 4
  sigma2 = 2.38^2/(7)
  alpha1 = 0.5
  ###
  ###
  if(restrict_prior_flag == FALSE){
    Pmin <- rep(1.25,3)
    Pmax <- rep(10^4,3)
  }else{
    Pmax = c(44.6684, 12.8825, 10.7152)
    Pmin = c(39.8107, 11.4815, 10.0000)
  }
  num_planets <- 1
  dataset_num <- 1
  paras <- matrix(NA,2,7)
  paras[1,] <- c(C=-1,tau=42.4,K=0.1,e=0.1,w=2.0,M0=3.0,sigmaJ=2)
  paras[2,] <- c(C=-1,tau=42.4,K=0.1,e=0.1,w=2.0,M0=3.0,sigmaJ=3)
  p1(paras)
  ###
  ###
  sample_range = matrix(c(-2.2292,1.2552,41.67,42.49,1.832,3.381,0.007472,0.538570,0.006595,4.860418,0.000676,4.834212,1.057,1.788),2,7)
  load('data/stanData.RData')
  True_Data = data
  sigma_I_vec = c()
  for(i in 1:(dim(True_Data)[2])){
    sigma_I_vec = c(sigma_I_vec,var(True_Data[,i]))
  }
  sigma_I = diag(sigma_I_vec)
  sigma_I  = covEstimation(True_Data, control = list(type = 'ewma', lambda = 0.9))
  sigma_I  = diag(sigma_I)
  sigma_I = diag(sigma_I)
  star_w = True_Data[2,]
}

sigma_I[1,1] = sigma_I[1,1] * sigma_I_11_factor

inserversion_num = 23 # 12: warpU+gibbs_changes  13: for gibbs only, 10000: only for warpU, 25: MH(Norm+Unif)+warpU,24: MH(Norm+Unif) only # 23 adaptive warp U method


time_begin = proc.time()
density_num = 123


N_Rpeat = 1
results_sample_list = c()
# Run the adaptive Warp-U sampler
for( rep in 1:N_Rpeat){
  results_sample = adaptive_warpU(com_num = com_num,N = N,inserversion_num = inserversion_num, stage_num=stage_num, sample_range = sample_range,density_num = density_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I,star_w = star_w,last_stage_num_factor = last_stage_num_factor)
  results_sample_list = c(results_sample_list,list(results_sample))
}

time_used = proc.time() - time_begin
save(list = ls(),file ='output/results/AdaptiveWarpU_Real.RData')

