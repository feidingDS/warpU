## ============================================================================
## Runs Parallel Tempering on the exoplanet posterior using the pre-tuned
## temperature ladder, and saves the samples for downstream comparisons
## (Figure 6 right panel and Table 2).
##
## Inputs (under realdata/data/):
##   dataset1_one_planet_posterior.RData  -- posterior setup
##   real_tmperature_get.RData            -- temperature ladder from
##                                           tune_PL_real.R
##
## Output:
##   output/results/PL_samples_Real.RData
## ============================================================================

rm(list = ls())
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(smfsb)
library(ggplot2)
library(LaplacesDemon)
library(ggsci)
library(microbenchmark)
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
library(RiskPortfolios)
density_fun_used = "real_exo"
density_num = "real_exo"
density_num = 123
set.seed(23)
if(density_fun_used == "skew_t"){
  source('code/functions/mcmc/GWLD_skewT.R')
  source('code/functions/model/skew_t_density.R')
}else if(density_fun_used == "real_exo"){
  load('data/dataset1_one_planet_posterior.RData')
  source("code/functions/model/log_like.R")
  source("code/functions/model/log_post.R")
  source("code/functions/model/cov_fun.R")
  source("code/functions/model/cov_make.R")
  source("code/functions/model/planet_model.R")
  source('code/functions/warpu/Functions_AdaptiveWarpU.R')
}else{
  source('code/functions/mcmc/GWLD.R')
}
source('code/functions/mcmc/PL_functions_used.R')


#
if(density_fun_used == "skew_t"){
  Dim = 10
  num_level = 8
  iters=  N = (31800*2)
}else if(density_fun_used == "real_exo"){
  N =2e4
  Dim = 7
  num_level = 7
  iters=2500#2e4
}




time_begin_PL = proc.time()
nrep = 1
set.seed(223)
PL_list = list()
length(PL_list) = nrep
# Load the pre-tuned temperature ladder
if(density_fun_used == "real_exo"){
  load('data/real_tmperature_get.RData')
  temps = 1/tempturature_get
}else if (density_fun_used == 'skew_t'){
  load('output/results/skewed_t_tmperature_get.RData')
  temps = 1/tempturature_get
}

special_factor = 2.38^2/(7)
element_factor = 1
num_level = length(temps)
for (i_in in 1:nrep) {
  if(density_fun_used == "skew_t"){
    True_data_simulated = rp1(1000000)
    Sigma_mat = covEstimation(True_data_simulated, control = list(type = 'ewma', lambda = 0.9))
    #Init_mean = meanEstimation(True_data_simulated, control = list(type = 'ewma',lambda = 0.9))
    Init_mean = True_data_simulated[1,]
  }else if(density_fun_used == "real_exo"){
    load('data/stanData.RData')
    True_Data = data
    Sigma_mat = covEstimation(True_Data, control = list(type = 'ewma', lambda = 0.9))
    Init_mean = True_Data[2,]
    Sigma_mat = diag(Sigma_mat)
    Sigma_mat = diag(Sigma_mat)
    Sigma_mat[1,1] = element_factor * Sigma_mat[1,1]
  }
  # Run PT
  sample_parallel = chains_parallel_tempering(effective_sigma_flag = 1, Sigma_mat = Sigma_mat, init = Init_mean)
  variable_num = 2
  sample_parallel_one_dim = matrix( 0,nrow = iters, ncol = length(temps))
  for (i in 1:iters) {
    sample_parallel_one_dim[ i, ] = sample_parallel[ ,variable_num ,i]
  }
  colnames(sample_parallel_one_dim)=paste("temps=",1/temps,sep="")
  PL_list[[i_in]] = sample_parallel
}
time_used_PL =proc.time()  - time_begin_PL
save(list = c("PL_list","time_used_PL", "num_level","temps"), file = 'output/results/PL_samples_Real.RData')

