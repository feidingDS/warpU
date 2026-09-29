## ============================================================================
## Tunes the temperature ladder for the Parallel Tempering sampler on the
## exoplanet posterior and saves it for downstream PT runs.
##
## Inputs (under realdata/data/):
##   dataset1_one_planet_posterior.RData  -- posterior setup
##
## Output:
##   data/real_tmperature_get.RData       -- tempturature_get vector consumed
##                                           by get_sample_real_PL.R
## ============================================================================

rm(list = ls())
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
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
  source('code/functions/model/simple_case_density.R')
}
source('code/functions/mcmc/PL_functions_used.R')


#
if(density_fun_used == "skew_t"){
  #N = 100000
  N = (31800*2)
  Dim = 10
  num_level = 13
  #iters=5e5
  iters=(31800*2)
}else if(density_fun_used == "real_exo"){
  N = 20000
  Dim = 7
  num_level = 6
  iters=5e4
}else{
  N = 20000
  Dim = 6
  num_level = 6
  iters=5e4
}



special_factor = 1
iner_maximum = 10000
###
time_efficient = proc.time()
if(density_fun_used == "skew_t"){
  ## calculate the sampling efficiency fo PT
  True_data_simulated = rp1(1000000)
  Sigma_mat = covEstimation(True_data_simulated, control = list(type = 'ewma', lambda = 0.9))
  # Init_mean = meanEstimation(True_data_simulated, control = list(type = 'ewma',lambda = 0.9))
  Init_mean = True_data_simulated[1,]
}else if(density_fun_used == 'mixture_Normal'){
  True_Data = c()
  mean_true = c(-11,12,-8,7,-2)
  #mean_true = rbind(mean_true,mean_true)
  mean_true = t(matrix(mean_true,5,Dim))
  for(i in 1:40000){
    tmpsample1 = sample(x =1:5 ,size = 1,replace = TRUE,prob = c(1/15,2/15,3/15,4/15,5/15))
    tmpsample2 = mvrnorm(n = 1,mean_true[,tmpsample1],Sigma = diag(Dim))
    True_Data = rbind(True_Data,tmpsample2)
  }
  Sigma_mat = covEstimation(True_Data, control = list(type = 'ewma', lambda = 0.9))
  Init_mean = True_Data[1,]
}else if(density_fun_used == "real_exo"){
  load('data/stanData.RData')
  True_Data = data
  Sigma_mat = covEstimation(True_Data, control = list(type = 'ewma', lambda = 0.9))
  Init_mean = True_Data[1,]
}


# Select the temperature ladder
selected_scale = select_scale(initi_rho = -300,beta_min = 0.1, Sigma_mat = Sigma_mat,init = Init_mean)
time_efficient = proc.time() - time_efficient
Index_small_1 = which(selected_scale$beta_store < (1-1e-3) )
tempturature_get = c(1,1/selected_scale$beta_store[Index_small_1])
save(list = c('tempturature_get'), file = 'data/real_tmperature_get.RData')
