rm(list = ls())
library(doRNG)
library(foreach)
library(doParallel)
##
# our method
#####################################################
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
#library(bridgesampling)
# source the function
# source('./functionused/Functionsexo_new.R')
# source('./functionused/simple_case_density.R')
## skew_t correnspond to density number 123
density_fun_used = 'skew_t'
density_num = 'skewt'
#density_fun_used = 'mixture_Normal'
#density_num = 'mixture_Normal'
# density_fun_used = "real_exo"
# density_num = "real_exo"
density_num = 123
set.seed(23)
if(density_fun_used == "skew_t"){
  source('./code/functions/mcmc/GWLD_skewT.R')
}else if(density_fun_used == "real_exo"){
  load('./data/raw/dataset1_one_planet_posterior.RData')
  source("./code/functions/model/log_like.R")
  source("./code/functions/model/log_post.R")
  source("./code/functions/model/cov_fun.R")
  source("./code/functions/model/cov_make.R")
  load('./data/raw/prior_bounds.RData')
  source("./code/functions/model/planet_model.R") # Added in sqrt to phi
  if(restrict_prior_flag == TRUE){
    source("./code/functions/model/restricted_priors.R")
  }else{
    source('./code/functions/model/full_priors.R')
  }
}else{
  source('./code/functions/mcmc/GWLD.R')
}
source('./code/functions/warpu/Functions_AdaptiveWarpU.R')
if(density_fun_used == "skew_t"){
  source('./code/functions/model/skew_t_density.R')
}
source('./code/functions/warpu/PL_functions_used.R')


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
  load('stanData')
  True_Data = data
  Sigma_mat = covEstimation(True_Data, control = list(type = 'ewma', lambda = 0.9))
  Init_mean = True_Data[1,]
}


##get the scale
selected_scale = select_scale(initi_rho = -300,beta_min = 0.1, Sigma_mat = Sigma_mat,init = Init_mean)
time_efficient = proc.time() - time_efficient
Index_small_1 = which(selected_scale$beta_store < (1-1e-3) )
tempturature_get = c(1,1/selected_scale$beta_store[Index_small_1])
save(list = c('tempturature_get'), file = './output/results/PT/skewT_temperature_get.RData')
# 
# 
# ########
# ##
# print('BEGIN')
# Sample_efficiency = c()
# Sample_accept_rate = c()
# for (num_level in 6) {
#   print('num_level')
#   print(num_level)
#   temps <- 10^( -2 * ((1:num_level) -1)/(num_level-1))
#   temps = (1- max( log10(10^( -1* ((1:num_level) -1)/(num_level-1)) + 1) )) + log10(10^( -1* ((1:num_level) -1)/(num_level-1)) + 1)
#   temps = (-0.8^(1:num_level))
#   temps = (temps - min(temps))/(max(temps) - min(temps))*(0.9) + 0.1
#   temps = sort(temps,decreasing = TRUE)
#   iters=2e3
#   Sample_efficiency_tmp = chains_parallel_tempering(MSD_flag = 1,effective_sigma_flag = 1, Sigma_mat = Sigma_mat, init = Init_mean)
#   Sample_efficiency = c(Sample_efficiency,Sample_efficiency_tmp$swapefficient)
#   Sample_accept_rate = rbind(Sample_accept_rate,Sample_efficiency_tmp$accept_rate)
# }
# time_efficient = proc.time() - time_efficient
# #op = par(mfrow = c(1,1))
# 
# 

####
#plot(x = 2:20, y = Sample_efficiency, type = 'o', xlab = 'Number of the Temperature LeveLs',ylab = 'MSD Jumps of Cold Chain (1/T)',main = 'Sampling Efficiency of Parallel Tempering')
#save(list = ls(), file =  paste0('./Results_stored/check_eff_Dec2_special_factor',special_factor,'iner_maximum',iner_maximum,'density_fun_used_',density_fun_used))
