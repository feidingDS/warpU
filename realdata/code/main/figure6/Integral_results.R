## ============================================================================
## Computes a numerical-integration approximation of the posterior
## of the mean anomaly parameter, evaluated on a grid.
##
## For each grid value of the mean anomaly, hcubature integrates the joint
## posterior over the remaining 7 parameters; the resulting curve is the
## "Numerical Integral" reference plotted in Figure 6 (right panel).
##
## Inputs (under realdata/data/):
##   one_planets_full_prior_fit_only_3chains_3000grid_3000iters.RData
##                                      -- Stan fit used to derive integration
##                                         ranges
##   dataset1_one_planet_posterior.RData -- posterior setup (log_post, data_now)
##
## Output:
##   output/results/int_results_gridfrom_Real.RData
## ============================================================================

rm(list = ls())
# load the Stan samples
load('data/one_planets_full_prior_fit_only_3chains_3000grid_3000iters.RData')
load('data/dataset1_one_planet_posterior.RData')

# library the packages
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(numDeriv)
library(ggplot2)
library(mclust)
library(foreach)
library(doParallel)
set.seed(23)

# source the function
source("code/functions/model/log_like.R")
source("code/functions/model/log_post.R")
source("code/functions/model/cov_fun.R")
source("code/functions/model/cov_make.R")
source("code/functions/model/planet_model.R") # Added in sqrt to phi
source("code/functions_used/warpu code/EMhd_diagonal.R")
source("code/functions_used/warpu code/Bridge_Sampling_funs_eff.R")

fit_mat <- as.matrix(fit)
simulation_sample_size = dim(fit_mat)[1]
n = simulation_sample_size
data = fit_mat[,1:dim(fit_mat)[2]-1]
for (i in c(3,7)){
  data[,i] <- exp(data[,i])-1
}
i <- 2
data[,i] <- exp(data[,i])
colnames(data) <- c("C","P","K","e","w","M0","sigma")
Stansample = data

# set the variabel to compare. When variable_num = 2, it represent "P" etc.
variable_num = 6

ranges <- apply(data,2,range)
widths <- abs(apply(ranges,2,diff))
width_factor <- 0

# Joint posterior with the mean anomaly
int_q = function(v_w){
  planet_paras <- list()
  if(num_planets == 0){
    planet_paras=c()
  }else{
    for (tmpnum in 1:num_planets) {
      planet_paras[[tmpnum]] <- list(tau=v_w[2],K=v_w[3],e=v_w[4],w=v_w[5],M0=MM,gamma=0)
    }
    for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=v_w[1],sigmaJ=v_w[7],
                     cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=planet_paras)
  }
  return(exp(log_post(for_post)))
}
###
library(cubature)
time_int_start = proc.time()
print('begin')
grid_num = 200#100
grid_from = -0.4
grid_to = 5.1
tolorence_int = 0.05  #0.2
grid_search = seq(grid_from,grid_to,length.out = grid_num)
Repeat_num = 1
int_out_matrix = c()
Parallel_flag = FALSE
if(Parallel_flag == TRUE){
  cl <- makeCluster(10,type="FORK")
  registerDoParallel(cl)
  result_foreach = foreach(maininter = 1:Repeat_num)%dopar%{
    int_out = c()
    int_q = function(v_w){
      planet_paras <- list()
      if(num_planets == 0){
        planet_paras=c()
      }else{
        for (tmpnum in 1:num_planets) {
          planet_paras[[tmpnum]] <- list(tau=v_w[2],K=v_w[3],e=v_w[4],w=v_w[5],M0=MM,gamma=0)
        }
        for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=v_w[1],sigmaJ=v_w[7],
                         cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=planet_paras)
      }
      return(exp(log_post(for_post)))
    }
    for(MM in grid_search ){
      numerical_int <- hcubature(int_q, lowerLimit=ranges[1,]-width_factor*widths, upperLimit=ranges[2,]+width_factor*widths,tol = tolorence_int)
      int_out = c(int_out,numerical_int$integral/460)
      int_out
    }
    int_out
  }
  stopCluster(cl)
}else{
  # Sequential: evaluate the integral
  print('flag_FALSE')
  for (ret_i in 1:Repeat_num) {
    int_out = c()
    for(MM in grid_search ){
      print('MM')
      print(MM)
      numerical_int <- hcubature(int_q, lowerLimit=ranges[1,]-width_factor*widths, upperLimit=ranges[2,]+width_factor*widths,tol = tolorence_int)
      int_out = c(int_out,numerical_int$integral/460)
    }
    int_out_matrix = rbind(int_out_matrix, int_out)
  }
}

time_int_used_int = proc.time() - time_int_start

save(list = ls(),file ='output/results/int_results_gridfrom_Real.RData')
