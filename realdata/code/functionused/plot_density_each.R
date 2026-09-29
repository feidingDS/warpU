sample_pi_phi = function(dens,Stansample,our_sample,variable_num,sample_num){
  component_num = dens$G
  pi_vec = dens$parameters$pro
  mean_vec = dens$parameters$mean[variable_num,]
  sigma_vec = c()
  for (i in 1:component_num) {
    tmpsigma = dens$parameters$variance$sigma[,,i][variable_num,variable_num]
    sigma_vec = c(sigma_vec,tmpsigma)
  }
  sample_get = c()
  for(i in 1:sample_num){
    tmpsample1 = sample(x =1:component_num ,size = 1,replace = TRUE,prob = pi_vec)
    tmpsample2 = mvrnorm(n = 1,mean_vec[tmpsample1],sigma_vec[tmpsample1])
    sample_get = c(sample_get,tmpsample2)
  }
  sample_get = data.frame(sample = sample_get,ID = 'phi_mix')
  Stansample = data.frame(sample = Stansample[,variable_num],ID = 'Stan')
  our_sample = data.frame(sample = our_sample[,variable_num],ID = 'our_sample')
  data_sample = rbind(sample_get,Stansample,our_sample)
  print(data_sample)
  p1<-ggplot(data_sample, aes(x = sample,color = ID )) + geom_density()
  return(p1)
}

load('./sample_stored/new_sample_N_inversion12_gibbschange_+warpU')
our_sample =  sample_result$w_all
our_sample = our_sample[20000:30000,]
load('../Stan HMC output/one_planets_full_prior_fit_only_3chains_3000grid_3000iters.RData')
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(numDeriv)
library(ggplot2)
library(mclust)
library(bridgesampling)
# source the function
#source('./functionused/Functionsexo.R')
source("../R model code/log_like.R")
source("../R model code/log_post.R")
source("../R model code/cov_fun.R")
source("../R model code/cov_make.R")
source("../R model code/planet_model.R") # Added in sqrt to phi
source("../warpu code/EMhd_diagonal.R")
source("../warpu code/Bridge_Sampling_funs_eff.R")

# Given values
# tau <- 20 # days (stellar rotation period)
# alpha <- sqrt(3) # m/s
# lambda_e <- 50 # days
# lambda_p <- 0.5 # unitless
# num_planets <- 1

# set the components number
K_prop = com_num 
fit_mat <- as.matrix(fit)
simulation_sample_size = dim(fit_mat)[1] # Loaded stan HMC samples
n = simulation_sample_size
data = fit_mat[,1:dim(fit_mat)[2]-1]
for (i in c(3,7)){
  data[,i] <- exp(data[,i])-1
}
i <- 2
data[,i] <- exp(data[,i])
colnames(data) <- c("C","P","K","e","w","M0","sigma")
data
Stansample = data
variable_num = 6

sample_pi_phi(dens,Stansample,our_sample,variable_num,sample_num = 50000)
plot(our_sample[,6])

# plot(int_out/(sum(int_out)/30),pch = 20)



