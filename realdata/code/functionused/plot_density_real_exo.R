rm(list = ls())
load('intresults_grid_0.4_5.1_tol_0.1_int_out_large_200_new')
library(ggsci)
sample_pi_phi = function(dens,Stansample,our_sample,variable_num,sample_num,True_Data = NULL,grid_search = NULL,int_out = NULL){
  # add
  # True_Data = c()
  # mean_true = c(-5,-5)
  # #mean_true = cbind(mean_true,c(5,5))
  # for(i in 1:sample_num){
  #   #tmpsample1 = sample(x =1:2 ,size = 1,replace = TRUE,prob = c(1/3,2/3))
  #   # tmpsample2 = mvrnorm(n = 1,mean_true[,tmpsample1],Sigma = diag(2))
  #   tmpsample2 = mvrnorm(n = 1,mean_true,Sigma = diag(2))
  #   True_Data = rbind(True_Data,tmpsample2)
  # }
  # multidimension
  # True_Data = c()
  # mean_true = rep(-5,6)
  # mean_true = cbind(mean_true,rep(5,6))
  # for(i in 1:sample_num){
  #   tmpsample1 = sample(x =1:2 ,size = 1,replace = TRUE,prob = c(1/3,2/3))
  #   tmpsample2 = mvrnorm(n = 1,mean_true[,tmpsample1],Sigma = diag(6))
  #   True_Data = rbind(True_Data,tmpsample2)
  # }
  #multimode
  # True_Data = c()
  # mean_true = c(-11,12,-8,7,-2)
  # mean_true = rbind(mean_true,mean_true)
  # for(i in 1:sample_num){
  #   tmpsample1 = sample(x =1:5 ,size = 1,replace = TRUE,prob = c(1/15,2/15,3/15,4/15,5/15))
  #   tmpsample2 = mvrnorm(n = 1,mean_true[,tmpsample1],Sigma = diag(2))
  #   True_Data = rbind(True_Data,tmpsample2)
  # }
  #
  # multimode new version
  # True_Data1 = c()
  # mean_true1 = c(-11,12,-8,7,-2)
  # mean_true1 = rbind(mean_true1,mean_true1)
  # for(i in 1:sample_num){
  #   tmpsample1 = sample(x =1:5 ,size = 1,replace = TRUE,prob = c(1/15,2/15,3/15,4/15,5/15))
  #   tmpsample2 = mvrnorm(n = 1,mean_true1[,tmpsample1],Sigma = diag(2))
  #   True_Data = rbind(True_Data1,tmpsample2)
  # }
  #
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
  #sample_get = data.frame(sample = sample_get,ID = 'phi_mix')
  Stansample = data.frame(sample = Stansample[,variable_num],ID = 'Stan Sample')
  our_sample = data.frame(sample = our_sample[,variable_num],ID = 'Our sample')
  #True_Data_sample = data.frame(sample = True_Data[,variable_num],ID = 'True sample')
  data_sample = rbind(sample_get,Stansample,our_sample)
  print(data_sample)
  p1<-ggplot(data_sample, aes(x = sample,color = ID )) + geom_density(adjust = 0.3)
  grid_search = data.frame(x = grid_search)
  #int_out = data.frame(y=200*(int_out/(sum(int_out))))
  #c = 1/(10^(-191.71))
  c = 35/(sum(int_out))
  int_out = data.frame(y=(int_out*c))
  
  df_int = cbind(grid_search,int_out,ID = 'Integral Density')
  p2 = p1+geom_point(data = df_int,aes(x = x,y = y))+theme_bw()
  #p1 = p1 + scale_color_npg()
  #p1 = p1 + scale_color_ucscgb()
  #p1 = p1 + scale_color_igv()
  #p1 = p1 + scale_color_d3()
  return(p2)
}
# 12: warpU+gibbs_changes  13: for gibbs only, 10000: only for warpU, 25: MH(Norm+Unif)+warpU,24: MH(Norm+Unif) only
load('./sample_stored/Initial_Exo_Real_Data_com_num_10stage_num_6inser_23density_2N=20000')
our_sample =  results_sample$w_all
#our_sample = results_sample$w_all[5000:10000,]

#our_sample = our_sample[15000:(dim(our_sample)[1]),]
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
source('./functionused/simple_case_density.R')


# Given values
# tau <- 20 # days (stellar rotation period)
# alpha <- sqrt(3) # m/s
# lambda_e <- 50 # days
# lambda_p <- 0.5 # unitless
# num_planets <- 1

# set the components number

# for do not have initial sample only
dim_num = dim(sample_range)[2]
initial_data = matrix(0,nrow = N,ncol = dim_num)
for (i in 1:dim_num) {
  tmpx = runif(n = N,min = sample_range[1,i],max = sample_range[2,i])
  initial_data[,i] = tmpx
}
data = initial_data
#
K_prop = com_num 
Stansample = data
variable_num = 6

sample_pi_phi(results_sample$dens,Stansample=True_Data,our_sample,variable_num,sample_num = 50000,grid_search = grid_search,int_out = int_out)
#plot(our_sample[,2])

# plot(int_out/(sum(int_out)/30),pch = 20)



