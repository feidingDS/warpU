rm(list = ls())
library(ggsci)
library(plotly)
sample_pi_phi = function(dens,Stansample,our_sample,variable_num,sample_num,True_Data = NULL, last_stage_MCMC = NULL){
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
  True_Data = c()
  mean_true = c(-11,12,-8,7,-2)
  mean_true = rbind(mean_true,mean_true)
  for(i in 1:sample_num){
    tmpsample1 = sample(x =1:5 ,size = 1,replace = TRUE,prob = c(1/15,2/15,3/15,4/15,5/15))
    tmpsample2 = mvrnorm(n = 1,mean_true[,tmpsample1],Sigma = diag(2))
    True_Data = rbind(True_Data,tmpsample2)
  }
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
  sample_get = data.frame(sample = sample_get,ID = 'phi_mix')
  Stansample = data.frame(sample = Stansample[,variable_num],ID = 'Given Data')
  our_sample = data.frame(sample = our_sample[,variable_num],ID = 'Our sample')
  True_Data_sample = data.frame(sample = True_Data[,variable_num],ID = 'True sample')
  last_stage_MCMC = data.frame(sample = last_stage_MCMC[, variable_num], ID = 'MCMC')
  data_sample = rbind(sample_get,Stansample,our_sample,True_Data_sample, last_stage_MCMC)
  print(data_sample)
  p1<-ggplot(data_sample, aes(x = sample,color = ID )) + geom_density(adjust = 0.8)+theme_bw()
  #p1 = p1 + scale_color_npg()
  #p1 = p1 + scale_color_ucscgb()
  #p1 = p1 + scale_color_igv()
  #p1 = p1 + scale_color_d3()
  return(p1)
}

# get the stage animation 

get_stage_animation = function(stage_sample,True_Data1 = NULL,sample_num = NULL,variable_num = NULL,sample_range){
  if(is.null(True_Data1)){
    print("flag")
    True_Data1 = c()
    mean_true = c(-11,12,-8,7,-2)
    mean_true = rbind(mean_true,mean_true)
    for(i in 1:sample_num){
      tmpsample1 = sample(x =1:5 ,size = 1,replace = TRUE,prob = c(1/15,2/15,3/15,4/15,5/15))
      tmpsample2 = mvrnorm(n = 1,mean_true[,tmpsample1],Sigma = diag(2))
      True_Data1 = rbind(True_Data1,tmpsample2)
    }
    print('True_Data')
    print(head(True_Data1))
  }
  print('True_Data')
  print(True_Data1)
  
  
  # directly plot the density by function
  P_density_pro = c(1/15,2/15,3/15,4/15,5/15)
  P_density_mu = c(-11,12,-8,7,-2)
  P_density_sigma =  c(1,1,1,1,1)
  P_density_x <- seq(from=sample_range[1,variable_num], to=sample_range[2,variable_num], length.out = 2000)
  P_density_value = dnormm(P_density_x, P_density_pro , P_density_mu, P_density_sigma, log=FALSE)
  
  
  stage_number = dim(stage_sample)[2]
  each_stage_num = dim(stage_sample)[1]
  df_stage = c()
  True_Data_sample = c()
  P_density_data = c()
  
  tmpTrue_Data_sample1 = True_Data1[,variable_num]
  print('True_Data[1:2,variable_num]')
  print(True_Data1[1:2,variable_num])
  print('tmpTrue_Data_sample1')
  print(head(tmpTrue_Data_sample1))
  for (i in 1:stage_number) {
    tmp_df_stage = data.frame(sample = stage_sample[,i],stage_num = i)
    df_stage = rbind(df_stage,tmp_df_stage)
    
    tmpTrue_Data_sample = data.frame(sample = tmpTrue_Data_sample1, stage_num = i)
    
    print("tmpTrue_Data_sample")
    print(head(tmpTrue_Data_sample))
    
    True_Data_sample = rbind(True_Data_sample,tmpTrue_Data_sample)
    print("True_Data_sample")
    print(head(True_Data_sample))
    
    tmpP_density_data = data.frame(x = P_density_x ,y = P_density_value,stage_num = i)
    P_density_data = rbind(P_density_data,tmpP_density_data)
    
  }
  

  
  
  
  print(head(True_Data_sample))
  # p1 = ggplot(df_stage,aes(x =sample,color = stage_num,
  #                    frame = stage_num))+ geom_density()+theme_bw() + transition_time(stage_num) +ease_aes('linear')
  # p1 = ggplot(df_stage,aes(x =sample,color = stage_num))+ geom_density()+theme_bw() +  transition_manual(tage_num)
  p1 = ggplot(df_stage,aes(x =sample,color = 'Our sample',
                                  frame = stage_num))+ geom_density(adjust = 0.21)+theme_bw() + geom_density(data = True_Data_sample,aes(x = sample,color = 'True sample'),adjust = 0.21)+geom_line(data = P_density_data,aes(x = x,y = y,color = 'Density Plot by Function'))
  p <- ggplotly(p1)
  p
  # list_out = list()
  # list_out$df_stage = df_stage
  # list_out$True_Data_sample = True_Data_sample
  return(p)
  
}

get_dis_with_baseline = function(results_sample1,baselinevalue, variable_num){
  baselinevalue = baselinevalue[variable_num]
  results_sample1 = results_sample1[,variable_num]
  df = data.frame(stage_num = 1:length(results_sample1),distance = results_sample1)
  p1 = ggplot(data =df,aes(x = stage_num,y = distance) ) + geom_point()+geom_line() + geom_hline(aes(yintercept=baselinevalue))
  return(p1)
}


#
# 12: warpU+gibbs_changes  13: for gibbs only, 10000: only for warpU, 25: MH(Norm+Unif)+warpU,24: MH(Norm+Unif) only
load('./sample_stored/List_Final_small_num_sample_animation_com_meanLatest_New_Adaptive_large_sigma50_ourdistance_with_three_distance_N2_com_num_10stage_num_10inser_23density_2N=40000N_Reat1Pre_TRUElarger_than_3')
#our_sample =  results_sample$w_all
sample_len = dim(results_sample$w_all)[1]
our_sample = results_sample$w_all[ceiling(sample_len/2) :sample_len,]
stage_sample = results_sample$store_sample_stage
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
library(gganimate)
library(plotly)
library(LaplacesDemon)
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
variable_num = 2
#
last_stage_MCMC = Stand_MCMC(Num_sample = 50000, proposal_density = proposal_density, proposal_sample_fun = proposal_sample_fun, w_star = c(-10,-10),Dim_w = dim(results_sample$w_all)[2],dens = results_sample$dens )
#
sample_pi_phi(results_sample$dens,Stansample,our_sample,variable_num,sample_num = 50000,last_stage_MCMC = last_stage_MCMC)

p= get_stage_animation (stage_sample = stage_sample,sample_num = 5000,variable_num = 1,sample_range = sample_range)
#mean
get_dis_with_baseline(results_sample1 = results_sample$mean_mat,baselinevalue = results_sample$baseline_mean_dis,variable_num = variable_num)
# pro
get_dis_with_baseline(results_sample1 = results_sample$pro_mat,baselinevalue = results_sample$baseline_pro_dis,variable_num = variable_num)

#plot(our_sample[,2])

# plot(int_out/(sum(int_out)/30),pch = 20)

## compare with MCMC with Phi_mix as proposal




