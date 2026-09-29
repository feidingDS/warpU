## Comparison Simulation -- Stochastic Bridge Estimation
rm(list = ls())
# Libraries
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)
library(ggplot2)

num_level = 1

load('./results/AdaptiveWarpU_Skewt.RData')


# provided data method
#method_used = 'gwl'
time_num_used = 1
method_used = 'Adaptive_warpU'
#method_used = 'Adaptive_warpU'
## source the related functions
## skew_t correnspond to density number 123
density_fun_used = 'skew_t'
#density_fun_used = 'mixture_Normal'
#density_fun_used = "real_exo"
narrow_flag = FALSE
density_num = 123
set.seed(23)
## source the GWL algorithm
if(density_fun_used == "skew_t"){
  source('./codes/functions_used/GWLD_skewT.R')
}else{
  source('./codes/functions_used/GWLD.R')
}
source('./codes/functions_used/Stochastics_functions.R')
source('./codes/functions_used/warpu code/EMhd_diagonal.R')
source('./codes/functions_used/warpu code/Bridge_Sampling_funs_eff.R')
if(density_fun_used == "skew_t"){
  source('./codes/functions_used/skew_t_density.R')
}
#####
#total_sample_num = dim(est_result$x_final_stage)[1]
if(density_fun_used == "skew_t"){
  true_Z = 1
}else if(density_fun_used == 'mixture_Normal'){
  true_Z = 2*pi
}else if(density_fun_used == 'real_exo'){
  #true_Z = -193.68
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
  load('./codes/set posterior function/dataset1_one_planet_posterior.RData')
  if(narrow_flag == FALSE){
    source('./codes/R model code/full_priors.R')
    Pmin <- rep(1.25,3)
    Pmax <- rep(10^4,3)
  }
  source("./codes/R model code/log_like.R")
  source("./codes/R model code/log_post.R")
  source("./codes/R model code/cov_fun.R")
  source("./codes/R model code/cov_make.R")
  source("./codes/R model code/planet_model.R") # Added in sqrt to phi
  source('./codes/functions_used/Functions_AdaptiveWarpU.R')
  true_Z = -193.71
  matrix_orig = c()
  matrix_Warp = c()
  matrix_Stoc = c()
}

p1
if(method_used == "PL"){
  num_level = dim(PL_list[[1]])[1]
  ## condition
  # sample_condition = t(PL_list[[1]][1,,])
  # max_condition = apply(sample_condition, 1, max)
  # min_condition = apply(sample_condition, 1, min)
  # condition_index = (max_condition<=20)&(min_condition>=-20)
  # sample_condition = sample_condition[condition_index,]
  # aa = (dim(sample_condition)[1]/2):(dim(sample_condition)[1])
  # index_PL_used = sample(aa,25937)
  # sample_results = sample_condition[ index_PL_used, ]
  ##
  # aa = floor((dim(PL_list[[1]])[3]/3)) : ( dim(PL_list[[1]])[3])
  #index_PL_used = sort(1:dim(PL_list[[1]])[3],decreasing = TRUE)[1:25937]
  # index_PL_used  = sample(1:dim(PL_list[[1]])[3],25938)
  #index_PL_used=  floor(seq(from = min(aa), to = max(aa), length.out = 5000))
  #index_PL_used  = sample(aa,25937)
  #index_PL_used = 1:25938#15715:31428
  # sample_results = t(PL_list[[1]][ time_num_used , , index_PL_used  ])
  # sample_results = sample_results[1:25937, ]
  hist(sample_results[,6])
  plot(density(sample_results[,6]))
  
}else if(method_used == 'Adaptive_warpU'){
  Index_set = seq(from = 1, to= N, by = 1)
  #Index_set = sample(1:10000, size = 5000)
  sample_results = results_sample$w_all[Index_set,]
  #hist(sample_results[,6])
}

if(method_used == 'gwl'){
  mean_zhat = c()
  for (rept_num in 1:RepeatNum) {
    mean_zhat = c(mean_zhat,Repeat_store_zhat[[rept_num]][[11]])
  }
  mean_zhat = mean(mean_zhat)
  GWL_estimator = mean_zhat
  GWL_estimator_RMSE = sqrt((GWL_estimator - true_Z)^2)
  print('this is RMSE of GWL')
  GWL_estimator_RMSE
  
  length_record = c()
  for (i in 1:length(x_subregion_store)) {
    length_record = c(length_record, dim(x_subregion_store[[i]])[1])
    
  }
  sum(length_record)
  x_subregion_store
  #number_last_stage_target_evaluation = 2* (sum(length_record)  -  (dim(x_subregion_store[[1]])[1]-1)) +  (dim(x_subregion_store[[1]])[1]-1)
  number_last_stage_target_evaluation = n_store[[11]]
  
}else if(method_used  == 'PL'){
  number_last_stage_target_evaluation = dim(sample_results)[1]
  sampleNum_by_GWL = dim(sample_results)[1]
}else if (method_used == 'Adaptive_warpU'){
  number_last_stage_target_evaluation = dim(sample_results)[1]
  sampleNum_by_GWL = dim(sample_results)[1]
}
if(density_fun_used == 'real_exo'){
  #Cost_est = 49685
  Cost_est = 283000
  N1_in = number_last_stage_target_evaluation
  number_last_stage_target_evaluation = Cost_est
}


### get the samples from GWL results
#scale_grid = c(-1.5,-1,-0.5,0,0.5,1)
#number_grid = floor ( ((10^scale_grid) * number_last_stage_target_evaluation ) / com_num )
#sampleNum_by_GWL = floor(number_last_stage_target_evaluation/com_num)
#sample_by_GWL =  GetSamples_from_LastStage(sample_size = sampleNum_by_GWL, estimate_result_GWL = est_result,david_data = 1)
RMSE_orig_mat = c()
RMSE_WarpBs_mat = c()
RMSE_StochBs_mat = c()
time_com_begin = proc.time()
for (repNum in 1:RepeatNum) {
  set.seed(repNum+23)
  RMSE_orig = c()
  RMSE_WarpBs = c()
  RMSE_StochBs = c()
  Est_orig = c()
  Est_WarpBs = c()
  Est_StochBs = c()
  #number_grid = seq(from = 5000, to = 20000, by = 50)
  #scale_grid = c(-1,-0.5,0,0.5)
  scale_grid = c(-1,-0.5,0,0.5,1)
  #scale_grid = c(0)
  number_grid = floor ( ((10^scale_grid) * number_last_stage_target_evaluation ) / (2*com_num-1) )
  if(method_used == 'gwl'){
    log_g_estimates = Repeat_store_log_g_estimate[[1]]
    zhat_store = Repeat_store_zhat[[1]]
    x_subregion_store = Repeat_store_x_subregion_store[[1]]
  }
  
  for (num_sample in (2*number_grid)) {
    sample_used = num_sample
    print('timebegin')
    time_beggin = proc.time()
    print(time_beggin)
    # if(method_used == 'gwl'){
    #   all_GWLSample_used =  GetSamples_from_LastStage(sample_size = sample_used, estimate_result_GWL = est_result,david_data = 1)
    # }else{
    #   #index_used = sample(x = (1:sampleNum_by_GWL), size = (((2*com_num-1) ) *sample_used) ,replace = TRUE)
    #   index_used = sample(x = (1:sampleNum_by_GWL), size = (sample_used) ,replace = TRUE)
    #   all_GWLSample_used = sample_results[index_used,]
    # }
    if(method_used == 'gwl'){
      
      sample_index_1 = sample(1:25937,size = floor(sample_used/2),replace = FALSE)
      sample_index_2 = sample(1:25937,size = floor(sample_used/2),replace = FALSE)
      sample_index = c(sample_index_1 ,sample_index_2)
      all_GWLSample_used =  sample_results[sample_index,]#all_GWLSample_used[1:25937,1:10]#GetSamples_from_LastStage(sample_size = sample_used, estimate_result_GWL = est_result,david_data = 1)
    }else{
      #index_used = sample(x = (1:sampleNum_by_GWL), size = (((2*com_num-1) ) *sample_used) ,replace = TRUE)
      # index_used = sample(x = (1:sampleNum_by_GWL), size = (sample_used) ,replace = TRUE)
      sample_index_1 = sample(1:sampleNum_by_GWL,size = floor(sample_used/2),replace = FALSE)
      sample_index_2 = sample(1:sampleNum_by_GWL,size = floor(sample_used/2),replace = FALSE)
      index_used = c(sample_index_1 ,sample_index_2)
      all_GWLSample_used = sample_results[index_used,]
    }
    print('timeused')
    print(proc.time()-time_beggin)
    
    # divide the data into two parts
    fit_mat = all_GWLSample_used
    simulation_sample_size = dim(fit_mat)[1] 
    n = simulation_sample_size
    first_half = 1:floor(n/2)#sample(1:n,size=floor(n/2),replace=FALSE)# 1:(n/2)
    second_half = setdiff(1:n,first_half) #(n/2+1):n
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
    
    # transfer the data from the target density
    #subsample_ind = sample(x = (1:dim(data_p1_second_half)[1]), size =  floor(dim(data_p1_second_half)[1]/(2*K_prop-1)) )
    if(density_fun_used == 'real_exo'){
      data_p1_second_half = sample_results
      data_tp1 = rtp1(sample_results)
    }else{
      data_tp1 = rtp1(data_p1_second_half)
    }
      
    proposal_evaluations <- proposal_evaluations+dim(data_p1_second_half)[1]*K_prop*(K_prop+1)
    
    ##########
    # specify the number of samples from normal
    if(density_fun_used == 'real_exo'){
      
      if(method_used == 'Adaptive_warpU'){
        # N2 = floor(n/2) * (2*com_num-1)
        N2 = floor(n/2*(2*com_num-1)/com_num)
      }else{
        N2 = floor( (n/2*(2*com_num-1) - (com_num-1)*N1_in) /com_num)
      }
      
    }else{
      if(method_used == 'Adaptive_warpU'){
        # N2 = floor(n/2) * (2*com_num-1)
        N2 = floor(n/2*(2*com_num-1)/com_num)
      }else{
        N2 = floor(n/2)
      }
    }
    
    
    #N2 = floor(N2/(2*K_prop-1))
    data_normal = rmvnorm(N2,mean=rep(0,Dim),sigma=diag(1,Dim))
    # apply the optimal bridge sampling on data_tp1 and data_normal
    # (n/2+N2)K target evaluations
    # (n/2+N2)K^2 Normal evaluations
    # 2(N2 + (n/2)) standard Normal evaluations
    proposal_evaluations <- proposal_evaluations+(dim(data_p1_second_half)[1]+N2)*K_prop^2 
    target_evaluations <- (dim(data_p1_second_half)[1]+N2)*K_prop
    standard_normal_evaluations <- (dim(data_p1_second_half)[1]+N2)*2
    
    ###
    ### get the estimator of original Bridge Sampling method
    #data_phimix =rphi_mix(floor(n/2))
    data_phimix =rphi_mix(floor(n/2*(2*com_num-1)))
    log_bs_Z_hat_original_Bs <- Opt_BS_hd(x1=data_p1_second_half,x2=data_phimix, q1=p1,q2=phi_mix,r0=10)
    print('this is the estimator of original Bridge method ')
    exp(log_bs_Z_hat_original_Bs)
    if(density_fun_used == 'real_exo'){
      RMSE_orig =c(RMSE_orig, ((log10(exp(log_bs_Z_hat_original_Bs)) - true_Z)^2))
      Est_orig = c(Est_orig, log10(exp(log_bs_Z_hat_original_Bs)))
    }else{
      RMSE_orig =c(RMSE_orig, ((exp(log_bs_Z_hat_original_Bs) - true_Z)^2))
    }
    
    ###
    ### get the estimator of warpU Bridge Sampling method
    log_bs_Z_hat_WarpU_Bs <- Opt_BS_hd(x1=data_tp1,x2=data_normal, q1=tp1,q2=p2_warp0,r0=10)
    print('this is the estimator of WarpU Bridge method ')
    exp(log_bs_Z_hat_WarpU_Bs)
    if(density_fun_used == 'real_exo'){
      RMSE_WarpBs =c(RMSE_WarpBs, ((log10(exp(log_bs_Z_hat_WarpU_Bs)) - true_Z)^2))
      Est_WarpBs = c(Est_WarpBs, log10(exp(log_bs_Z_hat_WarpU_Bs)))
    }else{
      RMSE_WarpBs =c(RMSE_WarpBs, ((exp(log_bs_Z_hat_WarpU_Bs) - true_Z)^2))
      
    }
    
    ###
    ### get the estimator of Stochastics Bridge Sampling method
    
    ## one way to use the normal data as the same size with tilde{w_k}
    # split normal distribution
    if(density_fun_used == 'real_exo'){
      
      Opt_Stochas_BS_splitNorm_same = function(w_in){
        tildeW_k  = rtp1_split(w_in)
        split_norm = list()
        length(split_norm) = K_prop
        for (kk in 1:K_prop) {
          if(ceiling(length(tildeW_k[[kk]])) > 0){
            split_norm[[kk]] = rmvnorm( (( floor(Cost_est/N1_in) )*ceiling(length(tildeW_k[[kk]]))),mean=rep(0,Dim),sigma=diag(1,Dim))
          }
          
        }
        return(Opt_BS_hd_Sto_splitNorm(split_data = tildeW_k,split_norm = split_norm))
      }
      
    }else{
      
      Opt_Stochas_BS_splitNorm_same = function(w_in){
        tildeW_k  = rtp1_split(w_in)
        split_norm = list()
        length(split_norm) = K_prop
        for (kk in 1:K_prop) {
          if(ceiling(length(tildeW_k[[kk]])) > 0){
            split_norm[[kk]] = rmvnorm( ((2*com_num-1)*ceiling(length(tildeW_k[[kk]]))),mean=rep(0,Dim),sigma=diag(1,Dim))
          }
          
        }
        return(Opt_BS_hd_Sto_splitNorm(split_data = tildeW_k,split_norm = split_norm))
      }
    }
    
    
    
    ## excute
    #bs_Z_hat_StochasticWarp_Bs = try(Opt_Stochas_BS_splitNorm_same(data_p1_second_half), silent = TRUE)
    
    for(try_num in 1:100){
      try({
        bs_Z_hat_StochasticWarp_Bs = Opt_Stochas_BS_splitNorm_same(data_p1_second_half)
        break #break/exit the for-loop
      }, silent = FALSE)
    }
    
    
    if (try_num == 100){
      if(density_fun_used == 'real_exo'){
        
        Opt_Stochas_BS_splitNorm_same = function(w_in){
          tildeW_k  = rtp1_split(w_in)
          split_norm = list()
          length(split_norm) = K_prop
          for (kk in 1:K_prop) {
            if(ceiling(length(tildeW_k[[kk]])) > 0){
              split_norm[[kk]] = rmvnorm( (( floor(Cost_est/N1_in) )*ceiling(length(tildeW_k[[kk]]))),mean=rep(0,Dim),sigma=diag(1,Dim))
            }
            
          }
          return(Opt_BS_hd_Sto_splitNorm(split_data = tildeW_k,split_norm = split_norm))
        }
        
      }else{
        Opt_Stochas_BS_splitNorm_same = function(w_in){
          tildeW_k  = rtp1_split(w_in)
          split_norm = list()
          length(split_norm) = K_prop
          for (kk in 1:K_prop) {
            if(ceiling(length(tildeW_k[[kk]])) > 0){
              split_norm[[kk]] = rmvnorm( ((2*com_num-1)*ceiling(length(tildeW_k[[kk]]))),mean=rep(0,Dim),sigma=diag(1,Dim))
            }
            
          }
          return(Opt_BS_hd_Sto_splitNorm_Error(split_data = tildeW_k,split_norm = split_norm))
        }
      }
      
      
      
      bs_Z_hat_StochasticWarp_Bs = Opt_Stochas_BS_splitNorm_same(data_p1_second_half)
      if(density_fun_used == 'real_exo'){
        RMSE_StochBs =c(RMSE_StochBs, ((log10((bs_Z_hat_StochasticWarp_Bs)) - true_Z)^2))  
        Est_StochBs = c(Est_StochBs,log10((bs_Z_hat_StochasticWarp_Bs)) )
      }else{
        RMSE_StochBs =c(RMSE_StochBs, (((bs_Z_hat_StochasticWarp_Bs) - true_Z)^2))  
        
      }
    }else{ 
      print('this is the estimator of stochastics WarpU Bridge method ')
      #exp(log_bs_Z_hat_StochasticWarp_Bs)
      if(density_fun_used == 'real_exo'){
        RMSE_StochBs =c(RMSE_StochBs, ((log10((bs_Z_hat_StochasticWarp_Bs)) - true_Z)^2))  
        Est_StochBs = c(Est_StochBs,log10((bs_Z_hat_StochasticWarp_Bs)) )
        
      }else{
        RMSE_StochBs =c(RMSE_StochBs, (((bs_Z_hat_StochasticWarp_Bs) - true_Z)^2))  
        
      }
    }
    
    
  }
  RMSE_orig_mat = rbind(RMSE_orig_mat, (RMSE_orig))
  RMSE_WarpBs_mat = rbind(RMSE_WarpBs_mat, (RMSE_WarpBs))
  RMSE_StochBs_mat = rbind(RMSE_StochBs_mat, (RMSE_StochBs))
  if(density_fun_used == 'real_exo'){
    matrix_orig = rbind(matrix_orig,Est_orig)
    matrix_Warp = rbind(matrix_Warp, Est_WarpBs)
    matrix_Stoc = rbind(matrix_Stoc, Est_StochBs)
  }
  
  
}

RMSE_orig_mat1 = sqrt(colMeans(RMSE_orig_mat))
RMSE_WarpBs_mat1 = sqrt(colMeans(RMSE_WarpBs_mat))
RMSE_StochBs_mat1= sqrt(colMeans(RMSE_StochBs_mat))
RMSE_orig_mat2 = (apply(RMSE_orig_mat, 2, sd))
RMSE_WarpBs_mat2 = (apply(RMSE_WarpBs_mat, 2, sd))
RMSE_StochBs_mat2 = (apply(RMSE_StochBs_mat, 2, sd))
time_com_used = proc.time() -  time_com_begin 

plot(RMSE_orig)
plot(RMSE_WarpBs)
plot(RMSE_StochBs)


#x_seq = c(-1,-0.5,0,0.5)
x_seq = c(-1,-0.5,0,0.5,1)
#x_seq = c(0)

# save(list = ls(), file = paste0('./results/May23_cost_large_Comparison','com_',com_num,'Repeat_num',RepeatNum,'num_level',num_level,'method_used_',method_used,'time_num_used',time_num_used,'density_fun_used',density_fun_used,'.RData'))
save(list = ls(), file = paste0('./results/WarpU_samples_skewt_cost_large_Comparison.RData'))
