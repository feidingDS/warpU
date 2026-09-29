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
library(ggpubr)


num_level = 1

# provided data method
#method_used = 'gwl'
#method_used = 'PL'
time_num_used = 1
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


#x_seq = c(-1,-0.5,0,0.5)
x_seq = c(-1,-0.5,0,0.5,1)

##
# below to get the plot

# below to get the plot
# load('./results/April_9_super_large_0.15_Comparisoncom_10Repeat_num50num_level12method_used_PLtime_num_used1') # pl data
load('./results/PT_samples_skewt_cost_large_Comparison.RData')
###
# RMSE_orig_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_orig_mat1[2:5], se = RMSE_orig_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2))[2:5])
# RMSE_WarpBs_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_WarpBs_mat1[2:5], se = RMSE_WarpBs_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2))[2:5])
# RMSE_StochBs_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_StochBs_mat1[2:5], se = RMSE_StochBs_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2))[2:5])
RMSE_orig_data = data.frame(x_seq = x_seq, RMSE =  RMSE_orig_mat1, se = RMSE_orig_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2)))
RMSE_WarpBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_WarpBs_mat1, se = RMSE_WarpBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2)))
RMSE_StochBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_StochBs_mat1, se = RMSE_StochBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2)))
RMSE_orig_data$curveID = 'BS'
RMSE_WarpBs_data$curveID = 'WB'
RMSE_StochBs_data$curveID = 'SWB'
RMSE_data_all = rbind(RMSE_orig_data,RMSE_WarpBs_data,RMSE_StochBs_data)
if(method_used=='gwl'){
  RMSE_gwl_data = data.frame(x_seq = x_seq, RMSE =  GWL_estimator_RMSE)
}



RMSE_data_all$curveID <- factor(RMSE_data_all$curveID , levels = c("BS", "WB", "SWB"))
p2 = ggplot(RMSE_data_all, aes(x_seq, RMSE, group = curveID, color = curveID)) +
  geom_line(aes(linetype = curveID), size = 1)+ geom_point(aes(shape= curveID), size = 3) +
  geom_errorbar(aes(x = x_seq, y = RMSE,ymin= RMSE-2*se,ymax=RMSE+2*se,color = curveID, linetype = curveID),alpha=I(1),width=.1, size = 1)

p2 = p2  + theme_bw() + scale_linetype_manual(values=c("solid", "dotdash", 'dotted'))+ scale_colour_manual(values =  c("blue","red",'green')  )+scale_fill_manual(values =  c("blue","red",'green') ) +
  theme(axis.title.x = element_text(size = 16),axis.text.x = element_text(size = 16),axis.title.y = element_text(size = 16),axis.text.y = element_text(size = 16),legend.position = c(0.85,0.72), legend.title = element_blank())

p2 = p2+ ylab("RMSE")  +
  theme(legend.position = c(0.82, 0.81),legend.justification = c(0, 1))+
  scale_color_manual(values = c("#2ca02c","red","blue")) + scale_linetype_manual(values=c("dotdash", 'dotted',"solid"))
p2 = p2+ ylim(c(0,0.5))
p2

# below to get the plot
# load('./results/Compare_newexact_Mar26com_10Repeat_num50num_level1method_used_Adaptive_warpU') # adaotuve warp-U data
load('./results/WarpU_samples_skewt_cost_large_Comparison.RData') # adaotuve warp-U data
###
# RMSE_orig_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_orig_mat1[2:5], se = RMSE_orig_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2))[2:5])
# RMSE_WarpBs_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_WarpBs_mat1[2:5], se = RMSE_WarpBs_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2))[2:5])
# RMSE_StochBs_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_StochBs_mat1[2:5], se = RMSE_StochBs_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2))[2:5])
RMSE_orig_data = data.frame(x_seq = x_seq, RMSE =  RMSE_orig_mat1, se = RMSE_orig_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2)))
RMSE_WarpBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_WarpBs_mat1, se = RMSE_WarpBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2)))
RMSE_StochBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_StochBs_mat1, se = RMSE_StochBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2)))
RMSE_orig_data$curveID = 'BS'
RMSE_WarpBs_data$curveID = 'WB'
RMSE_StochBs_data$curveID = 'SWB'
RMSE_data_all = rbind(RMSE_orig_data,RMSE_WarpBs_data,RMSE_StochBs_data)
if(method_used=='gwl'){
  RMSE_gwl_data = data.frame(x_seq = x_seq, RMSE =  GWL_estimator_RMSE)
}



RMSE_data_all$curveID <- factor(RMSE_data_all$curveID , levels = c("BS", "WB", "SWB"))
p3 = ggplot(RMSE_data_all, aes(x_seq, RMSE, group = curveID, color = curveID)) +
  geom_line(aes(linetype = curveID), size = 1)+ geom_point(aes(shape= curveID), size = 3) +
  geom_errorbar(aes(x = x_seq, y = RMSE,ymin= RMSE-2*se,ymax=RMSE+2*se,color = curveID, linetype = curveID),alpha=I(1),width=.1, size = 1)

p3 = p3  + theme_bw() + scale_linetype_manual(values=c("solid", "dotdash", 'dotted'))+ scale_colour_manual(values =  c("blue","red",'green')  )+scale_fill_manual(values =  c("blue","red",'green') ) +
  theme(axis.title.x = element_text(size = 16),axis.text.x = element_text(size = 16),axis.title.y = element_text(size = 16),axis.text.y = element_text(size = 16),legend.position = c(0.85,0.72), legend.title = element_blank())

p3 = p3+ ylab("RMSE") +
  theme(legend.position = c(0.82, 0.81),legend.justification = c(0, 1))+
  scale_color_manual(values = c("#2ca02c","red","blue")) + scale_linetype_manual(values=c("dotdash", 'dotted',"solid"))
p3 = p3+ ylim(c(0,0.5))
p3
p3 = p3+ xlab('')
p3 = p3 +  theme(axis.title.x = element_text(size = 16),axis.text.x = element_text(size = 16),axis.title.y = element_text(size = 16,angle=90,hjust=0.5),axis.text.y = element_text(size = 16),legend.position = c(0.65,0.72), legend.title = element_blank(),) + ylab('RMSE')
p3


# load('./results/Sept9_cost_large_Comparisoncom_10Repeat_num30num_level1method_used_gwltime_num_used1density_fun_usedskew_t') # gwl data
load('./results/gwl_samples_skewt_cost_large_Comparison.RData') # gwl data
###
# RMSE_orig_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_orig_mat1[2:5], se = RMSE_orig_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2))[2:5])
# RMSE_WarpBs_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_WarpBs_mat1[2:5], se = RMSE_WarpBs_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2))[2:5])
# RMSE_StochBs_data = data.frame(x_seq = x_seq[2:5], RMSE =  RMSE_StochBs_mat1[2:5], se = RMSE_StochBs_mat2[2:5]/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2))[2:5])
RMSE_orig_data = data.frame(x_seq = x_seq, RMSE =  RMSE_orig_mat1, se = RMSE_orig_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_orig_mat))^(-1/2)))
RMSE_WarpBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_WarpBs_mat1, se = RMSE_WarpBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_WarpBs_mat))^(-1/2)))
RMSE_StochBs_data = data.frame(x_seq = x_seq, RMSE =  RMSE_StochBs_mat1, se = RMSE_StochBs_mat2/sqrt(RepeatNum)*(1/2*(colMeans(RMSE_StochBs_mat))^(-1/2)))
RMSE_orig_data$curveID = 'BS'
RMSE_WarpBs_data$curveID = 'WB'
RMSE_StochBs_data$curveID = 'SWB'
RMSE_data_all = rbind(RMSE_orig_data,RMSE_WarpBs_data,RMSE_StochBs_data)
if(method_used=='gwl'){
  RMSE_gwl_data = data.frame(x_seq = x_seq, RMSE =  GWL_estimator_RMSE)
}



RMSE_data_all$curveID <- factor(RMSE_data_all$curveID , levels = c("BS", "WB", "SWB"))
p1 = ggplot(RMSE_data_all, aes(x_seq, RMSE, group = curveID, color = curveID)) +
  geom_line(aes(linetype = curveID), size = 1)+ geom_point(aes(shape= curveID), size = 3) +
  geom_errorbar(aes(x = x_seq, y = RMSE,ymin= RMSE-2*se,ymax=RMSE+2*se,color = curveID, linetype = curveID),alpha=I(1),width=.1, size = 1)
  
p1 = p1  + theme_bw() + scale_linetype_manual(values=c("solid", "dotdash", 'dotted'))+ scale_colour_manual(values =  c("blue","red",'green')  )+scale_fill_manual(values =  c("blue","red",'green') ) +
  theme(axis.title.x = element_text(size = 16),axis.text.x = element_text(size = 16),axis.title.y = element_text(size = 16),axis.text.y = element_text(size = 16),legend.position = c(0.85,0.72), legend.title = element_blank())

p1 = p1+ ylab("RMSE")  +
  theme(legend.position = c(0.82, 0.81),legend.justification = c(0, 1))+
  scale_color_manual(values = c("#2ca02c","red","blue")) + scale_linetype_manual(values=c("dotdash", 'dotted',"solid"))
p1 = p1+ ylim(c(0,0.5))
p1



p1 = p1 + theme(axis.title.x = element_text(size = 16),axis.text.x = element_text(size = 16),axis.title.y = element_blank(),axis.text.y = element_blank(),legend.position = 'none', legend.title = element_blank())
p1
p1 = p1 + xlab( expression(Log[10]~ 'prop.'~'of gwl stage 11 target eval.\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ '))
p1
p2 = p2 + theme(axis.title.x = element_text(size = 16),axis.text.x = element_text(size = 16),axis.title.y = element_blank(),axis.text.y = element_blank(),legend.position = 'none', legend.title = element_blank())
p2 = p2 + xlab('')

p6_final =  ggarrange(p3,p1,p2,nrow = 1,widths = c(1.2,1,1))
p6_final
###
pdf.width <- 9
pdf.height <- 4
pdf('./plots/p6_final.pdf',height = pdf.height,width = pdf.width)
print(p6_final)
dev.off()
