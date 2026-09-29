rm(list = ls())
jid = 0
Repeat_num = 1
load('./data/Data_Jones.Rdata')

p1_single <- function(x,log=F){
  log_results = sapply(1:lweights,function(i){log(weights[i])+dmst(x,xis[[i]],Omegas[[i]],alphas[[i]],nus[i],log=T)})
  max_log_results = max(log_results)
  if(!log){
    return(sum(exp(log_results-max_log_results))*exp(max_log_results))
  }
  if(log){
    return(log(sum(exp(log_results-max_log_results)))+max_log_results)
  }
}

pdf.width <- 11
pdf.height <- 7
plot_gaps <- c(5,7,3,2) # bottom, left, top, right
image_gaps <- c(1,1,1,1) # bottom, left, top, right
axis_gaps <- c(4,1.5,0) # label, number, tick (?)
mag_levels <- c(1.5,2.25,2.25,1.75,1.5)  # main, lab, axis, line, pts

true_value <- 1

#lib <- "/n/home06/david_jones/R/x86_64-unknown-linux-gnu-library/"
library(mnormt)
library(sn)

#library(sn)

library(mvtnorm)
library(numDeriv)
library(stats4)
library(MASS)
library(mvtnorm)

bound_lower <- -20
bound_upper <- 20
log_phi_fun <- function(x){
  constraints <- sum(sapply(2:Dim,function(i){x[i]<bound_upper & x[i]>bound_lower}))==(Dim-1)
  if (constraints){
    if (x[1]<(bound_upper+1) & x[1]>bound_lower){
      if (x[1]<bound_upper){
        value <- log(bound_upper-bound_lower)+p1_single(x,log=T)
      } else {
        value <- -(Dim-1)*log(bound_upper-bound_lower)
      }
    } else {
      value <- -Inf
    }
  } else {
    value <- -Inf
  }
  return(value)
}

energy_at_means <- numeric(25)
for (i in 1:25){
  energy_at_means[i] <- -log_phi_fun(xis[[i]])
}
#energy_at_means

energy_at_1000draws <- numeric(1000)
draws <- matrix(NA,1000,10)
for (i in 1:1000){
  draws[i,] <- rp1(1)
  energy_at_1000draws[i] <- -log_phi_fun(draws[i,])
}
index <- which(is.finite(energy_at_1000draws))
range_1000draws <- range(energy_at_1000draws[index])
#plot(draws)
#abline(v=-20,col=2)
#abline(v=20,col=2)
#abline(h=-20,col=2)
#abline(h=20,col=2)

# Subregions definition
lower_energy <- min(floor(min(energy_at_means)-diff(range(energy_at_means))/25),range_1000draws)
upper_energy <- ceiling(max(range_1000draws))
subregion_bounds <- seq(lower_energy,upper_energy,0.1)
num_regions <- length(subregion_bounds)
subregion_bounds[1] <- -Inf
subregion_bounds[num_regions] <- Inf
findInterval(median(subregion_bounds),subregion_bounds,rightmost.closed = TRUE,left.open = TRUE) 

# Proposal dist
sd_now <- 3
prop_dist <- function(x,sd_now){
  u <- runif(1,0,1)
  if (u < ((bound_upper-bound_lower)/(1+bound_upper-bound_lower))){
    value <- rp1(1)
  } else {
    value <- c(runif(1,bound_upper,bound_upper+1),runif(Dim-1,bound_lower,bound_upper))
  }
  return(value)
}

# Settings
rho <- 0
delta1 <- exp(1)-1
delta_end <- 1e-7
n1 <- 1e4
batch_size <- 500 #n1/50
nfactor <- 1.1

delta <- delta1
n <- n1
count <- 0

log_g_estimates <- matrix(0,num_regions,1)
x <- prop_dist(rep(0,Dim),sd_now)
log_phi_value <- log_phi_fun(x)
constraints <- sum(sapply(2:Dim,function(i){x[i]<bound_upper & x[i]>bound_lower}))==(Dim-1)
constraints <- constraints & (x[1]<(bound_upper+1) & x[1]>bound_upper)
if (constraints){
  subregion <- 1
} else {
  # NOTE: -Inf won't happen as density never infinite so left open OK
  # NOTE: Inf can heppen but this is covered by right cloased
  subregion <- 1 + findInterval(-log_phi_value,subregion_bounds,rightmost.closed = TRUE,left.open = TRUE)
}

subregion_count <- numeric(num_regions)
subregion_count[subregion] <- subregion_count[subregion]+1

Repeat_store_log_g_estimate = list()
Repeat_store_x_subregion_store = list()
Repeat_store_zhat = list()
length(Repeat_store_log_g_estimate) = Repeat_num
length(Repeat_store_x_subregion_store) = Repeat_num
length(Repeat_store_zhat) = Repeat_num
time_start = proc.time()

for (rep_num in 1:Repeat_num) {
  subregion_count_store <- list()
  n_store <- list()
  delta_store <- list()
  final_stability <- list()
  log_g_estimates <- matrix(0,num_regions,1)
  
  dist_store <- list()
  zhat_store <- list()
  acceptance_rate <- list()
  adapt_batch_size <- 100
  sd_final_store <- list()
  x_subregion_store <- list()
  for (i in 1:num_regions){
    x_subregion_store[[i]] <- rep(-100,Dim)
  }
  num_samples_store <- 10000
  delta <- delta1
  n <- n1
  count <- 0
  while (delta > delta_end & count < 11){
    
    # Update delta and n
    if (count==0){
      count <- count+1
    } else {
      delta <- sqrt(1+delta)-1
      n <- nfactor*n
      count <- count+1
    }
    n_store[[count]] <- n
    delta_store[[count]] <- delta
    store_all_counts <- matrix(NA,n,num_regions)
    stability <- numeric(n-2*batch_size+1)
    #if (delta <= delta_end){
    #  x_store <- matrix(NA,n,Dim)
    #}
    stab_count <- 1
    accepted <- 0 
    
    for (i in 1:n){
      
      x_prop <- prop_dist(x,sd_now)
      log_phi_value_prop <- log_phi_fun(x_prop)
      constraints <- sum(sapply(2:Dim,function(i){x_prop[i]<bound_upper & x_prop[i]>bound_lower}))==(Dim-1)
      constraints <- constraints & (x_prop[1]<(bound_upper+1) & x_prop[1]>bound_upper)
      if (constraints){
        subregion_prop <- 1
      } else {
        subregion_prop <- 1 + findInterval(-log_phi_value_prop,subregion_bounds,rightmost.closed = TRUE,left.open = TRUE)
      }
      log_subregion_part <- log_g_estimates[subregion]-log_g_estimates[subregion_prop]
      log_phi_part <- 0 # cancels with prop
      log_prop_part <- 0
      #if (delta <= delta_end){
      #  x_store[i,] <- x
      #}
      if (log_phi_value_prop > -Inf){
        u <- min(exp(log_subregion_part+log_phi_part+log_prop_part),1)
        if (runif(1,0,1) < u){
          accepted <- accepted+1
          x <- x_prop
          subregion <- subregion_prop 
          log_phi_value <- log_phi_value_prop
          #if (delta <= delta_end){
          #  x_store[i,] <- x_prop
          #}
          if (((length(x_subregion_store[[subregion]])/Dim) < (num_samples_store+1)) & (count < 11)){
            x_subregion_store[[subregion]] <- rbind(x_subregion_store[[subregion]],x)
          }
        }
      }
      current_log_g_est <- log_g_estimates[subregion]
      log_g_estimates[subregion] <- log(1+delta)+current_log_g_est # Only valid for rho=0
      subregion_count[subregion] <- subregion_count[subregion]+1
      store_all_counts[i,] <- subregion_count
      if (i >= 2*batch_size){
        initial_index <- i - batch_size
        props_initial <- store_all_counts[initial_index,]/sum(store_all_counts[initial_index,])
        props_current <- subregion_count/sum(subregion_count)
        props_ratio <- props_current/props_initial
        props_ratio[which(props_initial==0)] <- 1
        stability[stab_count] <- mean(abs(props_ratio-1))
        stab_count <- stab_count + 1
      }
    }
    acceptance_rate[[count]] <- accepted/n
    sd_final_store[[count]] <- sd_now
    
    final_stability[[count]] <- stability[stab_count-1]
    subregion_count_store[[count]] <- subregion_count
    subregion_count <- numeric(num_regions)
    zhat <- sum(exp(log_g_estimates[2:num_regions]-log_g_estimates[1])/(bound_upper-bound_lower)) # NOTE: only valid for rho=0 case
    zhat_store[[count]] <- zhat
    dist_store[[count]] <- abs(zhat-true_value)/true_value
    print(zhat)
    print(paste("Acceptance rate: ",accepted/n,sep=""))
    #print(paste("Final sd: ",sd_now,sep=""))
    
    if (jid == 1){
      filename <- paste(save_home,"gwl_104_cheat_prop_with_10stage_10000_storage_10d_example_run_stage",count,"_plots.pdf", sep = "")
      pdf(filename,width=pdf.width,height=pdf.height)
      par(mar=plot_gaps,mgp=axis_gaps,oma=image_gaps,mfrow=c(1,2))
      plot(subregion_count_store[[count]],xlab="Subregion",ylab="Count")
      plot(stability,type="l",ylab="Stability",xlab="Iteration")
      dev.off()
    }
  }
  
  Repeat_store_log_g_estimate[[rep_num]] = log_g_estimates
  Repeat_store_x_subregion_store[[rep_num]] = x_subregion_store
  Repeat_store_zhat[[rep_num]] = zhat_store
}
#plot(unlist(dist_store),type="l",xlab="GWL algorithm stage",ylab="Absolute relative error")
#plot(subregion_count_store[[count]])
#(zhat-1)^2
n_total <- sum(unlist(n_store))
time_used = proc.time()- time_start

# save(list=c("Repeat_num","Repeat_store_zhat","Repeat_store_x_subregion_store","Repeat_store_log_g_estimate","sd_final_store","acceptance_rate","final_stability","stability","n_total","zhat_store","dist_store",
#             "subregion_bounds","subregion_count_store","log_g_estimates","delta_store",
#             "n_store","x_subregion_store","rho","prop_dist","log_phi_fun","bound_lower","bound_upper","time_used"),
#      file=paste('./results/',Repeat_num,"RepeatPre9Good_gwl_104_cheat_prop_with_10stage_10000_storage_10d_example_run",jid,".RData",sep=""))


save(list=c("Repeat_num","Repeat_store_zhat","Repeat_store_x_subregion_store","Repeat_store_log_g_estimate","sd_final_store","acceptance_rate","final_stability","stability","n_total","zhat_store","dist_store",
            "subregion_bounds","subregion_count_store","log_g_estimates","delta_store",
            "n_store","x_subregion_store","rho","prop_dist","log_phi_fun","bound_lower","bound_upper","time_used"),
     file="./output/results/gwl/gwl_skewt_samples.RData")

## source the GWL algorithm
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

sample_range = matrix(c(-2.2292,1.2552,41.67,42.49,1.832,3.381,0.007472,0.538570,0.006595,4.860418,0.000676,4.834212,1.057,1.788),2,7)
load('./data/raw/stanData.RData')
True_Data = data

time_used = proc.time() - time_begin
save(list = ls(),file ='./output/results/gwl/gwl_samples_skewT.RData')
