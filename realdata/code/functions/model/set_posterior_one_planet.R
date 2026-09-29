rm(list=ls())

# Paths are resolved from the realdata/ project root via the here package.
library(here)

# Get posterior (including data)
source(here("code/functions/model/log_post.R"))
source(here("code/functions/model/cov_fun.R"))
source(here("code/functions/model/cov_make.R"))
source(here("code/functions/model/planet_model.R"))

dataset_num <- 1
data_now <- read.table(here(paste0("data/rvs_000",dataset_num,".txt")))
colnames(data_now) <- c("time","rv","sd")
head(data_now)

prior_bounds <- read.table(here(paste0("data/prior_bounds_000",dataset_num,".txt")),sep=",")
colnames(prior_bounds) <- c("parameter","planet","min","max")
prior_bounds
source(here("code/functions/model/full_priors.R"))

# Given values
tau <- 20 # days (stellar rotation period)
alpha <- sqrt(3) # m/s
lambda_e <- 50 # days
lambda_p <- 0.5 # unitless

Dim <- 7
num_planets <- 1

p1 <- function(x,log=F){
  m <- length(x)/Dim
  x <- matrix(x,m,Dim)
  value <- numeric(m)
  for (i in 1:m){
    planet_paras <- list()
    planet_paras[[1]] <- list(tau=x[i,2],K=x[i,3],e=x[i,4],w=x[i,5],M0=x[i,6],gamma=0)
    for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=x[i,1],sigmaJ=x[i,7],
                     cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=planet_paras)
    value[i] <- log_post(for_post)
  }
  if (log==F){
    value <- exp(value)
  }
  return(value)
}

# save(list=c("p1","Dim","num_planets","data_now","tau","alpha","lambda_p","lambda_e","cov_fun","cov_make",
#        "E","M","phi","v",
#        "log_post","log_planet_prior",
#        "Cmax","Pmin","Pmax","K0","Kmax","emax","sigma_e","sigmaJ0","sigmaJmax" ,
#        "log_Cprior","log_Pprior","log_Kprior","log_M0prior","log_eprior","log_wprior","log_sigmaJprior"),
#      file=paste("dataset",dataset_num,"_one_planet_posterior.RData",sep=""))


load(here("data/one_planet_fit_only_3chains_1000grid_1000iters.RData"))
fit_mat <- as.matrix(fit)
data = fit_mat[,1:dim(fit_mat)[2]-1]
for (i in c(3,7)){
  data[,i] <- exp(data[,i])-1
}
i <- 2
data[,i] <- exp(data[,i])
colnames(data) <- c("C","P","K","e","w","M0","sigma")
ranges <- apply(data,2,range)
widths <- abs(apply(ranges,2,diff))
width_factor <- 0 

library(cubature)
numerical_int <- hcubature(p1, lowerLimit=ranges[1,]-width_factor*widths, upperLimit=ranges[2,]+width_factor*widths)

log10(numerical_int$integral)
