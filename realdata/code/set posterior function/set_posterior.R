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

Dim <- 2
num_planets <- 0

p1 <- function(x,log=F){
  m <- length(x)/Dim
  x <- matrix(x,m,Dim)
  value <- numeric(m)
  for (i in 1:m){
    for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=x[i,1],sigmaJ=x[i,2],
                     cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=c())
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
#      file=paste("dataset",dataset_num,"_no_planet_posterior.RData",sep=""))

p1 <- function(x1,x2,log=F){
  m <- length(x1)
  value <- numeric(m)
  for (i in 1:m){
    for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=x1[i],sigmaJ=x2[i],
                     cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=c())
    value[i] <- log_post(for_post)
  }
  if (log==F){
    value <- exp(value)
  }
  return(value)
}

help(outer)
cvec <- seq(-3,2,0.1)
svec <- seq(0,6,0.1)
grid_values <- outer(cvec,svec,p1)

contour(cvec,svec,grid_values)

library(cubature)
numerical_int <- hcubature(p1, lowerLimit=c(-3,0), upperLimit=c(2,5))

log10(numerical_int$integral)

map <- optim(c(-2,2),p1,log=T,control=list(fnscale=-1,abstol=1e-15))
exp(-487.4545)

library(mvtnorm)



