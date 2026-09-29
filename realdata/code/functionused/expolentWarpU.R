# given function
# q_density
# Mean_t,Sigma_t,df_t,pro_t
q_density = function(v_w){
  planet_paras <- list()
  if(num_planets == 0){
    planet_paras=c()
    }else{
      for (tmpnum in 1:num_planets) {
        planet_paras[[tmpnum]] <- list(tau=42.4,K=v_w[3],e=v_w[4],w=v_w[5],M0=v_w[6],gamma=0)
      }
      for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=v_w[1],sigmaJ=v_w[7],
                       cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=planet_paras)
    }
  return(log_post(for_post))
}

# WarpU transformation
WarpU_sample = function(w,ComNum,dens){
  mu = dens$parameters$mean[,ComNum]
  S = dens$parameters$variance
  S_use = S$sigma[,,ComNum]
  w_tran = (solve(S_use))%*%(w-mu)
  return(w_tran)
}

# InverseWarpU
InversWarpU_sample = function(w,ComNum){
  mu = dens$parameters$mean[,ComNum]
  S = dens$parameters$variance
  S_use = S$sigma[,,ComNum]
  w_tran = S_use%*%w + mu
  return(w_tran)
}

# conditional probability
get_condition_prob = function(dens,w){
  w = rbind(w,w)
  d1 = predict(dens, x0, what = c("cdens"), logarithm = FALSE)
  phik = d1%*%diag(dens$parameters$pro)
  phik = phik[1,]
  d2 = predict(dens, x0, what = c("dens"), logarithm = FALSE)
  phi_mix = d2[1]
  condition_prob =  phik/phi_mix
  return(condition_prob)
}
# inverse conditional probability
get_inverse_condition_prob = function(dens,w,q_density){
  Num_com = dens$G
  prob_condi = rep(0,Num_com)
  for (k in 1:Num_com) {
    tmpinv_w = InversWarpU_sample(w,k)
    tmpcondi_prob = get_condition_prob(dens,tmpinv_w)
    q_tmpinv = q_density(tmpinv_w)
    tmpfunc = function(tmpw){
      return(InversWarpU_sample(tmpw,k))
    }
    Jacobian_matrix = jacobian(tmpfunc, w)
    Jacobian_det = det(Jacobian_matrix)
    prob_condi[k] = tmpcondi_prob*q_tmpinv*Jacobian_det
  }
  prob_condi = prob_condi/sum(prob_condi)
  return(prob_condi)
}
# phi_mix estimation
get_phi_mix = function(original_sample,com_num){
  dens = densityMclust(data = original_sample,G = com_num)
  return(dens)
}

get_original_sample = function(obsNum,Mean_t,Sigma_t,df_t,pro_t){
  Num_com = length(pro_t)
  variable_dim = length(Mean_t[[1]])
  original_sample = matrix(0,nrow = obsNum,ncol = variable_dim)
  for (i in 1:obsNum) {
    com_num_chosen = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = pro_t)
    tmpsample = rmvt(n=1, sigma = Sigma_t[[com_num_chosen]], df = df_t[com_num_chosen], delta = Mean_t[[com_num_chosen]],type = c("shifted", "Kshirsagar"))
    original_sample[i,] = tmpsample
  }
  return(original_sample)
}

# main function
get_one_sample_update = function(w_pre,dens,q_density){
  Num_com = dens$G
  prob1 = get_condition_prob(dens,w_pre)
  tmpvarphi1 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob1)
  # mean_use = dens$parameters$mean[,com_num_chosen1]
  # Sigma_use = dens$parameters$variance
  # Sigma_use = Sigma_use$sigma[,,com_num_chosen1]
  tmpmiddle = WarpU_sample(w_pre,tmpvarphi1,dens)
  prob2 = get_inverse_condition_prob(dens,tmpmiddle,q_density)
  tmpvarphi2 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob2)
  w_next = InversWarpU_sample(tmpmiddle,tmpvarphi2)
  return(w_next)
}
# get all sample
get_N_sample_update = function(N,w_start,dens,q_density){
  w_all = c()
  w_pre = w_start
  for (i in 1:N) {
    w_next = get_one_sample_update(w_pre,dens,q_density)
    w_pre = w_next
    w_all = c(w_all,w_next)
  }
  return(w_all)
}
