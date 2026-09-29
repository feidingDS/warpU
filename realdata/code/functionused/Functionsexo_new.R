# given function
# q_density
# Mean_t,Sigma_t,df_t,pro_t
q_density = function(v_w){
  planet_paras <- list()
  if(num_planets == 0){
    planet_paras=c()
  }else{
    for (tmpnum in 1:num_planets) {
      planet_paras[[tmpnum]] <- list(tau=v_w[2],K=v_w[3],e=v_w[4],w=v_w[5],M0=v_w[6],gamma=0)
    }
    for_post <- list(t=data_now[,1],y=data_now[,2],sd=data_now[,3],C=v_w[1],sigmaJ=v_w[7],
                     cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=planet_paras)
  }
  return(exp(log_post(for_post)))
}
# q_density used for bridge sample
q_density_b = function(v_w,data){
  planet_paras <- list()
  if(num_planets == 0){
    planet_paras=c()
  }else{
    for (tmpnum in 1:num_planets) {
      planet_paras[[tmpnum]] <- list(tau=v_w[2],K=v_w[3],e=v_w[4],w=v_w[5],M0=v_w[6],gamma=0)
    }
    for_post <- list(t=data[,1],y=data[,2],sd=data[,3],C=v_w[1],sigmaJ=v_w[7],
                     cov_paras=c(log(tau),log(alpha),log(lambda_e),log(lambda_p)),planet_paras=planet_paras)
  }
  return(log_post(for_post))
}

q_pro_25 = function(w_pre,w_star,alpha,sigma2,sigma_I){
  s = alpha*dmvnorm(w_star,mean= w_pre ,sigma = sigma2*sigma_I) + (1-alpha)*(1/(prod(sample_range[2,] - sample_range[1,])))
  
}

# WarpU transformation
WarpU_sample = function(w,ComNum,dens){
  mu = dens$parameters$mean[,ComNum]
  S = dens$parameters$variance
  S_use = S$sigma[,,ComNum]
  # add
  #choS = chol(S_use)
  #S_use = t(choS)
  ##
  # change using eigen
  Eigen_S = eigen(S_use)
  S_use = Eigen_S$vectors %*% sqrt(diag(Eigen_S$values))
  ##
  #
  w_tran = (solve(S_use))%*%(w-mu)
  return(w_tran)
}

# InverseWarpU
InversWarpU_sample = function(w,ComNum,dens){
  mu = dens$parameters$mean[,ComNum]
  S = dens$parameters$variance
  S_use = S$sigma[,,ComNum]
  # add
  #choS = chol(S_use)
  #S_use = t(choS)
  # change using eigen
  Eigen_S = eigen(S_use)
  S_use = Eigen_S$vectors %*% sqrt(diag(Eigen_S$values))
  ##
  w_tran = S_use%*%w + mu
  #w_tran = S_use%*%w + mu
  return(w_tran)
}

# conditional probability
get_condition_prob = function(dens,w){
  #w = rbind(w,w)
  tmpw = matrix(w,nrow = 1)
  d1 = predict(dens, tmpw, what = c("cdens"), logarithm = FALSE)
  phik = d1%*%diag(dens$parameters$pro)
  phik = phik[1,]
  d2 = predict(dens, tmpw, what = c("dens"), logarithm = FALSE)
  phi_mix = d2[1]
  if (phi_mix == 0) {
    condition_prob = rep(1/dens$G,dens$G)
  }else{
    condition_prob =  phik/phi_mix
  }
  if(sum(is.na(condition_prob)) != 0){
    condition_prob = rep(1/dens$G,dens$G)
    print('flag')
  }

  return(condition_prob)
}

get_condition_prob_change = function(dens,w){
  #w = rbind(w,w)
  tmpw = matrix(w,nrow = 1)
  #print("1")
  d1 = predict(dens, tmpw, what = c("cdens"), logarithm = FALSE)
  #print("2")
  phik = d1%*%diag(dens$parameters$pro)
  #print('4')
  phik = phik[1,]
  #print('3')
  d2 = q_density(tmpw)
  #print('d2')
  #print(d2)
  phi_mix = d2
  if (phi_mix == 0 | (sum(phik) == 0) ) {
    condition_prob = rep(1/dens$G,dens$G)
  }else{
    condition_prob =  phik/phi_mix
    #print('phik')
    #print(phik)
    condition_prob = condition_prob/sum(condition_prob)
    
  }
  if(sum(is.na(condition_prob)) != 0){
    condition_prob = rep(1/dens$G,dens$G)
    print('flag')
  }
  
  return(condition_prob)
}
# inverse conditional probability
get_inverse_condition_prob = function(dens,w,q_density){
  Num_com = dens$G
  prob_condi = rep(0,Num_com)
  for (k in 1:Num_com) {
    tmpinv_w = InversWarpU_sample(w,k,dens)
    tmpcondi_prob = get_condition_prob(dens,tmpinv_w)
    tmpcondi_prob = tmpcondi_prob[k]
    q_tmpinv = q_density(tmpinv_w)
    tmpfunc = function(tmpw){
      return(InversWarpU_sample(tmpw,k,dens))
    }
    Jacobian_matrix = jacobian(tmpfunc, w)
    Jacobian_det = abs(det(Jacobian_matrix))
    #print('q_tmpinv')
    #print(q_tmpinv)
    #print('Jacobian_det')
    #print(Jacobian_det)
    #print('tmpcondi_prob')
    #print(tmpcondi_prob)
    prob_condi[k] = tmpcondi_prob*q_tmpinv*Jacobian_det
  }
  #print('prob_condi')
  #print(prob_condi)
  prob_condi = prob_condi/sum(prob_condi)
  if(sum(is.na(prob_condi)) != 0){
    prob_condi = rep(1/dens$G,dens$G)
    #print('flag111')
  }
  return(prob_condi)
}
# phi_mix estimation
get_phi_mix = function(original_sample,com_num){
  #dens = densityMclust(data = original_sample,G = com_num,modelNames = "VVV")
  tryCatch(
    {
      dens = densityMclust(data = original_sample,G = com_num)
      return(dens)
    },error = function(e){
      dens = densityMclust(data = original_sample)
      return(dens)
    }
  )
  
  #dens = densityMclust(data = original_sample)
  
}

get_phi_mix_evi = function(original_sample,com_num){
  testa = emControl()
  testa$equalPro = TRUE
  #dens = densityMclust(data = original_sample,G = com_num,modelNames = "EVI")
  dens = densityMclust(data = original_sample,G = com_num,control = testa)
  ##dens = densityMclust(data = original_sample,G = com_num)
  #dens = densityMclust(data = original_sample)
  plot(dens$parameters$mean[1,])
  plot(dens$parameters$pro)
  return(dens)
}
get_phi_mix_1 = function(original_sample,com_num){
  dens = densityMclust(data = original_sample,G = com_num,modelNames = "E")
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
get_one_sample_update = function(w_pre,dens,q_density,inserversion = NULL,alpha = NULL,sigma2 = NULL,sigma_I = NULL,skip_MH = NULL){
  Num_com = dens$G
  p_a_r = 0
  if(is.null(sigma_I)){

    sigma_I = diag(length(w_pre))
  }
  if(inserversion == 223){
    w_star = mvrnorm(n = 1,mu = w_pre,Sigma = sigma2*sigma_I)
    #print('w_pre')
    #print(w_pre)
    #print('w_star')
    #print(w_star)
    
    #
    t_num = dmvnorm(t(w_pre),mean= w_star ,sigma = sigma2*sigma_I)
    t_den = dmvnorm(t(w_star),mean= t(w_pre),sigma = sigma2*sigma_I)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    # print('t_num')
    # print(t_num)
    # print('t_den')
    # print(t_den)
    # print('qden_num')
    # print(qden_num)
    # print('qden_den')
    # print(qden_den)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    # print('p_accept')
    # print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      p_a_r = 1
      w_pre = w_star}
  }
  if(inserversion == 00){
    w_star = mvrnorm(n = 1,mu = w_pre,Sigma = sigma2*sigma_I)
    #print('w_pre')
    #print(w_pre)
    #print('w_star')
    #print(w_star)
    
    #
    t_num = dmvnorm(t(w_pre),mean= w_star ,sigma = sigma2*sigma_I)
    t_den = dmvnorm(t(w_star),mean= t(w_pre),sigma = sigma2*sigma_I)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    # print('t_num')
    # print(t_num)
    # print('t_den')
    # print(t_den)
    # print('qden_num')
    # print(qden_num)
    # print('qden_den')
    # print(qden_den)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    # print('p_accept')
    # print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      p_a_r = 1
      w_pre = w_star}
    ## add the MH above
    last_stage_MCMC1 = Stand_MCMC(Num_sample = 1, proposal_density = proposal_density, proposal_sample_fun = proposal_sample_fun, w_star = w_pre,Dim_w = length(w_pre),dens = dens )
    last_stage_MCMC = last_stage_MCMC1$sample_get
  }
  

  if(inserversion == 1){
    w_pre = mvrnorm(n = 1,w_pre,0.1*diag(length(w_pre)))
  }
  prob1 = get_condition_prob(dens,w_pre)
  tmpvarphi1 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob1)
  #print('prob1')
  #print(prob1)
  #print('tmpvarphi1')
  #print(tmpvarphi1)
  
  # mean_use = dens$parameters$mean[,com_num_chosen1]
  # Sigma_use = dens$parameters$variance
  # Sigma_use = Sigma_use$sigma[,,com_num_chosen1]
  if(inserversion == 2){
    mean_cur = dens$parameters$mean[,tmpvarphi1]
    Sigma_cur = dens$parameters$variance
    Sigma_cur = Sigma_cur$sigma[,,tmpvarphi1]
    w_pre = mvrnorm(n = 1,mean_cur,Sigma_cur)
  }
  if(inserversion == 3){
    ## MH sampling
    mean_cur_gen = dens$parameters$mean[,tmpvarphi1]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi1]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    t_num = prop_density(w_t = w_star,w_new = w_pre)
    t_den = prop_density(w_t = w_pre,w_new = w_star)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      w_pre = w_star}
    w_pre1 = w_pre
    index_store =tmpvarphi1
  }
  
  #
  if(inserversion == 8){
    prob4 = get_condition_prob(dens,w_pre)
    #print('prob4')
    #print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    #print('tmpvarphi4')
    #print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    w_pre = w_star
  }
  if(inserversion == 9){
    print('zz')
    prob4 = get_condition_prob_change(dens,w_pre)
    print('prob4')
    print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    print('tmpvarphi4')
    print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    w_pre = w_star
  }
  if(inserversion == 10){
    prob4 = get_condition_prob(dens,w_pre)
    print('prob4')
    print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    print('tmpvarphi4')
    print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    w_pre = w_star
    w_pre1 = w_pre
  }
  
  if(inserversion == 11){
    prob4 = get_condition_prob(dens,w_pre)
    print('prob4')
    print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    print('tmpvarphi4')
    print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    w_pre = w_star
    w_pre1 = w_star
  }
  if(inserversion == 12){
    # gibbs + warpU
    # changed gibbs only
    
    #print('zz')
    prob4 = get_condition_prob_change(dens,w_pre)
    #print('prob4')
    #print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    #print('tmpvarphi4')
    #print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    w_pre = w_star # for warp U
    index_store =tmpvarphi4
  }
  
  # for normal proposal
  if(inserversion == 23){
    # gibbs + warpU
    # changed gibbs only
    if(is.null(skip_MH)){
      #print('23')
      w_star = mvrnorm(n = 1,mu = w_pre,Sigma = sigma2*sigma_I)
      #print('w_pre')
      #print(w_pre)
      #print('w_star')
      #print(w_star)
      
      #
      t_num = mclust::dmvnorm(t(w_pre),mean= w_star ,sigma = sigma2*sigma_I)
      t_den = mclust::dmvnorm(t(w_star),mean= t(w_pre),sigma = sigma2*sigma_I)
      qden_num = q_density(w_star)
      qden_den = q_density(w_pre)
      # print('t_num')
      # print(t_num)
      # print('t_den')
      # print(t_den)
      # print('qden_num')
      # print(qden_num)
      # print('qden_den')
      # print(qden_den)
      tmp_accept = (t_num*qden_num)/(t_den*qden_den)
      p_accept = min(tmp_accept,1)
      # print('p_accept')
      # print(p_accept)
      tmp_compare = runif(1,min = 0,max = 1)
      if (tmp_compare <= p_accept) {
        p_a_r = 1
        w_pre = w_star}
      
    }
  }
  # alpha*normal + (1-alpha)*uniform
  if(inserversion == 25){
    # gibbs + warpU
    # changed gibbs only
    norm_unif_index = sample(1:2,size = 1,prob = c(alpha,(1-alpha)))
    if(norm_unif_index == 1){
      w_star = mvrnorm(n = 1,w_pre,Sigma = sigma2*sigma_I)

    }else{
      w_star = c()
      for (unif_i in 1:length(w_pre)) {
        tmp_new_star = runif(n = 1,min = sample_range[1,unif_i],max = sample_range[2,unif_i])
        w_star = c(w_star,tmp_new_star)
      }
    }
    #
    t_num = q_pro_25(w_pre =w_star ,w_star = t(w_pre)  ,alpha,sigma2,sigma_I)
    t_den = q_pro_25(w_pre =t(w_pre) ,w_star = w_star  ,alpha,sigma2,sigma_I)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    # print('p_accept')
    # print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      p_a_r = 1
      w_pre = w_star}
  }
  #
  ## for Normal + Uniform only
  if(inserversion == 24){
    # gibbs + warpU
    # changed gibbs only
    norm_unif_index = sample(1:2,size = 1,prob = c(alpha,(1-alpha)))
    if(norm_unif_index == 1){
      w_star = mvrnorm(n = 1,w_pre,Sigma = sigma2*sigma_I)
      
    }else{
      w_star = c()
      for (unif_i in 1:length(w_pre)) {
        tmp_new_star = runif(n = 1,min = sample_range[1,unif_i],max = sample_range[2,unif_i])
        w_star = c(w_star,tmp_new_star)
      }
    }
    #
    t_num = q_pro_25(w_pre =w_star ,w_star = t(w_pre)  ,alpha,sigma2,sigma_I)
    t_den = q_pro_25(w_pre =t(w_pre) ,w_star = w_star  ,alpha,sigma2,sigma_I)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    # print('p_accept')
    # print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      p_a_r = 1
      w_pre = w_star}
    w_pre1 = w_pre
  }
  #
  
  if(inserversion == 13){
    # gibbs only
    # changed gibbs only
    print('zz')
    prob4 = get_condition_prob_change(dens,w_pre)
    print('prob4')
    print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    print('tmpvarphi4')
    print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    w_pre1 = w_star # for only
    index_store =tmpvarphi4
  }
  # only for inserversion 3
  prob1 = get_condition_prob_change(dens,w_pre)
  tmpvarphi1 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob1)
  #
  tmpmiddle = WarpU_sample(w_pre,tmpvarphi1,dens)
  if(inserversion == 4){
    mean_cur = dens$parameters$mean[,tmpvarphi1]
    Sigma_cur = dens$parameters$variance
    Sigma_cur = Sigma_cur$sigma[,,tmpvarphi1]
    tmpmiddle = mvrnorm(n = 1,tmpmiddle,Sigma_cur)
  }
  
  if(inserversion == 5){
    
    prob1 = get_condition_prob(dens,tmpmiddle)
    print('prob2')
    print(prob1)
    tmpvarphi1 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob1)
    print('tmpvarphi2')
    print(tmpvarphi1)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi1]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi1]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    print('w_star')
    print(w_star)
    t_num = prop_density(w_t = w_star,w_new = w_pre)
    t_den = prop_density(w_t = w_pre,w_new = w_star)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    print('p_accept')
    print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      tmpmiddle = w_star}
  }
  
  if(inserversion == 6){
    prob4 = get_condition_prob(dens,tmpmiddle)
    print('prob4')
    print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    print('tmpvarphi4')
    print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    tmpmiddle = w_star
  }
  if(inserversion == 7){
    print('zz')
    prob4 = get_condition_prob_change(dens,tmpmiddle)
    print('prob4')
    print(prob4)
    tmpvarphi4 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob4)
    print('tmpvarphi4')
    print(tmpvarphi4)
    mean_cur_gen = dens$parameters$mean[,tmpvarphi4]
    Sigma_cur_gen = dens$parameters$variance
    Sigma_cur_gen = Sigma_cur_gen$sigma[,,tmpvarphi4]
    w_star = mvrnorm(n = 1,mean_cur_gen,Sigma_cur_gen)
    tmpmiddle = w_star
  }
  
  prob2 = get_inverse_condition_prob(dens,tmpmiddle,q_density)

  tmpvarphi2 = sample(x =1:Num_com ,size = 1,replace = TRUE,prob = prob2)
  #print('tmpvarphi2')
  #print(tmpvarphi2)
  w_next = InversWarpU_sample(tmpmiddle,tmpvarphi2,dens)
  if(inserversion == 13){
    w_next = w_pre1 # for gibbs only
  }
  if(inserversion == 24){
    w_next = w_pre1 # for gibbs only
  }
  
  if((inserversion != 12) & (inserversion != 13)){
    index_store = 0
  }
  if((inserversion == 23) ||(inserversion == 24)||(inserversion == 25)){
    
    
  }
  if(inserversion == 00){
    w_next = as.vector(last_stage_MCMC)
  }
  if(inserversion == 223){
    w_next = as.vector(w_pre)
  }
  
  tima_list = list()
  tima_list$w_next = w_next
  tima_list$indexcom = tmpvarphi1
  tima_list$inv_indexcom = tmpvarphi2
  tima_list$index_store = index_store
  tima_list$p_a_r =p_a_r 
  if(inserversion == 00){
    tima_list$MCMC_pro_cur = last_stage_MCMC1$proposal_value_cur
    tima_list$MCMC_pro_next = last_stage_MCMC1$proposal_value_next
    tima_list$MCMC_reject = last_stage_MCMC1$probability_rej
  }
  return(tima_list)
}
# get all sample
get_N_sample_update = function(N,w_start,dens,q_density,inserversion = NULL,jumpsteps = NULL,alpha = NULL,sigma2 = NULL,sigma_I = NULL,skip_MH = NULL){
  print("dens$G")
  print(dens$G)
  w_all = c()
  MCMC_rej = c()
  MCMC_pro_cur = c()
  MCMC_pro_next = c()
  index_forward = c()
  index_back = c()
  index_gibbs = c()
  change_forward = c()
  change_back = c()
  change_gibbs = c()
  w_pre = w_start
  p_acept = 0
  for (i in 1:N) {
    #print('i=')
    #print(i)
    ##
    if(i%%100 == 0){
      print("sampleNum")
      print(i)
    }
    ##
    if((inserversion !=25) &&(inserversion !=24) && ((inserversion !=23)) &&((inserversion !=00)) &&((inserversion !=223)) ){
      tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion)
    }else{
      if(!is.null(skip_MH)){
        if((i%%skip_MH) != 0){
          if(!is.null(sigma_I)){
            tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I,skip_MH = 1)
          }else{
            tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2,skip_MH = 1)
          }
        }else{
          if(!is.null(sigma_I)){
            tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
          }else{
            tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2)
          }
          
        }
        
      }else{
        if(!is.null(sigma_I)){
          tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
        }else{
          tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2)
          
        }
      }

    }
    
    w_next = tmp_w_next$w_next
    if(i>1){
      if((tmp_w_next$indexcom == 2)&&(index_forward[length(index_forward)] == 1)){
        change_forward = c(change_forward,1)
      }
      if((tmp_w_next$indexcom == 1)&&(index_forward[length(index_forward)] == 2)){
        change_forward = c(change_forward,2)
      }
      if((tmp_w_next$indexcom == 1)&&(index_forward[length(index_forward)] == 1)){
        change_forward = c(change_forward,3)
      }
      if((tmp_w_next$indexcom == 2)&&(index_forward[length(index_forward)] == 2)){
        change_forward = c(change_forward,4)
      }
      # back
      if((tmp_w_next$inv_indexcom == 2)&&(index_back[length(index_back)] == 1)){
        change_back = c(change_back,1)
      }
      if((tmp_w_next$inv_indexcom == 1)&&(index_back[length(index_back)] == 2)){
        change_back = c(change_back,2)
      }
      if((tmp_w_next$inv_indexcom == 1)&&(index_back[length(index_back)] == 1)){
        change_back = c(change_back,3)
      }
      if((tmp_w_next$inv_indexcom == 2)&&(index_back[length(index_back)] == 2)){
        change_back = c(change_back,4)
      }
      # gibbs
      if((tmp_w_next$index_store == 2)&&(index_gibbs[length(index_gibbs)] == 1)){
        change_gibbs = c(change_gibbs,1)
      }
      if((tmp_w_next$index_store == 1)&&(index_gibbs[length(index_gibbs)] == 2)){
        change_gibbs = c(change_gibbs,2)
      }
      if((tmp_w_next$index_store == 1)&&(index_gibbs[length(index_gibbs)] == 1)){
        change_gibbs = c(change_gibbs,3)
      }
      if((tmp_w_next$index_store == 2)&&(index_gibbs[length(index_gibbs)] == 2)){
        change_gibbs = c(change_gibbs,4)
      }
    }
    index_forward = c(index_forward,tmp_w_next$indexcom)
    index_back = c(index_back,tmp_w_next$inv_indexcom)
    index_gibbs = c(index_gibbs,tmp_w_next$index_store)
    if(tmp_w_next$p_a_r ==1 ){
      p_acept = p_acept +1
    }
     

    w_pre = w_next
    if(!is.null(jumpsteps)){
      if( (i%%jumpsteps) == 0){
        w_all = rbind(w_all,t(w_next))
        MCMC_pro_cur = rbind(MCMC_pro_cur, tmp_w_next$MCMC_pro_cur )
        MCMC_pro_next = rbind(MCMC_pro_next,tmp_w_next$MCMC_pro_next)
        MCMC_rej = rbind(MCMC_rej, tmp_w_next$MCMC_reject)
      }
    }else{
      w_all = rbind(w_all,t(w_next))
      MCMC_pro_cur = rbind(MCMC_pro_cur, tmp_w_next$MCMC_pro_cur )
      MCMC_pro_next = rbind(MCMC_pro_next,tmp_w_next$MCMC_pro_next)
      MCMC_rej = rbind(MCMC_rej, tmp_w_next$MCMC_reject)
    }
  }
  p_acept = p_acept/N
  zzz = list()
  zzz$w_all = w_all
  zzz$index_forward = index_forward
  zzz$index_back = index_back
  zzz$index_gibbs = index_gibbs
  zzz$change_forward = change_forward
  zzz$change_back = change_back
  zzz$change_gibbs = change_gibbs
  zzz$p_acept= p_acept
  zzz$MCMC_pro_cur = MCMC_pro_cur
  zzz$MCMC_pro_next = MCMC_pro_next
  zzz$MCMC_rej = MCMC_rej
  return(zzz)
}
# plot function
# get_density_plot = function(x_min,x_max,y_min,y_max,samplesize_plot){
get_density_plot = function(original_sample,sample_result){
  original_sample = data.frame(original_sample)
  sample_result = data.frame(sample_result)
  p = ggplot(data = original_sample, aes(x=original_sample[,1], y=original_sample[,2]) ) +
    geom_density_2d() + geom_point(data = sample_result,aes(x=sample_result[,1], y=sample_result[,2]),size = 0.1)
  return(p)
}

# get the plus(w_1 + w_2) density plot and histgram of the sample
# get_density_plot = function(x_min,x_max,y_min,y_max,samplesize_plot){
get_plus_plot = function(original_sample,sample_result){
  sample_result = rowSums(sample_result)
  #scale_coe = length(sample_result)
  sample_result = data.frame(sample_result)
  sample_result = sample_result
  original_sample = rowSums(original_sample)
  original_sample = data.frame(original_sample)
  p = ggplot(data = original_sample, aes(x=original_sample,y=..density..) ) +
    geom_histogram(data = sample_result,aes(x=sample_result,y=..density..),bins = 25)+
    geom_density(data = original_sample, aes(x=original_sample,y=..density..),position="identity",color = 'red')
  return(p)
}
  
  
  
## proposal density
prop_density = function(w_t,w_new){
  tmptol = dens$G
  p_phi_w = get_condition_prob(dens,w_t)
  s = 0
  for (i in 1:tmptol) {
    mean_cur = dens$parameters$mean[,i]
    Sigma_cur = dens$parameters$variance
    Sigma_cur = Sigma_cur$sigma[,,i]
    w_new_vec = as.vector(w_new)
    f_new = dmvnorm(w_new_vec,mean= mean_cur,sigma = Sigma_cur)
    s = s + p_phi_w[i]*f_new
  }
  return(s)
}


# adaptive warp U 
# sample_range: each col is the range of one dimension
# [,1] [,2]
# [1,]    0    2
# [2,]    1    3

adaptive_warpU = function(com_num,N, initial_data = NULL, True_Data = NULL, inserversion_num, stage_num, sample_range = NULL,density_num = NULL,alpha = NULL,sigma2 = NULL,sigma_I = NULL,star_w = NULL,True_com = NULL,mean_dis_true = NULL,pro_dis_true = NULL,store_sample_dim = NULL,Pre_stage_form = NULL, maxupdate_stage = NULL, all_pre_flag = NULL,q_density = q_density, plotfirstflag = NULL,probability_flag = TRUE){
  K_prop = com_num
  if(is.null(initial_data)){
    dim_num = dim(sample_range)[2]
    initial_data = matrix(0,nrow = N,ncol = dim_num)

    for (i in 1:dim_num) {
      tmpx = runif(n = N,min = sample_range[1,i],max = sample_range[2,i])
      initial_data[,i] = tmpx
    }
  }
  data = initial_data
  start_last_sample = data[N,]
  store_sample_stage = c()
  store_sample_stage_all = c()
  if(!is.null(store_sample_dim)){
    store_sample_stage = cbind(store_sample_stage,data[,store_sample_dim])
  }
  # if(!is.null(Pre_stage_form)){
  #   store_sample_stage_all = rbind(store_sample_stage_all,data)
  # }
  Dim_use = dim(data)[2]
  distance = list(NULL)
  distance_close = list(NULL)
  distance_mean = list(NULL)
  distance_var = list(NULL)
  
  distance_kl = list(NULL)
  distance_wass = list(NULL)
  length(distance_kl) = Dim_use
  length(distance_wass) = Dim_use
  length(distance) = Dim_use
  length(distance_mean) = Dim_use
  length(distance_var) = Dim_use
  length(distance_close) = Dim_use
  distance_close_dif = list(NULL)
  length(distance_close_dif) = Dim_use
  index_distance = (ceiling((dim(data)[1])/2)+1) : (dim(data)[1])
  pro_tmp_mat = c()
  mean_tmp_mat = c()
  baseline_mean_dis = c()
  baseline_pro_dis = c()
  for ( i in 1:Dim_use) {
    if(!is.null(True_com)){
      True_com_dens = get_phi_mix_1(data[index_distance,i],True_com[i])
      mean_dis_com = True_com_dens$parameters$mean
      pro_dis_com = True_com_dens$parameters$pro
      mean_tmp_mat = c(mean_tmp_mat,get_low_distance(w_1 = mean_dis_com,w_2 = mean_dis_true[[i]],prop_vec = pro_dis_true[[i]]))
      pro_tmp_mat = c(pro_tmp_mat,get_low_distance(w_1 = pro_dis_com,w_2 = pro_dis_true[[i]]))
      if(!is.null(True_Data)){
        
        baseline_True_com_dens = get_phi_mix_1(True_Data[index_distance,i],True_com[i])
        baseline_mean = baseline_True_com_dens$parameters$mean
        baseline_pro = baseline_True_com_dens$parameters$pro
        baseline_mean_dis = c(baseline_mean_dis,get_low_distance(w_1 = baseline_mean,w_2 = mean_dis_true[[i]],prop_vec = pro_dis_true[[i]]))
        baseline_pro_dis = c(baseline_pro_dis,get_low_distance(w_1 = baseline_pro,w_2 = pro_dis_true[[i]]))
        
      }
      tmp_data_index = data[index_distance,]
      distance_kl[[i]] = get_Wasserstein(sample1 = tmp_data_index,sample2 = True_Data,dim_num = i)$kl
      distance_wass[[i]] = get_Wasserstein(sample1 = tmp_data_index,sample2 = True_Data,dim_num = i)$wasserstein
  
      
    }
    #distance[[i]] = get_Wasserstein(sample1 = data,sample2 = True_Data,dim_num = i)

    
    mean_dif = mean(data[index_distance,i])-  mu2[i]
    var_dif = var(data[index_distance,i]) - Sigma2[i,i]
    distance_mean[[i]] = mean_dif
    distance_var[[i]] = var_dif
    #mu_dif = mu2[i]
    if(density_num ==1){
      com_dens = densityMclust(data = data[index_distance,i],G = 1)
      Sigma1 = com_dens$parameters$variance$sigmasq
      mu1 = com_dens$parameters$mean
      distance_close[[i]] = get_kl_multinormal(mu1 = mu1,mu2 = mu2[i],Sigma1 = Sigma1,Sigma2 = Sigma2[i,i])
      mu11 = mean(data[index_distance,i])
      Sigma11 = var(data[index_distance,i])
      distance_close_dif[[i]] = get_kl_multinormal(mu1 = mu11,mu2 = mu2[i],Sigma1 = Sigma11,Sigma2 = Sigma2[i,i])
      
    }
  }
  # if(Dim_use != 1){
  #   distance_join = get_Wasserstein(sample1 = data,sample2 = True_Data)
  # }
  # the main steps for adaptive warpU sampling
  mean_mat = mean_tmp_mat
  pro_mat = pro_tmp_mat
  
  #w_start = data[1,]
  if(is.null(star_w)){
    #w_start = dens$parameters$mean[,1]
    w_start = start_last_sample
  }else{
    w_start = star_w
  }
  
  for (i in 1:stage_num) {
    print('i')
    print(i)
    if(is.null(maxupdate_stage)){
      if(probability_flag){
        prob_update_index = sample(1:2,size = 1,replace = FALSE, prob = c(exp(-i^(1/8)+1),(1-exp(-i^(1/8)+1))) )
        if(prob_update_index == 1){
          if(i == 1){
            dens = get_phi_mix_evi(data,com_num)

          }else{
            dens = get_phi_mix(data,com_num)
          }
        }
      }else{
        if(i == 1){
          dens = get_phi_mix_evi(data,com_num)
        }else{
          dens = get_phi_mix(data,com_num)
        }
        
      }
    }else{
      if(i <= maxupdate_stage){
        # dens = get_phi_mix(data,com_num)
        if(probability_flag){
          prob_update_index = sample(1:2,size = 1,replace = FALSE, prob = c(exp(-i^(1/8)+1),(1-exp(-i^(1/8)+1))) )
          if(prob_update_index == 1){
            if(i == 1){
              dens = get_phi_mix_evi(data,com_num)
            }else{
              dens = get_phi_mix(data,com_num)
            }
          }
        }else{
          if(i == 1){
            dens = get_phi_mix_evi(data,com_num)
          }else{
            dens = get_phi_mix(data,com_num)
          }
        }
      }
      }
    

    
    # plot(dens)
    # plot the density
    if(!is.null(plotfirstflag)){
      if( i == 1){
        pdf(file = paste0('./Plot_stored/Firststage_sigma2_',sigma2,'Largestlayer_',Layer_num,'N_',N,'dim_',dim(sample_range)[2],'initial_type_',initial_type,'plot.pdf'),height = 5,width = 8)
        print(plot(dens, what = "density"))
        dev.off()
      }
    }
    # 2
    # 0
    if((inserversion_num !=25 )&& (inserversion_num !=24) && (inserversion_num !=23)&&(inserversion_num !=00)){
      sample_result = get_N_sample_update (N,w_start,dens,q_density,inserversion = inserversion_num)
    }else{
      if(i<= 2){
        sample_result = get_N_sample_update ((N*3),w_start,dens,q_density,inserversion = inserversion_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
      }else{
        sample_result = get_N_sample_update (N,w_start,dens,q_density,inserversion = inserversion_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
      }
      
      
    }
    data = sample_result$w_all
    ## Jan 21 2022, start_last_sample
    ##
    start_last_sample = sample_result$w_all[dim(sample_result$w_all)[1],]
    w_start = start_last_sample
    ##
    # store the special sample

    if(!is.null(Pre_stage_form)){
      if(i > 3){
        store_sample_stage_all = rbind(store_sample_stage_all,data)
        index_pre = sample(1:(dim(store_sample_stage_all)[1]),size = N,replace = FALSE)
        if(!is.null(all_pre_flag)){
          index_pre = 1:(dim(store_sample_stage_all)[1])
        }
        data  = store_sample_stage_all[index_pre, ]
      }
    }
    if(!is.null(store_sample_dim)){
      #store_sample_stage = cbind(store_sample_stage,data[,store_sample_dim])
      store_sample_stage = cbind(store_sample_stage,sample_result$w_all[,store_sample_dim])
      
    }
    index_distance = (ceiling((dim(data)[1])/2)) : (dim(data)[1])
    pro_tmp_mat = c()
    mean_tmp_mat = c()
    ## Jan20 2022 comment below for Error in switch(EXPR = modelName, X = "E", XII = "EII", XXI = "EEI", XXX = "EEE",  : EXPR must be a length 1 vector
    ###
    ###
    # for (j in 1:Dim_use) {
    #   
    #   if(!is.null(True_com)){
    #     True_com_dens = get_phi_mix_1(data[index_distance,j],True_com[j])
    #     mean_dis_com = True_com_dens$parameters$mean
    #     pro_dis_com = True_com_dens$parameters$pro
    #     mean_tmp_mat = c(mean_tmp_mat,get_low_distance(w_1 = mean_dis_com,w_2 = mean_dis_true[[j]],prop_vec = pro_dis_true[[j]]))
    #     pro_tmp_mat = c(pro_tmp_mat,get_low_distance(w_1 = pro_dis_com,w_2 = pro_dis_true[[j]]))
    #     
    #     
    #     #
    #     #distance[[j]] = c( distance[[j]],get_Wasserstein(sample1 = data,sample2 = True_Data,dim_num = j))
    #     distance_kl[[j]] = c( distance_kl[[j]],get_Wasserstein(sample1 = data[index_distance,],sample2 = True_Data,dim_num = j)$kl)
    #     distance_wass[[j]] = c( distance_wass[[j]],get_Wasserstein(sample1 = data[index_distance,],sample2 = True_Data,dim_num = j)$wasserstein)
    #   }
    #   
    # 
    #   
    #   mean_dif = mean(data[index_distance,j])-  mu2[j]
    #   var_dif = var(data[index_distance,j]) - Sigma2[j,j]
    #   # distance_mean[[j]] = mean_dif
    #   # distance_var[[j]] = var_dif
    #   distance_mean[[j]] = c( distance_mean[[j]],mean_dif)
    #   distance_var[[j]] = c( distance_var[[j]],var_dif)
    #   
    #   if(density_num == 1){
    #     com_dens = densityMclust(data = data[index_distance,j],G = 1)
    #     Sigma1 = com_dens$parameters$variance$sigmasq
    #     mu1 = com_dens$parameters$mean
    #     distance_close[[j]] = c(distance_close[[j]],get_kl_multinormal(mu1 = mu1,mu2 = mu2[j],Sigma1 = Sigma1,Sigma2 = Sigma2[j,j]))
    #     mu11 = mean(data[index_distance,j])
    #     Sigma11 = var(data[index_distance,j])
    #     distance_close_dif[[j]] = c(distance_close_dif[[j]],get_kl_multinormal(mu1 = mu11,mu2 = mu2[j],Sigma1 = Sigma11,Sigma2 = Sigma2[j,j]))
    #     
    #   }
    #   
    # 
    # 
    # }
    # ###
    # ###
    # ###
    mean_mat = rbind(mean_mat,mean_tmp_mat)
    pro_mat = rbind(pro_mat,pro_tmp_mat)
    # if(Dim_use != 1){
    #   distance_join = c (distance_join, get_Wasserstein(sample1 = data,sample2 = True_Data))
    # }
    
  }
  
  sample_result$dens = dens
  sample_result$distance_kl = distance_kl
  sample_result$distance_wass = distance_wass
  sample_result$distance_mean= distance_mean
  sample_result$distance_var = distance_var
  sample_result$p_acept = sample_result$p_acept
  if(!is.null(store_sample_dim)){
    sample_result$store_sample_stage = store_sample_stage
  }
  if(density_num ==1){
    sample_result$distance_close = distance_close
    sample_result$distance_close_dif = distance_close_dif
    
  }
  if(!is.null(True_com)){
    sample_result$mean_mat = mean_mat
    sample_result$pro_mat = pro_mat
    if(!is.null(True_Data)){
      sample_result$baseline_mean_dis = baseline_mean_dis
      sample_result$baseline_pro_dis = baseline_pro_dis
    }
    
  }
  # if(Dim_use != 1){
  #   sample_result$distance_join = distance_join
  #   }
  return(sample_result)
}


# compute the Wasserstein distance 
get_Wasserstein = function(sample1,sample2,dim_num = NULL){
  if(!is.null(dim_num)){
    sample1 = sample1[,dim_num]
    sample2 = sample2[,dim_num]
    distance = wasserstein1d(sample1,sample2)
    distance_kl = median(KLx.divergence(sample1, sample2, k = 100))
  }else{
    print(1)
    min_N = min(dim(sample1)[1],dim(sample2)[1])
    print(1)
    sample1 = sample1[1:min_N,]
    print(2)
    sample2 = sample2[1:min_N,]
    print(3)
    sample1 = pp(sample1)
    print(4)
    sample2 = pp(sample2)
    print(5)
    # print.default(sample2)
    distance = wasserstein(sample1,sample2)
    print(6)
    print(1)
  }
  distance_all = list()
  distance_all$kl = distance_kl
  distance_all$wasserstein = distance
  return(distance_all)
}


# get the di

# closed form for kl divergence of two multinormal distributions
get_kl_multinormal = function(mu1,mu2,Sigma1,Sigma2){
  d = length(mu1)
  s = 1/2*log((Sigma2)/(Sigma1)) + 1/2*(sum(diag(solve(Sigma2)%*%Sigma1)))+
    1/2*t(mu1-mu2)%*%(solve(Sigma2))%*%(mu1-mu2) - d/2
  return(s)
}



# last stage distance changing with number of sample
distance_last_stage = function(com_num,N,inserversion_num, stage_num, sample_range,True_Data,density_num = 1,last_num,results_sample,alpha = NULL,sigma2 = NULL){
  Dim_use = dim(True_Data)[2]
  distance_change_kl = list(NULL)
  length(distance_change_kl) = Dim_use
  distance_change_wass = list(NULL)
  length(distance_change_wass) = Dim_use
  distance_change_kl_close = list(NULL)
  length(distance_change_kl_close) = Dim_use
  distance_change_kl_close_dif = list(NULL)
  length(distance_change_kl_close_dif) = Dim_use
  distance_change_mean_dif = list(NULL)
  length(distance_change_mean_dif) = Dim_use
  distance_change_var_dif = list(NULL)
  length(distance_change_var_dif) = Dim_use
  for(i in last_num){
    if((inserversion_num !=25) &&(inserversion_num !=24) && (inserversion_num !=23)){
      results_sample1 = adaptive_warpU(initial_data = results_sample$w_all,com_num = com_num,N = i,inserversion_num = inserversion_num, stage_num=1, sample_range = sample_range,True_Data = True_Data,density_num = density_num)
    }else{
      results_sample1 = adaptive_warpU(initial_data = results_sample$w_all,com_num = com_num,N = i,inserversion_num = inserversion_num, stage_num=1, sample_range = sample_range,True_Data = True_Data,density_num = density_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
    }
    for (j in 1:Dim_use) {
      if(density_num ==1){
        distance_change_kl_close[[j]] =c(distance_change_kl_close[[j]],(results_sample1$distance_close[[j]][length(results_sample1$distance_close[[j]])]))
        distance_change_kl_close_dif[[j]] =c(distance_change_kl_close_dif[[j]],(results_sample1$distance_close_dif[[j]][length(results_sample1$distance_close_dif[[j]])]))
      }
      distance_change_kl[[j]] =c(distance_change_kl[[j]],(results_sample1$distance_kl[[j]][length(results_sample1$distance_kl[[j]])]))
      distance_change_wass[[j]] =c(distance_change_wass[[j]],(results_sample1$distance_wass[[j]][length(results_sample1$distance_wass[[j]])]))
      distance_change_mean_dif[[j]] =c(distance_change_mean_dif[[j]],(results_sample1$distance_mean[[j]][length(results_sample1$distance_mean[[j]])]))
      distance_change_var_dif[[j]] =c(distance_change_var_dif[[j]],(results_sample1$distance_var[[j]][length(results_sample1$distance_var[[j]])]))
      

      
    }
  }
  distance_last_stage = list()
  distance_last_stage$kl= distance_change_kl
  distance_last_stage$wass = distance_change_wass
  distance_last_stage$distance_change_mean_dif = distance_change_mean_dif
  distance_last_stage$distance_change_var_dif = distance_change_var_dif
  if(density_num==1){
    distance_last_stage$distance_change_kl_close_dif = distance_change_kl_close_dif
    distance_last_stage$kl_close = distance_change_kl_close
    
  }

  
  return(distance_last_stage)
}


# get the permutation's lowest distance
get_low_distance = function(w_1,w_2,prop_vec = NULL){
  tmp_w_1 = w_1
  n_per = factorial(length(w_1))
  per_loc = permutations(length(w_1))
  dis_get = 10000000
  for (i in 1:n_per) {
    com_w_1 = tmp_w_1[per_loc[i,]]
    if(!is.null(prop_vec)){
      new_dis = (com_w_1-w_2)%*%diag(prop_vec)%*%(com_w_1-w_2)
    }else{
      new_dis = sum((com_w_1 - w_2)^2)
    }
    if(new_dis < dis_get){
      
      dis_get = new_dis
      dis_loc = com_w_1
    }
  }
  return(dis_get)
}

## compare with MCMC with phi_mix as proposal. -proposal_density(y|x)
Stand_MCMC = function(Num_sample, proposal_density, proposal_sample_fun, w_star = NULL,Dim_w,dens){
  w_cur <- w_star
  sample_get <- matrix(0,nrow = Num_sample, ncol = Dim_w)
  proposal_value_cur = rep(0,Num_sample)
  proposal_value_next = rep(0,Num_sample)
  probability_rej = rep(0,Num_sample)
  for (i in 1:Num_sample) {
    w_next <- proposal_sample_fun(w_cur,dens)
    t_num = proposal_density(w_cur,dens)
    t_den = proposal_density(w_next,dens)
    qden_num = q_density(w_next)
    qden_den = q_density(w_cur)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      w_cur = w_next
    }
    proposal_value_cur[i] = t_num
    proposal_value_next[i] = t_den
    probability_rej[i] = p_accept
    sample_get[i, ] <- w_cur
  }
  return(list(sample_get = sample_get, probability_rej = probability_rej, proposal_value_cur = proposal_value_cur, proposal_value_next = proposal_value_next))
}

# proposal density function -- phi_mix
proposal_density = function(w,dens){
  tmpw <- t(w)
  d2 = predict(dens, tmpw, what = c("dens"), logarithm = FALSE)
  phi_mix = d2[1]
  return(phi_mix)
}
# proposal sample function -- phi_mix
proposal_sample_fun = function(w = NULL,dens){
  component_num = dens$G
  pi_vec = dens$parameters$pro
  # mean_vec = dens$parameters$mean[variable_num,]
  # sigma_vec = c()
  # for (i in 1:component_num) {
  #   tmpsigma = dens$parameters$variance$sigma[,,i][variable_num,variable_num]
  #   sigma_vec = c(sigma_vec,tmpsigma)
  # }
  tmpsample1 = sample(x =1:component_num ,size = 1,replace = TRUE,prob = pi_vec)
  mean_cur = dens$parameters$mean[,tmpsample1]
  Sigma_cur = dens$parameters$variance
  Sigma_cur = Sigma_cur$sigma[,,tmpsample1]
  tmpsample2 = mvrnorm(n = 1,mean_cur,Sigma_cur)
  return(tmpsample2)
}


## compare MCMC with phi_mix and compare our modified warpU sampling method with same phi_mix
Compare_N_sample = function(N, w_start, dens, q_density, inserversion_num,alpha = NULL,sigma2 = NULL,sigma_I = NULL,skip_MH = NULL){
  if((inserversion_num !=25 )&& (inserversion_num !=24) && (inserversion_num !=23)&& (inserversion_num !=00) && (inserversion_num !=223) ){
    sample_result = get_N_sample_update (N,w_start,dens,q_density,inserversion = inserversion_num)
  }else{
    sample_result = get_N_sample_update (N,w_start,dens,q_density,inserversion = inserversion_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I,skip_MH = skip_MH)
    
  }
  return(sample_result)
  
}

adaptive_warpU_tree = function(com_num,N, initial_data = NULL, inserversion_num, Layer_num, sample_range = NULL,density_num = NULL,alpha = NULL,sigma2 = NULL,sigma_I = NULL,star_w = NULL,True_com = NULL,mean_dis_true = NULL,pro_dis_true = NULL,store_sample_dim = NULL,Pre_stage_form = NULL, maxupdate_stage = NULL, all_pre_flag = NULL,equal_separate = FALSE,initial_type = NULL){
  K_prop = com_num
  if(is.null(initial_data)){
    # initial_type = 1: uniform, initial_type != 1: Gaussian mixture
    if(initial_type == 1){
      dim_num = dim(sample_range)[2]
      initial_data = matrix(0,nrow = N,ncol = dim_num)
      
      for (i in 1:dim_num) {
        tmpx = runif(n = N,min = sample_range[1,i],max = sample_range[2,i])
        initial_data[,i] = tmpx
      }
      
    }else{
      dim_num = dim(sample_range)[2]
      quantile_to_seperate = apply(sample_range, 2, quantile)
      # init_data_tmpdata1 = mvrnorm(n = floor(N/2),quantile_to_seperate[2, ],Sigma = initial_type*diag(dim_num))
      # init_data_tmpdata2 = mvrnorm(n = floor(N/2),quantile_to_seperate[4, ],Sigma = initial_type*diag(dim_num))
      init_data_tmpdata1 = mvrnorm(n = floor(N/2),quantile_to_seperate[1, ],Sigma = initial_type*diag(dim_num))
      init_data_tmpdata2 = mvrnorm(n = floor(N/2),quantile_to_seperate[2, ],Sigma = initial_type*diag(dim_num))
      
      initial_data = rbind(init_data_tmpdata1, init_data_tmpdata2)
      
    }

  }
  data = initial_data
  store_sample_stage = c()
  store_sample_stage_all = c()
  
  if(!is.null(store_sample_dim)){
    store_sample_stage = cbind(store_sample_stage,data[,store_sample_dim])
  }
  num_store_alllayer_list = list()
  length(num_store_alllayer_list) = Layer_num
  for (i in 1:Layer_num) {
    # if (i == 1){
    #   q_density_layer = q_density
    #   data_layer = data
    #   # dens
    #   dens_layer = get_phi_mix(data_layer,com_num = 2)
    #   # start point
    #   w_start_layer = dens_layer$parameters$mean[,1]
    #   sample_result_layer = get_N_sample_update (N,w_start = w_start_layer,dens = dens_layer,q_density = q_density_layer,inserversion = inserversion_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
    #   q_density_layer_list = q_density_layer_list_new
    #   data_layer_list = data_layer_list_new
    #   num_store_alllayer_list[[i]] = num_store_eachlayer_list
    # }else{
    # 
    #     
    # }
    
    
    q_density_layer_list_new = list()
    length(q_density_layer_list_new) = 2^i
    data_layer_list_new = list()
    length(data_layer_list_new) = 2^i
    num_store_eachlayer_list = list()
    length(num_store_eachlayer_list) = 2^i
    dens_layer_old_list = list()
    length(dens_layer_old_list) = 2^i
    # library(doRNG)
    # library(foreach)
    # library(doParallel)
    # cl <- makeCluster(10,type="FORK")
    # #cl.cores <- detectCores()
    # #cl<- makeCluster(cl.cores)
    # registerDoParallel(cl) 
    #result_foreach = foreach(maininter = (1:(2^(i))) )%dopar%{
    for(maininter in (1:(2^(i-1))) ){
      print("i=")
      print(i)
      print("maininter")
      print(maininter)
      if (i == 1){
        q_density_layer = q_density_global
        data_layer = data
      }else{
        ##
        dens_layer_old = dens_layer_old_list_store[[maininter]]
        q_density_layer = q_density_layer_list[[maininter]]
        if(equal_separate == TRUE){
          data_layer = data_layer_list[[maininter]]
          quantile_to_seperate = apply(data_layer, 2, quantile)
          var_diag = apply(data_layer, 2, var)
          dim_num = dim(quantile_to_seperate)[2]
          tmpdata1 = mvrnorm(n = floor(N/2),quantile_to_seperate[2, ],Sigma = 0.1*diag(var_diag))
          tmpdata2 = mvrnorm(n = floor(N/2),quantile_to_seperate[4, ],Sigma = 0.1*diag(var_diag))
          data_layer = rbind(tmpdata1, tmpdata2)
        }else{
          data_layer = data_layer_list[[maininter]]
        }
      }
      # dens
      #
      #dens_layer = get_phi_mix(data_layer,com_num = 2)
      # start point
      sample(1:(dim(data_layer)[1]), size = 1,replace = FALSE )

      w_start_layer = data_layer[sample(1:(dim(data_layer)[1]), size = 1,replace = FALSE ),]# dens_layer$parameters$mean[,1]
      quantile_to_range = apply(data_layer, 2, range)
      #sample_result_layer = get_N_sample_update (N,w_start = w_start_layer,dens = dens_layer,q_density = q_density_layer,inserversion = inserversion_num,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I)
      # sample_result_layer = adaptive_warpU(initial_data = data_layer,com_num = 2,N = N,inserversion_num = inserversion_num, stage_num=2, sample_range = quantile_to_range,True_Data = True_Data,density_num = density_num,alpha = alpha,sigma2 = sigma2,True_com = True_com,mean_dis_true = mean_dis_true,pro_dis_true = pro_dis_true,store_sample_dim = 2,Pre_stage_form = 1,all_pre_flag = 1)
      if(i == 1){
        sample_result_layer = adaptive_warpU(initial_data = data_layer,com_num = 2,N = N,inserversion_num = inserversion_num, stage_num=8, sample_range = quantile_to_range,q_density = q_density_layer,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I,density_num = density_num,plotfirstflag = 1)
        
      }else{
        sample_result_layer = adaptive_warpU(initial_data = data_layer,com_num = 2,N = N,inserversion_num = inserversion_num, stage_num=8, sample_range = quantile_to_range,q_density = q_density_layer,alpha = alpha,sigma2 = sigma2,sigma_I = sigma_I,density_num = density_num)
      }
      
      # dens_layer
      dens_layer_old = sample_result_layer$dens
      
      dens_layer_old_list[[2*maininter-1]] = sample_result_layer$dens
      dens_layer_old_list[[2*maininter]] = sample_result_layer$dens
      q_density_tmpnew1 = function(x_vec){
        return( (q_density_global(x_vec)*(get_condition_prob(dens_layer_old,x_vec)[1] ) ) )
      }
      q_density_tmpnew2 = function(x_vec){
        return( (q_density_global(x_vec)*(get_condition_prob(dens_layer_old,x_vec)[2] ) ) )
      }
      q_density_layer_list_new[[2*maininter-1]] = q_density_tmpnew1
      q_density_layer_list_new[[2*maininter]] = q_density_tmpnew2
      
      data_layer_list_new[[2*maininter-1]] = sample_result_layer$w_all[sample_result_layer$index_forward == 1, ]
      data_layer_list_new[[2*maininter]] = sample_result_layer$w_all[sample_result_layer$index_forward== 2,]
      plot_density1 = density(data_layer_list_new[[2*maininter-1]][,1],width = 1.3)
      plot_density2 = density(data_layer_list_new[[2*maininter]][,1],width = 1.3)
      pdf(file = paste0('./Plot_stored/Tree_',i,'-',(2*maininter-1),'sigma2_',sigma2,'Largestlayer_',Layer_num,'N_',N,'dim_',dim(sample_range)[2],'initial_type_',initial_type,'plot.pdf'),height = 5,width = 8)
      print(plot(plot_density1, main = paste0('Tree_density_',i,'-',(2*maininter-1),'sigma2_',sigma2,'Largestlayer_',Layer_num,'N_',N,'dim_',dim(sample_range)[2]),'initial_type_',initial_type ) )
      dev.off()
      pdf(file = paste0('./Plot_stored/Tree_',i,'-',(2*maininter),'sigma2_',sigma2,'Largestlayer_',Layer_num,'N_',N,'dim_',dim(sample_range)[2],'initial_type_',initial_type,'plot.pdf'),height = 5,width = 8)
      print(plot(plot_density2, main = paste0('Tree_density_',i,'-',(2*maininter),'sigma2_',sigma2,'Largestlayer_',Layer_num,'N_',N,'dim_',dim(sample_range)[2]),'initial_type_',initial_type ))
      dev.off()
      num_store_eachlayer_list[[2*maininter-1]] = sum(sample_result_layer$index_forward == 1) 
      #index_back #index_forward
      num_store_eachlayer_list[[2*maininter]] = sum(sample_result_layer$index_forward == 2) 
    }
    q_density_layer_list = q_density_layer_list_new
    data_layer_list = data_layer_list_new
    num_store_alllayer_list[[i]] = num_store_eachlayer_list
    dens_layer_old_list_store = dens_layer_old_list
  }
  sample_result = list()
  sample_result$num_store_alllayer_list = num_store_alllayer_list
  sample_result$data_layer_list = data_layer_list

  return(sample_result)
}

