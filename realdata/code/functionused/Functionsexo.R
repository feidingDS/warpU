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

q_pro_25 = function(w_pre,w_star,alpha,sigma2){
  s = alpha*dmvnorm(w_star,mean= w_pre ,sigma = sigma2*diag(length(w_pre))) + (1-alpha)*(1/(prod(sample_range[2,] - sample_range[1,])))
  
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
InversWarpU_sample = function(w,ComNum,dens){
  mu = dens$parameters$mean[,ComNum]
  S = dens$parameters$variance
  S_use = S$sigma[,,ComNum]
  w_tran = S_use%*%w + mu
  return(w_tran)
}

# conditional probability
get_condition_prob = function(dens,w){
  #w = rbind(w,w)
  tmpw = t(w)
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
  tmpw = t(w)
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
    print('phik')
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
    Jacobian_det = det(Jacobian_matrix)
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
get_one_sample_update = function(w_pre,dens,q_density,inserversion = NULL,alpha = NULL,sigma2 = NULL){
  Num_com = dens$G
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
    print('23')
    w_star = mvrnorm(n = 1,w_pre,50*diag(length(w_pre)))
    
    #
    t_num = dmvnorm(t(w_pre),mean= w_star ,sigma = 50*diag(length(w_pre)))
    t_den = dmvnorm(w_star,mean= t(w_pre),sigma = 50*diag(length(w_pre)))
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    print('p_accept')
    print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      w_pre = w_star}
  }
  # alpha*normal + (1-alpha)*uniform
  if(inserversion == 25){
    # gibbs + warpU
    # changed gibbs only
    norm_unif_index = sample(1:2,size = 1,prob = c(alpha,(1-alpha)))
    if(norm_unif_index == 1){
      w_star = mvrnorm(n = 1,w_pre,sigma2*diag(length(w_pre)))

    }else{
      w_star = c()
      for (unif_i in 1:length(w_pre)) {
        tmp_new_star = runif(n = 1,min = sample_range[1,unif_i],max = sample_range[2,unif_i])
        w_star = c(w_star,tmp_new_star)
      }
    }
    #
    t_num = q_pro_25(w_pre =w_star ,w_star = t(w_pre)  ,alpha,sigma2)
    t_den = q_pro_25(w_pre =t(w_pre) ,w_star = w_star  ,alpha,sigma2)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    print('p_accept')
    print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
      w_pre = w_star}
  }
  #
  ## for Normal + Uniform only
  if(inserversion == 24){
    # gibbs + warpU
    # changed gibbs only
    norm_unif_index = sample(1:2,size = 1,prob = c(alpha,(1-alpha)))
    if(norm_unif_index == 1){
      w_star = mvrnorm(n = 1,w_pre,sigma2*diag(length(w_pre)))
      
    }else{
      w_star = c()
      for (unif_i in 1:length(w_pre)) {
        tmp_new_star = runif(n = 1,min = sample_range[1,unif_i],max = sample_range[2,unif_i])
        w_star = c(w_star,tmp_new_star)
      }
    }
    #
    t_num = q_pro_25(w_pre =w_star ,w_star = t(w_pre)  ,alpha,sigma2)
    t_den = q_pro_25(w_pre =t(w_pre) ,w_star = w_star  ,alpha,sigma2)
    qden_num = q_density(w_star)
    qden_den = q_density(w_pre)
    tmp_accept = (t_num*qden_num)/(t_den*qden_den)
    p_accept = min(tmp_accept,1)
    print('p_accept')
    print(p_accept)
    tmp_compare = runif(1,min = 0,max = 1)
    if (tmp_compare <= p_accept) {
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
  print('tmpvarphi2')
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
   
  tima_list = list()
  tima_list$w_next = w_next
  tima_list$indexcom = tmpvarphi1
  tima_list$inv_indexcom = tmpvarphi2
  tima_list$index_store = index_store
  return(tima_list)
}
# get all sample
get_N_sample_update = function(N,w_start,dens,q_density,inserversion = NULL,jumpsteps = NULL,alpha = NULL,sigma2 = NULL){
  w_all = c()
  index_forward = c()
  index_back = c()
  index_gibbs = c()
  change_forward = c()
  change_back = c()
  change_gibbs = c()
  w_pre = w_start
  for (i in 1:N) {
    print('i=')
    print(i)
    if((inserversion !=25) &&(inserversion !=24)){
      tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion)
    }else{
      tmp_w_next = get_one_sample_update(w_pre,dens,q_density,inserversion = inserversion,alpha = alpha,sigma2 = sigma2)
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

    w_pre = w_next
    if(!is.null(jumpsteps)){
      if( (i%%jumpsteps) == 0){
        w_all = rbind(w_all,t(w_next))
      }
    }else{
      w_all = rbind(w_all,t(w_next))
    }
  }
  zzz = list()
  zzz$w_all = w_all
  zzz$index_forward = index_forward
  zzz$index_back = index_back
  zzz$index_gibbs = index_gibbs
  zzz$change_forward = change_forward
  zzz$change_back = change_back
  zzz$change_gibbs = change_gibbs
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

adaptive_warpU = function(com_num,N, initial_data = NULL, True_Data = NULL, inserversion_num, stage_num, sample_range = NULL,density_num = NULL,alpha = NULL,sigma2 = NULL){
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
  Dim_use = dim(data)[2]
  distance = list(NULL)
  distance_close = list(NULL)
  distance_kl = list(NULL)
  distance_wass = list(NULL)
  length(distance_kl) = Dim_use
  length(distance_wass) = Dim_use
  length(distance) = Dim_use
  length(distance_close) = Dim_use
  distance_close_dif = list(NULL)
  length(distance_close_dif) = Dim_use
  index_distance = (ceiling((dim(data)[1])/2)) : (dim(data)[1])
  for ( i in 1:Dim_use) {
    #distance[[i]] = get_Wasserstein(sample1 = data,sample2 = True_Data,dim_num = i)
    tmp_data_index = data[index_distance,]
    distance_kl[[i]] = get_Wasserstein(sample1 = tmp_data_index,sample2 = True_Data,dim_num = i)$kl
    distance_wass[[i]] = get_Wasserstein(sample1 = tmp_data_index,sample2 = True_Data,dim_num = i)$wasserstein
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
  for (i in 1:stage_num) {
    print('i')
    print(i)
    dens = get_phi_mix(data,com_num)
    w_start = data[1,]
    w_start = dens$parameters$mean[,1]
    # plot(dens)
    # 2
    # 0
    if((inserversion_num !=25 )&& (inserversion_num !=24)){
      sample_result = get_N_sample_update (N,w_start,dens,q_density,inserversion = inserversion_num)
    }else{
      sample_result = get_N_sample_update (N,w_start,dens,q_density,inserversion = inserversion_num,alpha = alpha,sigma2 = sigma2)
      
    }
    data = sample_result$w_all

    index_distance = (ceiling((dim(data)[1])/2)) : (dim(data)[1])
    for (j in 1:Dim_use) {
      #distance[[j]] = c( distance[[j]],get_Wasserstein(sample1 = data,sample2 = True_Data,dim_num = j))
      distance_kl[[j]] = c( distance_kl[[j]],get_Wasserstein(sample1 = data[index_distance,],sample2 = True_Data,dim_num = j)$kl)
      distance_wass[[j]] = c( distance_wass[[j]],get_Wasserstein(sample1 = data[index_distance,],sample2 = True_Data,dim_num = j)$wasserstein)
      if(density_num == 1){
        com_dens = densityMclust(data = data[index_distance,j],G = 1)
        Sigma1 = com_dens$parameters$variance$sigmasq
        mu1 = com_dens$parameters$mean
        distance_close[[j]] = c(distance_close[[j]],get_kl_multinormal(mu1 = mu1,mu2 = mu2[j],Sigma1 = Sigma1,Sigma2 = Sigma2[j,j]))
        mu11 = mean(data[index_distance,j])
        Sigma11 = var(data[index_distance,j])
        distance_close_dif[[j]] = c(distance_close_dif[[j]],get_kl_multinormal(mu1 = mu11,mu2 = mu2[j],Sigma1 = Sigma11,Sigma2 = Sigma2[j,j]))
        
      }
      


    }
    # if(Dim_use != 1){
    #   distance_join = c (distance_join, get_Wasserstein(sample1 = data,sample2 = True_Data))
    # }
    
  }
  
  sample_result$dens = dens
  sample_result$distance_kl = distance_kl
  sample_result$distance_wass = distance_wass
  if(density_num ==1){
    sample_result$distance_close = distance_close
    sample_result$distance_close_dif = distance_close_dif
    
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
  for(i in last_num){
    if((inserversion_num !=25) &&(inserversion_num !=24)){
      results_sample1 = adaptive_warpU(initial_data = results_sample$w_all,com_num = com_num,N = i,inserversion_num = inserversion_num, stage_num=1, sample_range = sample_range,True_Data = True_Data,density_num = density_num)
    }else{
      results_sample1 = adaptive_warpU(initial_data = results_sample$w_all,com_num = com_num,N = i,inserversion_num = inserversion_num, stage_num=1, sample_range = sample_range,True_Data = True_Data,density_num = density_num,alpha = alpha,sigma2 = sigma2)
    }
    for (j in 1:Dim_use) {
      if(density_num ==1){
        distance_change_kl_close[[j]] =c(distance_change_kl_close[[j]],(results_sample1$distance_close[[j]][length(results_sample1$distance_close[[j]])]))
        distance_change_kl_close_dif[[j]] =c(distance_change_kl_close_dif[[j]],(results_sample1$distance_close_dif[[j]][length(results_sample1$distance_close_dif[[j]])]))
      }
      distance_change_kl[[j]] =c(distance_change_kl[[j]],(results_sample1$distance_kl[[j]][length(results_sample1$distance_kl[[j]])]))
      distance_change_wass[[j]] =c(distance_change_wass[[j]],(results_sample1$distance_wass[[j]][length(results_sample1$distance_wass[[j]])]))

      
    }
  }
  distance_last_stage = list()
  distance_last_stage$kl= distance_change_kl
  distance_last_stage$wass = distance_change_wass
  if(density_num==1){
    distance_last_stage$distance_change_kl_close_dif = distance_change_kl_close_dif
    distance_last_stage$kl_close = distance_change_kl_close
    
  }

  
  return(distance_last_stage)
}

