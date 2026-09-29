## ============================================================================
## Generates Figure 6 of the paper (real-data analysis).
##
## Left  panel: observed radial-velocity (RV) measurements with their error
##              bars, overlaid with the predicted RV curve from the Keplerian
##              model evaluated at the MAP estimates of the planet parameters.
## Right panel: marginal posterior density of the mean anomaly parameter,
##              compared across Warp-U MCMC, Parallel Tempering, Hamiltonian
##              Monte Carlo (Stan), and a numerical-integration reference.
##
## Inputs  (all under realdata/data/):
##   dataset1_one_planet_posterior.RData  -- provides data_now (RV obs) and v()
##                                           (Keplerian RV function)
##   prior_bounds.RData                   -- prior box bounds (loaded for context)
##   stanData.RData                       -- HMC/Stan posterior samples
##   PL_samples_Real_plot.RData           -- PT (PL_list)
##   AdaptiveWarpU_Real_plot.RData        -- Warp-U (results_sample)
##   int_results_gridfrom_Real.RData      -- numerical integration of the
##                                           marginal posterior on a grid
##
## Output:
##   output/figures/p_density_final.pdf   -- Figure 6 (two-panel PDF)
## ============================================================================

rm(list = ls())
# Libraries
library(mvtnorm)
library(mnormt)
library(numDeriv)
library(stats4)
library(MASS)
library(ggplot2)
library(bayestestR)
library(numbers)
library(ggpubr)

# Load posterior setup, prior bounds, and Stan samples used as input below.
load('data/dataset1_one_planet_posterior.RData')
load('data/prior_bounds.RData')
load('data/stanData.RData')

results_sample = list()
results_sample$w_all = data
P_data = results_sample$w_all[,2]                 # column 2: orbital period

period_est = map_estimate(P_data)$MAP_Estimate
period_adjusted = (data_now$time %% period_est)/period_est

# Columns 3-6 of the Stan samples correspond, in order, to K, e, w, M0.
K_est  = map_estimate(results_sample$w_all[,3])$MAP_Estimate
e_est  = map_estimate(results_sample$w_all[,4])$MAP_Estimate
w_est  = map_estimate(results_sample$w_all[,5])$MAP_Estimate
M0_est = map_estimate(results_sample$w_all[,6])$MAP_Estimate
planet_paras <- list()
planet_paras[[1]] <- list(tau=period_est,K=K_est,e=e_est,w=w_est,M0=M0_est,gamma=0)

# Prepare data for plotting
time_dense = seq(from = 0, to = 600, length.out = 1000)
RV_get_dense = v(time_dense,planet_paras[[1]])
phrase_dense = (time_dense %% period_est)/period_est
RV_data_dense = data.frame(time_dense, phrase_dense, RV_get_dense)

# Create RV_data_plot first
RV_data_plot = data.frame(data_now, period_adjusted = period_adjusted)
RV_data_plot$RV_getfrom_function = v(RV_data_plot$time, planet_paras[[1]])

# Plot 1: RV time series
p1_plot = ggplot(RV_data_plot, aes(time, rv)) +
  geom_point(aes(color = 'RV'), shape = 2) +
  geom_errorbar(aes(x = time, y = rv, ymin = rv-sd, ymax = rv+sd, color = 'RV'),
                alpha = I(3/7), width = .03, position = position_dodge(0.05)) +
  geom_line(data = RV_data_dense, 
            aes(time_dense, RV_get_dense, color = 'predicted radial velocity at time t'),
            size = 1) +
  theme_bw() +
  ylab("RV (m/s)") +
  xlab('Time') +
  theme(axis.title.x = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        legend.position = 'none',
        legend.title = element_blank()) +
  scale_color_manual(values = c("red", "blue"))

# --- Right panel: kernel density of the mean-anomaly marginal for each method.
# variable_num = 6 selects the mean anomaly column in every sample matrix.
variable_num = 6

# Parallel Tempering samples.
load('data/PL_samples_Real_plot.RData')
sample_results = t(PL_list[[1]][ 1 , ,])
PT_density = density(sample_results[,variable_num], from = -0.5, to = 6)

# Warp-U MCMC samples.
load('data/AdaptiveWarpU_Real_plot.RData')
WarpU_sample_results = results_sample$w_all
WarpU_density = density(WarpU_sample_results[,variable_num], from = -0.5, to = 6)

# Stan / Hamiltonian Monte Carlo samples.
load('data/stanData.RData')
Stan_sample_results = data
Stan_density = density(Stan_sample_results[,variable_num], from = -0.5, to = 6)

# Numerical-integration reference curve evaluated on a 1d grid.
load('data/int_results_gridfrom_Real.RData')
abs_dif = int_out[1:199] - int_out[2:200]
Index_large = Index_small = c()
for (i_index in 1:20) {
  sorted_desc = sort(abs_dif, decreasing = TRUE)
  sorted_asc = sort(abs_dif, decreasing = FALSE)
  Index_large = c(Index_large, which(abs_dif == sorted_desc[i_index]))
  Index_small = c(Index_small, which(abs_dif == sorted_asc[i_index]))
}
Index_all = c(Index_large, Index_small)
int_out = int_out[-Index_all]
scale_int_out = int_out/(10^(-193.71))

# Combine all data for density plot
Data_all = rbind(
  data.frame(x = WarpU_density$x, y = WarpU_density$y, se = 0, curveID = 'Warp-U MCMC'),
  data.frame(x = PT_density$x, y = PT_density$y, se = 0, curveID = 'PT'),
  data.frame(x = Stan_density$x, y = Stan_density$y, se = 0, curveID = 'Hamiltonian Monte Carlo'),
  data.frame(x = grid_search[-Index_all], y = scale_int_out, se = 0, curveID = 'Numerical Integral')
)

# Plot 2: Density estimation
p2 = ggplot(Data_all, aes(x = x, y = y, group = curveID, color = curveID)) +
  geom_line(aes(linetype = curveID), size = 1) +
  theme_bw() + 
  scale_linetype_manual(values = c("dashed", "dotted", "dashed", "solid"),
                       breaks = c("Warp-U MCMC", "PT", "Hamiltonian Monte Carlo", "Numerical Integral")) +
  scale_colour_manual(values = c("blue", "orange", "green", "red"),
                     breaks = c("Warp-U MCMC", "PT", "Hamiltonian Monte Carlo", "Numerical Integral")) +
  scale_fill_manual(values = c("blue", "red")) +
  xlab('Mean anomaly') + 
  ylab('Density') +
  theme(axis.title.x = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        legend.position = c(0.33, 0.80),
        legend.title = element_blank(),
        legend.text = element_text(size = 10))

# Combine plots and save
p6_final = ggarrange(p1_plot, p2)
pdf.width <- 9
pdf.height <- 4
pdf('output/figures/p_density_final.pdf', height = pdf.height, width = pdf.width)
print(p6_final)
dev.off()





