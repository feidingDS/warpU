
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as gauss
import seaborn as sns
from tqdm.notebook import tqdm
from ot.sliced import sliced_wasserstein_distance as swDist
import matplotlib.gridspec as gridspec


# %% 
setID = 1
if setID == 1:
    import params.eg01setup as expset
    KComp = expset.KComp
    import params.eg01mcmc as mcmcparams
    theta_d = 10
    simu_setup_tmp = expset.simuSKT(theta_d)
    targetD_tmp = simu_setup_tmp.get_targetD()
    simu_setup_tmp.init_r()


# %%

methodList = [ 1, 2, 3, 4, 5, 6, 7]  #
dimList = [10, 50,100,  500, 1000] #

nMethod = 7
nDim = len(dimList)
totalRep = mcmcparams.get_totalRep()

labelName = [r'WarpU($\phi_{mix}^{gauss}$)', 
             r'WarpU($\phi_{mix}^{skew}$)', 
             r'WarpU($\phi_{mix}^{t}$)', 
             'PT-V', r'PT-V-Warp($\phi_{mix}^{gauss}$)',
             'PT-G', r'PT-G-WarpU($\phi_{mix}^{gauss}$)']


styleList = ["ro-.","bx:","g^--","k+--","m*--","yv--","c<--"]



#%%
def get_runtime():
    #totalRep = mcmcparams.get_totalRep()
    totalRep = 10
    result = np.zeros((nMethod, nDim, totalRep))
    J = -1
    for theta_d in dimList:
        J += 1
        for mm in methodList:
            for repK in range(totalRep):
                _, rt = mcmcparams.get_sample(mm, theta_d, repK)
                result[mm-1,J,repK] =rt

    ave = np.mean(result, axis= 2)
    return ave

def get_slicedWD():
    ## Compute the sliced Wasserstein distance between the MCMC samples
    ## and i.i.d samples for each method and each dimension.
    totalRep = mcmcparams.get_totalRep()
    result = np.zeros((nMethod, nDim, totalRep))
    J = -1
    for theta_d in tqdm(dimList, desc="Outer Loop"):
        J += 1
        simu_setup = expset.simuSKT(theta_d)
        targetD = simu_setup.get_targetD()
        simu_setup.init_r()
        numSamples = 6000
        sampleTrue = simu_setup.draw_sample(numSamples)[:,[0,9]]
        for repK in tqdm(range(totalRep), leave=False, desc="Inner Loop"):
            for mm in methodList:
                sample, _ = mcmcparams.get_sample(mm, theta_d, repK)
                result[mm-1, J, repK] = swDist(sampleTrue, sample, n_projections= 100)
    ave = np.mean(result, axis = 2)
    return ave
        
#%%
if __name__ == "__main__":
    aveTime = get_runtime()
    aveSWD = get_slicedWD()

#%%
# plot average running time and sliced Wasserstein distance
def show_rt_error(aveTime, aveSWD, save = False):
    theme_bw = "./theme_bw.mplstyle"
    plt.style.use(theme_bw)
    fig = plt.figure(figsize=(11, 4.5), dpi=100)
    gs = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])

    ax1 = fig.add_subplot(gs[0, 0])  # Left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Middle plot
    ax1.set_ylim(10, 1000)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel("Dimension d", fontsize=12)
    ax1.set_ylabel("Running Time (Seconds)", fontsize=12)
    for I in range(nMethod):
        ax1.plot(dimList, aveTime[I,:], styleList[I], label = labelName[I])
    ax1.grid(True)

    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylim(0.09, 10.5)
    ax2.set_xlabel("Dimension d", fontsize=12)
    ax2.set_ylabel("Sliced Wasserstein Distance", fontsize=12)
    for I in range(nMethod):
        ax2.plot(dimList, aveSWD[I,:], styleList[I], label = labelName[I])
    ax2.grid(True)

    # Add shared legend
    legend_ax = fig.add_subplot(gs[0, 2])  # Allocate the right column for the legend
    legend_ax.axis('off')  # Turn off axis for the legend space
    legend_ax.legend(
        *ax1.get_legend_handles_labels(),
        loc='center',
        frameon=True,
        fontsize=13,
        title="Samplers",
        title_fontsize='14'
    )
    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save:
        plt.savefig("./figs/simu1/summary.pdf",bbox_inches='tight')
    plt.show()

#%%
def plot_hist2d(dim = None, save = False):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    xseq = np.linspace(-6,5,num = 100).reshape(-1,1)
    repI = 0
    if dim is None:
        dimList = [10, 100, 1000]
    else:
        if not isinstance(dim, int):
            raise ValueError("dim must be an integer!")
        if dim not in [10, 100, 1000]:
            raise ValueError("dim must be one of 10, 100, 1000!")
        dimList = [dim]
    for theta_d in dimList:
        simu_setup = expset.simuSKT(theta_d)
        simu_setup.init_r()
        numSamples = 6000
        phimix = mcmcparams.get_phimix(1, expset.basepath, theta_d)
        sampleTrue = simu_setup.draw_sample(numSamples)
        for mm in [1, 4, 5, 6, 7]:
            sample, _ = mcmcparams.get_sample(mm, theta_d, repI)
            sns.set_theme(font_scale = 1.5)
            sns.set_style("white")
            g = sns.JointGrid(height = 5)
            g.figure.set_size_inches(4.5, 4.5) 
            ax = sns.kdeplot(x= sample[:,0], y= sample[:,-1], cmap="Blues", 
                            fill=True, ax=g.ax_joint)
            ax.set_xlabel(r'$\theta_{%d}$' % (1), fontsize = 16,labelpad = -2)
            ax.set_ylabel(r'$\theta_{%d}$' % (10), fontsize = 16,labelpad = -2)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-9, 9)
            sns.histplot(sample[:,0], ax=g.ax_marg_x, bins = 80, stat="density", ec="red")
            sns.kdeplot(data = sampleTrue[:,0], ax = g.ax_marg_x, color='black', linewidth=2)
            sns.histplot(y = sample[:,-1], bins = 80, ax=g.ax_marg_y,  stat="density", ec="red")
            sns.kdeplot(y = sampleTrue[:,9], ax = g.ax_marg_y, color='black', linewidth=2)
            labeltext = labelName[mm-1]
            ax.text(-9, 6.5, labeltext + ",\n" + r'$d={%d}$' % theta_d, fontsize = 16)
            if save:
                savePath = "./figs/simu1/simu1-" + str(mm) + "-" +  str(theta_d) +".pdf"
                plt.savefig(savePath, bbox_inches='tight')
                plt.close()
            else:
                plt.show()


