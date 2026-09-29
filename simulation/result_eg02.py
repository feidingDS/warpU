#%%
import numpy as np
import matplotlib.pyplot as plt
import params.eg02setup as expset
import params.eg02mcmc as mcmcset
#from ot.sliced import sliced_wasserstein_distance as swDist
import os
theta_d = 30
simu_setup = expset.simuSKT(theta_d)
simu_setup.init_r()

#%%
savepath = "./results/eg02mcmc/bridge"
theta_d = 30
sample_type = 2
phimix_type = 2

def organize(results,J):
    nRep = len(results)
    nCut = len(results[0])
    results_total = np.zeros((nCut, nRep))
    for i in range(nRep):
        results_total[:,i] = results[i][:,J]
    return results_total

def drawsave(cuts, results_total, 
            plotquery = False, savepath = None, 
            showlegend = True, title = None):
    styleList = ["ro-.","bx:","g^--","k+--","m*--","yv--","c<--"]
    labelName = [r'WB($\phi_{gauss}$)',
                r'S-WB($\phi_{gauss}$)', 
                r'S-WB($\phi_{skew}$)', r'S-WB($\phi_{t}$)']

    theme_bw = "./theme_bw.mplstyle"
    plt.style.use(theme_bw)
    if savepath is not None:
        fig, ax = plt.subplots(figsize=(4.3,4.5), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    for i in range(results_total.shape[1]):
        if plotquery:
            if i == 0:
                xvalues = cuts * 20
            else:
                xvalues = cuts * 11
        else:
            xvalues = cuts
        ax.plot(xvalues, results_total[:,i], styleList[i],
                label = labelName[i], markersize=5)
    ax.set_ylabel("Mean absolute error")
    if title is not None:
        #fig.text(0.6, 0.8, title, horizontalalignment='center', fontsize=15)
        fig.suptitle(title, fontsize=13, fontweight="bold")
    if plotquery:
        ax.set_xlabel("Number of target queries")
    else:
        ax.set_xlabel("Number of samples")
    ax.set_ylim(0,0.08)
    if plotquery:
        ax.set_xlim(9000, 120000)
    if showlegend:
        ax.legend(fontsize=15)
    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        query = "query" if plotquery else "sample"
        fig.savefig(savepath + "/sampler_" + str(sample_type) + "_" +
                    query + ".pdf",bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_results(sample_type, phimix_type, theta_d):
    results_total = None
    for phimix_type in [1,2,3]:
            savepath =  "./results/eg02mcmc/bridge"
            savepath = savepath + "/sampler_" + str(sample_type) +\
                "_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
            if not os.path.exists(savepath):
                print(savepath)
                continue
            data = np.load(savepath)
            results = data["results"]
            if phimix_type > 1:
                BS = 2
                results2 = np.abs(organize(results,BS) - 1)
                onemean = np.mean(results2, axis=1).reshape(-1,1)
            else:
                BS = 0
                results2 = np.abs(organize(results,BS) - 1)
                onemean = np.mean(results2, axis=1).reshape(-1,1)
                BS = 1
                results2 = np.abs(organize(results,BS) - 1)
                tmp = np.mean(results2, axis=1).reshape(-1,1)
                onemean = np.hstack((onemean, tmp))
            if results_total is None:
                results_total = onemean
            else:
                results_total = np.hstack((results_total, onemean))
    return results_total

# %%
def draw_compare(savepath = "./figs/simu2"):
    cuts = mcmcset.get_cutseq(mcmcset.numberIter - 500)
    showlegend = False
    samplerName = [r'WarpU($\phi_{mix}^{gauss}$)', 
                r'WarpU($\phi_{mix}^{skew}$)', r'WarpU($\phi_{mix}^{t}$)', 
                'PT-V', r'PT-V-Warp($\phi_{mix}^{gauss}$)',
                'PT-G', r'PT-G-WarpU($\phi_{mix}^{gauss}$)']
    for sample_type in [1,2,3, 6, 7]:
        results_total = get_results(sample_type, phimix_type, theta_d)
        drawsave(cuts, results_total, plotquery=False, savepath=savepath, 
            showlegend=showlegend, title="Sampler: " + samplerName[sample_type-1])
        if sample_type == 3:
            showlegend = True
        if savepath is not None:
            title = None
        else:
            title = "Sampler: " + samplerName[sample_type-1]
        drawsave(cuts, results_total, plotquery=True, savepath=savepath,
            showlegend=showlegend, title=title)
        showlegend = False
# %%
if __name__ == "__main__":
    draw_compare()
# %%
