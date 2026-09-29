#%%
import numpy as np
import matplotlib.pyplot as plt
import params.eg03setup as expset
import params.eg03mcmc as mcmcset
import matplotlib.gridspec as gridspec

#from ot.sliced import sliced_wasserstein_distance as swDist
theta_d = 30
simu_setup = expset.simuSshape(theta_d)

#%%
savepath = "./results/eg03mcmc/bridge"
theta_d = 30
sample_type = 2

def organize(results,J):
    nRep = len(results)
    nCut = len(results[0])
    results_total = np.zeros((nCut, nRep))
    for i in range(nRep):
        results_total[:,i] = results[i][:,J]
    return results_total

def get_result(sample_type, theta_d, phimixIDs):
    results_total = None
    thetaMCMC = mcmcset.get_sample(sample_type, theta_d, 0)
    nsample = thetaMCMC.shape[0]
    ncuts = mcmcset.get_cutseq(nsample)

    for phimix_type in phimixIDs:
        savepath =  "./results/eg03mcmc/bridge"
        savepath = savepath + "/sampler_" + str(sample_type) +\
            "_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
        data = np.load(savepath)
        results = data["results"]
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
    return ncuts,results_total
# %%
# linetypes for 6 estimators
def add_plot(ax, cutseq, totalresult, phimixIDs,
            methodID = 1,
             plotquery = False, Kseq = None):
    styleList = ["ro-.","bx:","g^--","k+--",
                 "m*--","yv--","c<--","r<--","c<--","r<--"]
    labelName = [r'WB($\phi_{mix,2}^{gauss}$)', r'S-WB($\phi_{mix,2}^{gauss}$)', 
                r'WB($\phi_{mix,6}^{gauss}$)', r'S-WB($\phi_{mi,6}^{gauss}$)', 
                r'WB($\phi_{mix}^{neu}$)', r'S-WB($\phi_{mix}^{neu}$)',
                r'N-WB($\phi_{mix,2}^{gauss}$)', r'N-S-WB($\phi_{mix,2}^{gauss}$)']
    

    for j in range(totalresult.shape[1]):
        if plotquery:
            K = Kseq[j//2]
            if j % 2 == 0:
                num_eval = cutseq * (2*K)
            else:
                num_eval = cutseq * (K+1)
            ax.set_xlim(4500, 30000)
            ax.set_xlabel("Number of target evaluations", fontsize=13)
            sel = num_eval > 500
            if K == 2:
                mGap = 2
            else:
                mGap = 1
        else:
            mGap = 1
            num_eval = cutseq
            ax.set_xlabel("Number of samples", fontsize=13)
            sel = num_eval > 0
        ax.set_ylabel("Mean absolute error", fontsize=13)
        i = (phimixIDs[j//2]-1)*2 + (j % 2) 
        ax.plot(num_eval[sel], totalresult[sel,j], styleList[i], 
            label = labelName[i], markersize=5, markevery=mGap)
    ax.set_yscale("log")
    ax.set_ylim(9e-4, 1.5)

    if methodID == 1:
        phimix_type = 1
        savepath =  mcmcset.savepath + "bridge"
        savepath = savepath + "/LAIS_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
        file = np.load(savepath)
        resultMat = file["resultMat"]
        err = np.mean(np.abs(resultMat-1), axis=1)
        cutseq = file["cutseq"]
        ax.plot(cutseq, err, "r-", label = "LAIS", markersize=5, markevery=1)

    if methodID == 2:
        phimix_type = 2
        savepath =  mcmcset.savepath + "bridge"
        savepath = savepath + "/LAIS_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
        file = np.load(savepath)
        resultMat = file["resultMat"]
        err = np.mean(np.abs(resultMat-1), axis=1)
        cutseq = file["cutseq"]
        ax.plot(cutseq, err, "r-", label = "LAIS", markersize=5, markevery=1)

    if methodID == 7:
        phimix_type = 4
        savepath =  mcmcset.savepath + "bridge"
        savepath = savepath + "/LAIS_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
        file = np.load(savepath)
        resultMat = file["resultMat"]
        err = np.mean(np.abs(resultMat-1), axis=1)
        cutseq = file["cutseq"]
        ax.plot(cutseq, err, "r-", label = "LAIS", markersize=5, markevery=1)

    if methodID == 5:
        phimix_type = 2
        savepath =  mcmcset.savepath + "bridge"
        savepath = savepath + "/LAIS_INDEP_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
        file = np.load(savepath)
        resultMat = file["resultMat"]
        err = np.mean(np.abs(resultMat-1), axis=1)
        cutseq = file["cutseq"]
        ax.plot(cutseq, err, "r-", label = "LAIS", markersize=5, markevery=1)

# %%
def draw_plot(save = False):
    theme_bw = "./theme_bw.mplstyle"
    plt.style.use(theme_bw)
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(12, 8.5))
    samplerName = [r'WarpU($\phi_{mix,2}^{gauss}$)', 
                r'WarpU($\phi_{mix,6}^{gauss}$)', 
                r'WarpU($\phi_{mix}^{neu}$)',
                r'MH($\phi_{mix,2}^{gauss}$)',
                    r'MH($\phi_{mix,6}^{gauss}$)',
                    r'MH($\phi_{mix}^{neu}$)',
                    r'N-WarpU($\phi_{mix,2}^{gauss}$)']
    phimixIDs = [1,2,4]
    Kseq = [2, 6, 2]
    methodIDs = [1,2,7,5]
    for j in range(3,-1,-1):
        methodID = methodIDs[j]

        row = [0,0,1,1][j]
        col = [1,2,1,2][j]
        ncuts, totalresult = get_result(methodID, theta_d, phimixIDs)
        #ax = fig.add_subplot(gs[0, col])
        #add_plot(ax, totalresult, False, False)
        ax = fig.add_subplot(gs[row, col])
        add_plot(ax, ncuts, totalresult, phimixIDs, methodID,True, Kseq)
        ax.set_title("Sampler: "+ samplerName[methodID-1], 
                    fontsize = 14, fontweight='bold')
        #ax.text(40000, 0.12, "Sampler: "+ samplerName[methodID-1], 
        #    horizontalalignment='right', fontsize = 14)


    legend_ax = fig.add_subplot(gs[1, 0])  # Allocate the right column for the legend
    legend_ax.axis('off')  # Turn off axis for the legend space
    legend_ax.legend(
        *ax.get_legend_handles_labels(),
        loc='center',
        frameon=True,
        fontsize=13,
        title='Estimator',
        title_fontsize='14'
    )
    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    ax = fig.add_subplot(gs[0, 0]) 
    targetD = simu_setup.get_targetD()
    numSamples = 1e6
    sampleTrue, _ = targetD.drawsample(numSamples)
    t = ax.hist2d(sampleTrue[:,0], sampleTrue[:,1], bins=150, cmap="Blues", rasterized=True )
    ax.text(-0.5, -5, r"$\theta_1$", fontsize=13, color='black', ha='center', va='center')  # X-axis label
    ax.text(-6, 2.5, r"$\theta_2$", fontsize=13, color='black', rotation=90, ha='center', va='center')  # Y-axis label
    if save:
        plt.savefig("./figs/simu3/summary_query.pdf", bbox_inches='tight')
    plt.show()
    #plt.savefig("./figs/simu3/summary_query.pdf", bbox_inches='tight')




# %%
