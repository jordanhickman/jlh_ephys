import numpy as np




def psth_arr(spiketimes, stimtimes, pre=0.5, post=2.5,binsize=0.05,variance=True):
    '''
    Generates avg psth, psth for each trial, and variance for a list of spike times (usually a single unit)
    '''
    numbins = int((post+pre)/binsize)
    x = np.arange(-pre,post,binsize)

    bytrial = np.zeros((len(stimtimes),numbins-1))
    for j, trial in enumerate(stimtimes):
        start = trial-pre
        end = trial+post
        bins_ = np.arange(start,end,binsize)
        trial_spikes = spiketimes[np.logical_and(spiketimes>=start, spiketimes<=end)]
        hist,edges = np.histogram(trial_spikes,bins=bins_)
        if len(hist)==numbins-1:
            bytrial[j]=hist
        elif len(hist)==numbins:
            bytrial[j]=hist[:-1]
        if variance == True:
            var = np.std(bytrial,axis=0)/binsize/np.sqrt((len(stimtimes)))
            
    psth = np.nanmean(bytrial,axis=0)/binsize
    return psth, bytrial, var                            
