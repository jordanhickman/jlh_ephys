import numpy as np

import matplotlib.pyplot as plt


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


def plot_psth(psth, variance, bin_centers, plot_type='line', color = 'blue', alpha = 1, ax=None):
    """
    Plots the PSTH with variance, with time represented in milliseconds (ms).
    
    Parameters:
    - psth: array, averaged firing rate per bin.
    - variance: array, variance of firing rate per bin.
    - bin_centers: array, centers of the bins.
    - plot_type: str, 'histogram' or 'line'.
    - ax: matplotlib.axes.Axes, axis for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Convert bin_centers from seconds to milliseconds
    bin_centers_ms = bin_centers * 1000
    
    if plot_type == 'histogram':
        # Adjust the width proportionally if necessary
        width_ms = np.diff(bin_centers_ms)[0] if len(bin_centers_ms) > 1 else 1
        ax.bar(bin_centers_ms, psth, width=width_ms, alpha=0.6, label='Mean Firing Rate')
    elif plot_type == 'line':
        var = variance # variance specified from psth function (sem, std, or var)
        ax.plot(bin_centers_ms, psth, color = color,  label='Mean PSTH')
        ax.fill_between(bin_centers_ms, psth - var, psth + var, alpha=0.3, color = color)
    else:
        print("Invalid plot type. Choose 'histogram' or 'line'.")
        return
    
    ax.set_xlabel('Time from Stimulus (ms)')
    ax.set_ylabel('Average Firing Rate (Hz)')
    ax.set_title('PSTH (' + plot_type.capitalize() + ')')
    ax.legend()

def raster(self, times, triggers, pre=0.25, post=0.5, color = 'blue', 
               linewidth = 0.5, 
               labelsize = 8, 
               axis_labelsize = 2, 
               marker = 'o',
               marker_size = 5,  
               axes=None):
        if axes is None:
            plt.figure()
            axes = plt.gca()
        times = np.array(times).astype(float)
        triggers = np.array(triggers).astype(float)
        by_trial = []

        for i, t in enumerate(triggers):
            start_indices = np.where(times >= t - pre)[0]
            end_indices = np.where(times >= t + post)[0]
            if len(start_indices) > 0 and len(end_indices) > 0:
                start = start_indices[0]
                end = end_indices[0]
                by_trial.append(np.array(times[start:end])-t)
                axes.plot(np.array(times[start:end])-t, np.ones(len(np.array(times[start:end])-t))*i+1, marker, 
                          linewidth = linewidth,  color=color, markersize = marker_size)

        axes.set_xlim(-pre, post)
        axes.set_ylim(len(triggers), 1)

        # Styling
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.yaxis.set_ticks_position('left')
        axes.xaxis.set_ticks_position('bottom')
        axes.set_yticks(np.linspace(1, len(triggers), 3))
        tick_locations = np.round(np.linspace(-pre, post, 5), 2)
        axes.set_xticks(tick_locations)
        axes.axvline(x=0, color='r', linestyle='--', linewidth=0.5)
        axes.set_xlabel('Time (ms)', fontsize = labelsize)
        axes.set_ylabel('Trial', fontsize = labelsize)
        axes.set_xticklabels(np.linspace(-pre, post, 5)*1000)  # Convert to ms