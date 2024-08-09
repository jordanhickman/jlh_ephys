import os,glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
from matplotlib import cm



from jlh_ephys.utils import OE

class Raw:
    def __init__(self, analysis_obj):
        self.mouse = analysis_obj.mouse
        self.date = analysis_obj.date
        self.path = analysis_obj.path
        try:
            self.probes = analysis_obj.probes
        except:
            print('No probes loaded, manually add probes to analysis object')
        if analysis_obj.processed:
            self.units = analysis_obj.units


    def get_raw(self, probe, band='ap'):
        # Normalize case for flexibility
        probe = probe.lower()
        band = band.lower()

        # Get the recording
        recording = OE(self.path)

        # Search for the correct data stream by examining metadata
        for data in recording.continuous:
            stream_name = data.metadata['stream_name'].lower().replace('-', '_')
            if stream_name == f"{probe}_{band}":
                print(f'confirming stream name: {data.metadata["stream_name"]}')
                return data

        print('No matching bands or probes')
        return None
    

    def plot_average_LFP_power(self, axes = None, band='gamma'):
        # Frequency bands
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80)
        }
        low, high = freq_bands[band]
            # Get the viridis colormap
        viridis = cm.get_cmap('viridis', 5)  # 5 to match the number of frequency bands
        band_colors = {
            'delta': viridis(0),
            'theta': viridis(1),
            'alpha': viridis(2),
            'beta': viridis(3),
            'gamma': viridis(4),
        }
        color = band_colors[band]


        if axes is None:
            fig, axes = plt.subplots(1, len(self.probes), sharey=True)
        elif not isinstance(axes, list):  # Make sure axes is a list
            axes = [axes]
        ax_list = []
        
        for ax, probe in zip(axes, self.probes):
            # Get the LFP data object
            print(probe)
            lfp_data_obj = self.get_raw(probe, band='lfp')
            all_data = lfp_data_obj.get_samples(start_sample_index=int(2500*600), end_sample_index=int(2500*700), selected_channels=np.arange(384))
            
            power = np.zeros(384)  # Assuming 300 channels

            for ch in range(384):
                ch_data = all_data[:, ch]

                # Bandpass filter
                nyquist = 0.5 * 2500  # Assuming a sampling rate of 2.5 kHz
                low_freq = low / nyquist
                high_freq = high / nyquist
                b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                filtered_data = signal.filtfilt(b, a, ch_data)

                # Calculate the Power Spectral Density (PSD)
                f, Pxx = signal.welch(filtered_data, fs=2500, nperseg=1024)

                # Isolate the power
                relevant_f = np.logical_and(f >= low, f <= high)
                power[ch] = np.mean(Pxx[relevant_f])

            # Log and Z-score transformation
            log_power = np.log1p(power)
            power_z = zscore(log_power)

            # Identify and remove outliers
            outliers = []
            for ch in range(1, len(power_z) - 1):  
                neighbors_mean = np.mean([power_z[ch - 1], power_z[ch + 1]])
                if np.abs(power_z[ch] - neighbors_mean) > 1.5:  # Z-score threshold
                    outliers.append(ch)

            power_clean = np.delete(power_z, outliers)

            # Line plot
            ax.plot(power_clean, np.arange(0, len(power_clean)), label=probe, c = color)

            ax.set_title(f'{probe} {band} power')
            if ax.get_subplotspec().is_first_col():
                ax.spines['right'].set_visible(False)
                ax.set_ylabel('Channel')
            elif ax.get_subplotspec().is_last_col():
                ax.spines['left'].set_visible(False)
            else:
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
            # Remove top spines for all
            ax.spines['top'].set_visible(False)

            # Remove y-ticks for second and third plots
            if not ax.get_subplotspec().is_first_col():
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            
            ax.set_xlim(-2,2)
            ax.set_yticks(np.arange(0, len(power_clean), 20))  # Show every 20th channel
            ax.set_yticklabels(np.arange(0, len(power_clean), 20))
            ax_list.append(ax)
        ax_list[1].set_xlabel(f'Z-scored {band.capitalize()} Power')

       
        
        return ax_list






    def get_chunk(self, probe, # can use binary_path output directly
              stim_times,
              band = 'ap',
              pre = 100, # time in ms
              post = 500, # time in ms
              chs = np.arange(0,200,1), # channels
              median_subtraction = False,
              offset_subtraction = False,
              ):
        """
        Takes in a continuous binary object and a list of stimulation times and returns a chunk of the data
        """
        data_stream = self.get_raw(probe = probe, band = band)
        sample_rate = data_stream.metadata['sample_rate']
        
        
        pre_samps = int((pre/1000 * sample_rate))
        post_samps = int((post/1000 * sample_rate))
        total_samps = pre_samps + post_samps

        n_chs = len(chs)
        
        response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
        stim_indices = np.searchsorted(data_stream.timestamps, stim_times)
        for i, stim in enumerate(stim_indices):
            start_index = int(stim - ((pre/1000)*sample_rate))
            end_index = int(stim + ((post/1000)*sample_rate))   
            chunk = data_stream.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                selected_channels = chs)
            
            if median_subtraction == True:
                corrected_chunk = chunk - np.median(chunk, axis = 0) #subtract offset 
                corrected_chunk = np.subtract(corrected_chunk.T, np.median(corrected_chunk[:,chs[-10]:chs[-1]], axis=1)) #median subtraction

                response[i,:,:] = corrected_chunk.T
            elif offset_subtraction == True:
                response[i,:,:] = chunk - np.median(chunk, axis = 0)
            else:
                response[i,:,:] = chunk

        return response
        
    def plot_ap(self, probe, stim_times, 
                pre = 4, post = 20, 
                first_ch = 125, last_ch = 175, 
                title = '', 
                median_subtraction = True,
                spike_overlay = False,
                n_trials = 10, spacing_mult = 350, 
                save = False, savepath = '', format ='png'):
        
        data = self.get_raw(probe = probe,band = 'ap')
        response = self.get_chunk(probe = probe, stim_times = stim_times, 
                                  pre = pre, post = post, median_subtraction = median_subtraction, 
                                  chs = np.arange(first_ch,last_ch))
        
        
        sample_rate = data.metadata['sample_rate']
        total_samps = int((pre/1000 * sample_rate) + (post/1000 * sample_rate))            
        if spike_overlay == True:
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            condition = (
                (self.units['ch'] >= first_ch) &
                (self.units['ch'] <= last_ch) &
                (self.units['probe'] == probe) &
                (self.units['group'] == 'mua')
            )

            spikes = np.array(self.units.loc[condition, 'spike_times'])
            spike_ch = np.array(self.units.loc[condition, 'ch'])

            spike_dict = {}
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))  
                window = data.timestamps[start_index:end_index]
                filtered_spikes = [spike_times[(spike_times >= window[0]) & (spike_times <= window[-1])] for spike_times in spikes]  
                spike_dict[i] = filtered_spikes

        ## plotting 
        
        trial_subset = np.linspace(0,len(stim_times)-1, n_trials) #choostrse random subset of trials to plot 
        trial_subset = trial_subset.astype(int)
    
        #set color maps
        cmap = sns.color_palette("crest",n_colors = n_trials)
        #cmap = sns.cubehelix_palette(n_trials)
        colors = cmap.as_hex()
        if spike_overlay == True:
            cmap2 = sns.color_palette("ch:s=.25,rot=-.25", n_colors = len(spikes))
            colors2 = cmap2.as_hex()
        fig=plt.figure(figsize=(16,24))
        time_window = np.linspace(-pre,post,(total_samps))
        for trial,color in zip(trial_subset,colors):
            for ch in range(0,int((last_ch - first_ch))): 
                plt.plot(time_window,response[trial,:,ch]+ch*spacing_mult,color=color)
        
            if spike_overlay == True:
                for i,ch in enumerate(spike_ch): 

                    if spike_dict[trial][i].size > 0:
                        for spike in spike_dict[trial][i]:
                            spike = spike - stim_times[trial]
                            plt.scatter(spike*1000, (spike/spike) + ((ch-(first_ch)))*spacing_mult, 
                            alpha = 0.5, color = colors2[i], s = 500)
        
        plt.gca().axvline(0,ls='--',color='r')       
        plt.xlabel('time from stimulus onset (ms)')
        plt.ylabel('uV')
        plt.title(title)
        
        if save == True:
            plt.gcf().savefig(savepath,format=format,dpi=600)
        
        return fig
    

## general functions
from open_ephys.analysis import Session
def OE(path):
    'lazy wrappy to return recording session from open ephys'
    session = Session(path)
    recording = session.recordnodes[0].recordings[0]
    return recording

def load_datastream(path, probe, band='ap'):
    '''
    purpose of this is to return a data_stream object. can then access metadata and load raw data from here.
    path: path to recording
    probe: probe name (probeA, probeB, probeC)
    '''
    # normalize case for flexibility
    probe = probe.lower()
    band = band.lower()

    # Get the recording
    recording = OE(path)

    # Search for the correct data stream by examining metadata
    for data_stream in recording.continuous:
        stream_name = data_stream.metadata['stream_name'].lower().replace('-', '_')
        if stream_name == f"{probe}_{band}":
            print(f'confirming stream name: {data_stream.metadata["stream_name"]}')
            return data_stream

    print('No matching bands or probes')
    return None

def get_chunk(path,
              probe, 
            stim_times,
            band = 'ap',
            pre = 100, # time in ms
            post = 500, # time in ms
            chs = np.arange(0,200,1), # channels
            median_subtraction = False,
            offset_subtraction = False,
            ):
    """
    for open ephys data
    Takes in a continuous datastream object (from open ephys) and a list of stimulation times and returns a chunk of the data

    return: data: np.array, shape = (trials, samples, channels)
    """
    
    data_stream = load_datastream(path, probe, band = band)
    sample_rate = data_stream.metadata['sample_rate']
    
    
    pre_samps = int((pre/1000 * sample_rate))
    post_samps = int((post/1000 * sample_rate))
    total_samps = pre_samps + post_samps

    n_chs = len(chs)
    
    response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
    stim_indices = np.searchsorted(data_stream.timestamps, stim_times)
    for i, stim in enumerate(stim_indices):
        start_index = int(stim - ((pre/1000)*sample_rate))
        end_index = int(stim + ((post/1000)*sample_rate))   
        chunk = data_stream.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                            selected_channels = chs)
        
        if median_subtraction == True:
            corrected_chunk = chunk - np.median(chunk, axis = 0) #subtract offset 
            corrected_chunk = np.subtract(corrected_chunk.T, np.median(corrected_chunk[:,chs[-10]:chs[-1]], axis=1)) #median subtraction

            response[i,:,:] = corrected_chunk.T
        elif offset_subtraction == True:
            response[i,:,:] = chunk - np.median(chunk, axis = 0)
        else:
            response[i,:,:] = chunk

    return response

def subtract_offset(data, subtraction_window = None, pre = None, post = None):
    '''
    takes a chunk of data and subtracts the median of the data from each channel

    data: np.array, shape = (trials, samples, channels)
    subtraction_window: 'pre', 'all'. If None, will use full.
    pre: pre window in ms
    post: post window in ms 
    '''
    if subtraction_window == 'pre':
        pre_samps = int((pre/1000 * 30000))
        window = (0, pre_samps)
        pre_data = data[:,window[0]:window[1],:]
        corrected_chunk = data - np.median(pre_data, axis = 1) # the samples axis
    else: # subtract the median of the entire chunk
        corrected_chunk = data - np.median(data, axis = 1)
    
    return corrected_chunk


def median_subtraction(data, channels):
    """Subtract the median of a window from the data.

    Args:
        data (np.array): The data to subtract from, shape = (trials, samples, channels).
        channels (list): The channels to subtract from.

    Returns:
        np.array: The data with the median subtracted.
    """

    corrected_data = data.copy()
    corrected_data[:, :, channels] = np.subtract(corrected_data[:, :, channels].T, np.median(corrected_data[:, :, channels], axis=1)).T

    return corrected_data


def find_artifact_start(data_ch, pre, threshold):
    """Find the start of artifact based on a threshold.
    
    Args:
        data_ch: np.array, shape = ch x sample
        threshold: The threshold used to detect the artifact. Function looks for 
                   the first value that exceeds this threshold in absolute terms.
                   
    Returns:
        The index of the first data point that exceeds the threshold.
    """
    # Use the absolute value to find the first occurrence beyond the threshold, 
    # regardless of the sign of the artifact.

    return next((i for i, val in enumerate(np.abs(data_ch)) if val > threshold), int((pre+2)/1000 * 30000))

def align_data(data, pre, post, channels, threshold = 400, median_subtraction = False):
    """ Align the data to the artifact onset based on artifact start times. Optionally perform median subtraction.

    Args:
        data (np.array): The data to align, shape = (trials, samples, channels)
        pre_samps (_type_): number of desired pre-samples (determined by pre time in ms) 
        post_samps (_type_): number of desired post-samples (determined by post time in ms)
        channels (_type_): number of channels to plot
        median_subtraction (_type_): whether to perform median subtraction on the data

    Returns:
        _type_: aligned_data, shape = (trials, total_samps, channels)
    """   
    starts = [find_artifact_start(data[trial, :, 0], pre = pre, threshold = threshold) for trial in range(data.shape[0])] # the sample number of when the artifact first starts 
    # the sample number of when the artifact first starts determined by the find_artifact_start function
    
    pre_samps = (int((pre/1000) * 30000)) 
    post_samps = (int((post/1000) * 30000))
    total_samps = pre_samps + post_samps
    aligned_data = np.zeros((data.shape[0], total_samps, data.shape[2]))
    
    for trial in range(data.shape[0]):
        start = starts[trial]
        chunk = data[trial, start - pre_samps:start + post_samps, :]
        if median_subtraction:
            corrected_chunk = chunk.T - np.median(chunk[:, channels-11:channels-1], axis=1)
            aligned_data[trial, :, :] = corrected_chunk.T
        else:
            aligned_data[trial, :, :] = chunk
    
    return aligned_data




def raw_heatmap(data, pre=1, post=2, dists=None, vmin=None, vmax=None, 
                save=False, save_path=None, save_type='png', title=None, ax=None):
    """
    Plots a heatmap of data with options for customization and saving.

    Args:
        data (np.array): The data to plot, shape = (trials, samples, channels).
        pre (int, optional): pre_window in ms. Defaults to 1.
        post (int, optional): post_window in ms. Defaults to 2.
        dists (np.array, optional): Distance from stimulation array, same shape as channels.
        vmin (float, optional): Minimum value for heatmap scaling. Defaults to None.
        vmax (float, optional): Maximum value for heatmap scaling. Defaults to None.
        save (bool, optional): If True, save the figure. Defaults to False.
        save_path (str, optional): Path to save the figure. Defaults to None.
        save_type (str, optional): Format of saved figure. Defaults to 'png'.
        title (str, optional): Figure title. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None (creates new figure).

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """

    # Check if an axes object was provided, if not create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        created_fig = True
    else:
        created_fig = False  # Flag to avoid creating a new figure

    channels = data.shape[2]
    data_to_plot = np.mean(data, axis=0).T  # average across trials

    time_ms = np.linspace(-pre, post, data.shape[1])

    cax = ax.imshow(data_to_plot, aspect='auto', extent=[time_ms[0], time_ms[-1], 0, channels],
                    vmax=vmax, vmin=vmin, origin='lower', cmap='vlag')

    if created_fig:
        fig.colorbar(cax, ax=ax, pad=0.20)
    ax.set_ylabel('Channel')
    ax.set_yticks(np.arange(0, channels, 10))

    if dists is not None:
        ax_dist = ax.twinx()
        ax_dist.set_ylim(ax.get_ylim())
        ax_dist.set_yticks(np.arange(0, len(dists), 25))
        ax_dist.set_yticklabels([int(d) for d in dists[::25]])
        ax_dist.set_ylabel('Distance from Stimulation (units)')
        zero_dist_channel = np.argmin(np.abs(dists))
        ax.axhline(y=zero_dist_channel, color='red', linestyle='--')

    ax.set_xlabel('Time (ms)')
    if title:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

    if save and created_fig:
        plt.savefig(save_path, format=save_type)

    return ax



def plot_ap(path, probe, stim_times, 
                pre = 5, post = 10, 
                first_ch = 125, last_ch = 175, 
                median_subtraction = False,
            
                spike_overlay = False,
                units = None,
                title = '', 

                n_trials = 10, spacing_mult = 350,
                ax = None, 
                save = False, savepath = '', format ='png'):

        '''
        path: recording path, 
        probe: probe (e.g., 'probeA', 'probeB', 'probeC')
        data: 3D array (trials x samples x channels)
        
        stim_times: list of stimulation times
        pre: pre window in ms
        post: post window in ms
        
        first_ch: first channel to plot
        last_ch: last channel to plot

        probeID: probe name ('A', 'B', 'C')
        spike_overlay: whether to overlay spike times from dataframe
        units: dataframe with spike times
        title: title of the plot

        n_trials: number of trials to plot (bc each trial is overlaid)
        spacing_mult: multiplier for spacing between channels
        save: whether to save the plot
        savepath: where to save the plot
        format: format of the saved plot (png, eps, etc)
        '''
        data_stream = load_datastream(path, probe)
        sample_rate = data_stream.metadata['sample_rate']
        data = get_chunk(path, probe, stim_times, 
                             pre = pre, post = post, 
                             chs =np.arange(0,300,1), 
                             median_subtraction = median_subtraction)
        

        probeID = probe.strip('probe')
        total_samps = int((pre/1000 * sample_rate) + (post/1000 * sample_rate))            
        
        if spike_overlay == True:
            stim_indices = np.searchsorted(data_stream.timestamps,stim_times)
            condition = (
                (units['ch'] >= first_ch) &
                (units['ch'] <= last_ch) &
                (units['probe'] == probeID) &
                (units['group'] == 'good')
            )
        
            spikes = np.array(units.loc[condition, 'spike_times'])
            spike_ch = np.array(units.loc[condition, 'ch'])

            spike_dict = {}
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))  
                window = data_stream.timestamps[start_index:end_index]
                filtered_spikes = [spike_times[(spike_times >= window[0]) & (spike_times <= window[-1])] for spike_times in spikes]  
                spike_dict[i] = filtered_spikes

        ## plotting 
        
        trial_subset = np.linspace(0,len(stim_times)-1, n_trials) #choostrse random subset of trials to plot 
        trial_subset = trial_subset.astype(int)
        #set color maps
        cmap = sns.color_palette("crest",n_colors = n_trials)
        #cmap = sns.cubehelix_palette(n_trials)
        colors = cmap.as_hex()
        if spike_overlay == True:
            cmap2 = sns.color_palette("ch:s=.25,rot=-.25", n_colors = len(spikes))
            colors2 = cmap2.as_hex()
        if ax is None:
            fig, ax =plt.figure(figsize=(6,8))
            
        time_window = np.linspace(-pre,post,(total_samps))
        for trial,color in zip(trial_subset,colors):
        
            for ch in range(first_ch,last_ch): 
                ax.plot(time_window,data[trial,:,ch]+ch*spacing_mult,color=color)
        
            if spike_overlay == True:
                for i,ch in enumerate(spike_ch): 

                    if spike_dict[trial][i].size > 0:
                        for spike in spike_dict[trial][i]:
                            spike = spike - stim_times[trial]
                            ax.scatter(spike*1000, (spike/spike) + ch*spacing_mult, 
                            alpha = 0.75, color = colors2[i], s = 500)
        
        ax.axvline(0,ls='--',color='r')       
        ax.set_xlabel('time from stimulus onset (ms)')
        ax.set_ylabel('uV')
        ax.set_title(title)
        
        if save == True:
            plt.gcf().savefig(savepath,format=format,dpi=600)