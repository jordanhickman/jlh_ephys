import numpy as np



def spike_probability(spike_times, stim_times, window = 0.003, min_spike_count = 0):
    # find how many spikes occur betwween the first stimulus and the last stimulus) 
    first_stim_time = stim_times[0]
    last_stim_time = stim_times[-1]
    spike_count = np.sum((spike_times >= first_stim_time) & (spike_times <= last_stim_time))
    
    if spike_count > min_spike_count:
        # Count spikes within 3 ms of any stim time
        spike_counts_within_window = sum(
            np.any((spike_times <= stim + window) & (spike_times >= stim)) for stim in stim_times
        )
        spike_counts_baseline = sum(
            np.any((spike_times <= stim - (0.003+window)) & (spike_times > stim - (0.006 + window))) for stim in stim_times
        )
        # Probability is the count of spikes within the window divided by the total number of stimulations
        probability = spike_counts_within_window / len(stim_times)
        probability_baseline = spike_counts_baseline / len(stim_times)
        norm_prob = probability - probability_baseline
        
    else:
        probability = np.nan
        probability_baseline = np.nan
        norm_prob = np.nan
    return probability, probability_baseline, norm_prob