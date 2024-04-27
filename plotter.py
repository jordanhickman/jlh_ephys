import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import zscore
from scipy.stats import sem

from tqdm.notebook import tqdm

from jlh_ephys.psth_raster import psth_arr

class Plotter():
    def __init__(self, analysis_obj):
        self.analysis = analysis_obj
        if self.analysis.processed == True:
            self.trials = self.analysis.trials
            self.units = self.analysis.units
            self.probes = self.analysis.probes
            self.parameters = self.analysis.parameters
            self.path = self.analysis.path
            self.mouse = self.analysis.mouse
            self.date = self.analysis.date
            self.raw = self.analysis.raw
            

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

    def plot_all_goodunit_rasters(self,
                          pre=0.5, post=0.5,
                          save_path = None, save = False, save_type = 'png',
                          color='blue', linewidth=0.25, 
                          marker_size = 4, marker = 'o',
                          labelsize=10, axis_labelsize=8):
        
        try:
            self.analysis.get_zetas()
            self.analysis.assign_zeta_sig()
            self.units = self.analysis.units
        except Exception as e:
            print(f"Could not obtain zetascores. Error: {e}")

        # Sort runs by amplitude
        sorted_runs = sorted(self.trials['run'].unique(), key=lambda x: abs(self.trials[self.trials['run'] == x]['amplitude'].iloc[0]))
        
        n_conditions = len(sorted_runs)
        n_rows = (n_conditions // 5) + int(n_conditions % 5 > 0)
        
        for unit_id in tqdm(self.units[self.units['group'] == 'good'].index):
            f, axs = plt.subplots(n_rows, 5, figsize=(24, 20))
            f.suptitle(f"Unit ID: {unit_id}, Ch: {self.units.loc[unit_id, 'ch']}", fontsize=18, y=0.95)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
            
            for i, ax in enumerate(axs.ravel()[:n_conditions]):
                run = sorted_runs[i]
                stim_time = self.trials.loc[self.trials['run'] == run, 'start_time']
                unit_spikes = self.units.loc[unit_id, 'spike_times']
                
                
                # Plotting raster
                self.raster(unit_spikes, stim_time, pre=pre, post=post, 
                            color=color, marker = marker, 
                            linewidth=linewidth,
                            marker_size = marker_size,
                            labelsize=labelsize, 
                            axis_labelsize=axis_labelsize, 
                            axes=ax)
                
                # Add text and other details
                ax.set_title(self.parameters[run], fontsize=labelsize)
                zeta = self.units.loc[unit_id, f'r{run}']
                if zeta is not None:
                    ax.text(0.9,0.93, zeta, transform=ax.transAxes, c='r')
                
            if save:
                if save_path is None:
                    save_path = os.path.join(self.path, 'raster_plots')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/{unit_id}_rasters.{save_type}")
            plt.show()
    
    def plot_vertical_histogram(self, ax, probe):
        # Filter out rows where group is 'nan'
        filtered_units = self.units[self.units['group'] != 'nan']
        subset = filtered_units[filtered_units['probe'] == probe].copy()

        # Create a new column for channel bins
        subset['ch_bin'] = (subset['ch'] // 10) * 10

        # Count the occurrences of each (ch_bin, group) pair
        counted_subset = subset.groupby(['ch_bin', 'group']).size().reset_index(name='count')

        # Make the count negative for the 'mua' group to plot them on the left
        counted_subset['count'] = np.where(counted_subset['group'] == 'mua', -counted_subset['count'], counted_subset['count'])

        # Create the bar plot
        sns.barplot(data=counted_subset, y='ch_bin', x='count', hue='group', orient='h', ax=ax)

        ax.set_title(f'Probe {probe}')
        ax.legend(loc='lower left')
        ax.invert_yaxis()
        ax.set_xlabel('Number of Units')
        ax.set_ylabel('Channel (binned by 10)')
        return ax

    
    def summary_plot(self, save = True):
        from matplotlib.gridspec import GridSpec
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        import seaborn as sns

        '''
        First Row: Text details
        Second Row: Unit Quality Histograms and DataCube
        Third row: LFP power bands
        Final 2 Rows: Rasters for each unique run
        TODO: add LFP gamma power plot
        '''
        plt.style.use('dark_background')
        # Main grid layout with 4 rows
        main_gs = GridSpec(5, 1, height_ratios=[0.5, 2, 3, 8, 8])

        # Create the main figure
        fig = plt.figure(figsize=(20, 40))

        ### first row: text descriptions 
        text_ax = fig.add_subplot(main_gs[0])
        text_ax.axis('off') # Turn off the axis 
        text_ax.text(0.05, 0.8, f'{self.mouse} Summary       {self.date}', fontsize = 16, fontweight = 'bold')
        probe_spacer = 0.50
        for probe in self.probes:
            probe_ID = probe.strip('probe')
            good = 'good'
            text_ax.text(0.1, probe_spacer, f'probe{probe_ID} Good Units: {len(self.units[(self.units.group == good) & (self.units.probe == probe_ID)])}', fontsize = 12)
            probe_spacer -= 0.25

        ### second row
        # Subgrid for units_quality (2x2) and trials_data_cube (one row for each polarity)
        gs_upper_row = GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[1], width_ratios = [2,1])
        
        # Subgrid for units_quality (2x2)
        gs_units_quality = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_upper_row[0, 0], wspace = 0.3, hspace = 0.7)
        # Subgrid for trials_data_cube (one row for each polarity)
        def combination_exists(group):
            group['exists'] = 1
            return group
        color_schemes = ['viridis', 'plasma', 'inferno']
        trials_with_existence = self.trials.groupby(['amplitude', 'pulse_duration', 'polarity']).apply(combination_exists)
        polarities = trials_with_existence['polarity'].unique()
        
        gs_trials_data_cube = GridSpecFromSubplotSpec(len(polarities), 1, subplot_spec=gs_upper_row[0, 1], wspace = 0.5, hspace = 0.5)



        # (Code for units_quality and trials_data_cube goes here...)
        # Plot the units_quality
        units_reset_index = self.units.reset_index()
        IDs = units_reset_index['probe'].unique()
        y_max = 80
        bins = 40

        for index, probe in enumerate(IDs):
            row = 0
            col = index
            ax_unit_quality = plt.subplot(gs_units_quality[row, col])
            ax_unit_quality = self.plot_vertical_histogram(ax_unit_quality, probe)
            """
            subset = units_reset_index[units_reset_index['probe'] == probe]
            subset = units_reset_index[units_reset_index['group'] != 'nan']
            sns.histplot(data=subset, x='no_spikes', hue='group', bins=bins, element='bars', common_norm=False, palette='viridis', alpha=0.6, ax=ax_unit_quality)
            ax_unit_quality.set_title(f'Distribution of Spike Counts for Probe {probe}', fontsize = 13)
            ax_unit_quality.set_xlabel('Spike Counts', fontsize = 13)
            ax_unit_quality.set_ylabel('Number of Units', fontsize = 13)
            ax_unit_quality.set_ylim(0, y_max)
            ax_unit_quality.tick_params(labelsize=8, rotation =45)
            """
        # Find the appropriate axes for the last subplot
        ax_last = plt.subplot(gs_units_quality[0, 3])
        # Plot the bar plot for unit group counts in the last subplot
        sns.countplot(data=self.units[self.units['group'] != 'nan'], x='probe', hue='group', ax=ax_last)
        ax_last.set_title('Unit Group Counts by Probe', fontsize = 13)
        ax_last.set_xlabel('Probe', fontsize = 13)
        ax_last.set_ylabel('Count', fontsize = 13)
        ax_last.tick_params(labelsize=12, rotation =45)
        ax_last.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # LFP Power!
        try:
            gs_lfp = GridSpecFromSubplotSpec(1, 6, subplot_spec=main_gs[2])

            gamma_axes = [fig.add_subplot(gs_lfp[0, i]) for i in range(3)]
            beta_axes = [fig.add_subplot(gs_lfp[0, i+3]) for i in range(3)]

            # Generate the plots
            self.raw.plot_average_LFP_power(axes=gamma_axes, band='gamma')
            self.raw.plot_average_LFP_power(axes=beta_axes, band='beta')
        except:
            print('Could not access LFP, skipping...')
            skip = True 

        ## trials 
        # Iterate through the polarities and plot the heatmaps

        for index, polarity in enumerate(polarities):
            ax2 = plt.subplot(gs_trials_data_cube[index, 0])
            subset = trials_with_existence[trials_with_existence['polarity'] == polarity]
            pivot_table = subset.pivot_table(values='exists', index='pulse_duration', columns='amplitude', fill_value=0)
            x_labels = [f"{x} ua" for x in pivot_table.columns]
            y_labels = [f"{y} us" for y in pivot_table.index]
            sns.heatmap(pivot_table, cmap=color_schemes[0], annot=False, cbar=True, linewidths=0.5, alpha=0.8, square=False, ax=ax2)
            ax2.set_title(f'Heatmap for {polarity} Polarity', fontsize=15)
            ax2.set_xlabel('Amplitude', fontsize=14)
            ax2.set_ylabel('Pulse Duration', fontsize=14)
            ax2.set_xticklabels(x_labels, fontsize=13, rotation=45) # Rotate x-axis tick labels
            ax2.set_yticklabels(y_labels, fontsize=13, rotation=45) # Rotate y-axis tick labels


        #################################################
        # Grid specification for the raster plot (2/3 height, divided into two columns)
        gs_raster = GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[3:])

        # Define the axes for the two columns of the raster plot
        ax_raster_left = plt.subplot(gs_raster[0, 0])
        ax_raster_right = plt.subplot(gs_raster[0, 1])

        # Code to generate the stacked raster plot goes here
        # Time window
        window_start = -0.1
        window_end = 0.3
        gap_size = 20

        # Split runs into two columns
       
        amplitude_by_run = self.trials.groupby('run')['amplitude'].first()
        # Sort the runs by amplitude
        sorted_runs_by_amplitude = amplitude_by_run.sort_values().index.values        
        unique_runs = sorted_runs_by_amplitude
        #unique_runs = self.trials['run'].unique()
        runs_per_column = np.array_split(unique_runs, 2)

        good_units = self.units.loc[self.units.group == 'good']
        good_units = good_units.sort_values(by = 'ch', ascending = False).reset_index()

        # Iterate over the two columns
        for col_index, runs in enumerate(runs_per_column):
            ax_raster = ax_raster_left if col_index == 0 else ax_raster_right

            plot_spikes = []
            plot_units = []
            plot_colors = []

            # Create a color map for the runs using the "viridis" colormap
            colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))

            # Iterate through the runs and collect the data
            for run_index, run in enumerate(runs):
                run_trials = self.trials[self.trials['run'] == run]
                color = colors[run_index]

                # Iterate through the units

                for unit_index, unit in good_units.iterrows():
                    spike_times = np.array(unit['spike_times'])

                    # Find spikes within the window for all trials at once
                    for start_time in run_trials['start_time']:
                        relative_spikes = spike_times - start_time
                        spikes_in_window = relative_spikes[(relative_spikes >= window_start) & (relative_spikes <= window_end)]
                        plot_spikes.extend(spikes_in_window)
                        plot_units.extend([unit_index + run_index * (len(good_units)+gap_size)] * len(spikes_in_window))
                        plot_colors.extend([color] * len(spikes_in_window))

            # Plot data for this subplot
            ax_raster.scatter(plot_spikes, plot_units, c=plot_colors, s=0.2)

            # Add text annotations for each run with white color
            for run_index, run in enumerate(runs):
                run_trials = self.trials[self.trials['run'] == run]
                amplitude = run_trials['amplitude'].iloc[0]
                pulse_duration = run_trials['pulse_duration'].iloc[0]
                polarity = run_trials['polarity'].iloc[0]
                label = f'Amplitude: {amplitude} ua\nPulse Duration: {pulse_duration} us\nPolarity: {polarity}'
                y_position = (run_index * (len(good_units)+gap_size) + (run_index + 1) * len(good_units)) / 2
                ax_raster.text(window_end + 0.01, y_position, label, fontsize=10, verticalalignment='center', color='white')

            # Add vertical line at time 0
            ax_raster.axvline(x=0, color='red', linestyle='--')

            # Add x-axis label and title
            ax_raster.set_xlabel('Time (s)')

            # Turn off y-axis ticks
            ax_raster.tick_params(left=False, labelleft=False)

            # Remove box
            for spine in ax_raster.spines.values():
                spine.set_visible(False)

        # Set title for the stacked rasters
            if col_index == 0:
                ax_raster.set_title('Stacked Rasters for All Runs', fontsize = 12)
        
        if save:
            plt.savefig(os.path.join(self.path, f'{self.mouse}Summary.png'))

        plt.tight_layout()
        plt.show()

    
    
    def dist_hist(self, run = None, all = False, save = False):
        if run is None and not all:
            mask = (self.analysis.trials.amplitude == -50) & (self.analysis.trials.pulse_duration == 100)
            run = self.analysis.trials[mask].run.iloc[0]
            ax = sns.histplot(data = self.analysis.units, 
                            x = 'distance_from_stim', binwidth = 50,
                            hue = f'r{run}', multiple = 'stack',
                            hue_order = ['non-sig','sig'])
            ax.set(xlabel='Distance from stimulation', 
                ylabel='Number of Units', 
                title = self.analysis.parameters[run])
            if save:
                plt.savefig(os.path.join(self.analysis.path, f'dist_hist_run{run}.eps'))
                plt.savefig(os.path.join(self.analysis.path, f'dist_hist_run{run}.png'))
        elif run is None and all:
            for run in self.analysis.trials.unique():
                ax = sns.histplot(data = self.analysis.units, 
                            x = 'distance_from_stim', binwidth = 50,
                            hue = f'r{run}', multiple = 'stack',
                            hue_order = ['non-sig','sig'])
                ax.set(xlabel='Distance from stimulation', 
                      ylabel='Number of Units', 
                      title = self.analysis.parameters[run])
                if save:
                    plt.savefig(os.path.join(self.analysis.path, f'dist_hist_run{run}.eps'))
                    plt.savefig(os.path.join(self.analysis.path, f'dist_hist_run{run}.png'))
        else:
            ax = sns.histplot(data = self.analysis.units, 
                            x = 'distance_from_stim', binwidth = 50,
                            hue = f'r{run}', multiple = 'stack',
                            hue_order = ['non-sig','sig'])
            ax.set(xlabel='Distance from stimulation', 
                ylabel='Number of Units', 
                title = self.analysis.parameters[run])
            if save:
                    plt.savefig(os.path.join(self.analysis.path, f'dist_hist_run{run}.eps'))
                    plt.savefig(os.path.join(self.analysis.path, f'dist_hist_run{run}.png'))

    ## heatmap by distance and brainregion functions
            
    def units_channel_heatmap(self,probeID, run, pre=0.5, post=0.5, binsize = 0.002, vmax=20, save=True, save_path='', ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))
            
        
        cmap = sns.color_palette("crest", as_cmap=True).reversed()
        
        _units = self.units[self.units['probe'] == probeID].sort_values('ch', ascending=True)
        stimtimes = np.array(self.trials[self.trials.run == run].start_time)

        psths_ch = []
        unit_ind = []
        channel_ticks = []
        max_ch = _units.ch.max()
        ch_list = np.arange(0, max_ch, 1)

        un_count = 0
        for ch in ch_list[::-1]:
            
            matching_units = _units[_units.ch == ch]

            if matching_units.empty:
                psth, _, _ = psth_arr(np.array([0]), stimtimes, pre=pre, post=post, binsize=binsize)
                psths_ch.append(psth)
                un_count+=1
                
                unit_ind.append(un_count)
                channel_ticks.append(ch)
            
            else:
                for _, unit in matching_units.iterrows():
                    spiketimes = np.array(unit['spike_times'])
                    psth, _, _ = psth_arr(spiketimes, stimtimes, pre=pre, post=post, binsize=binsize)
                    psths_ch.append(psth)
                    
                    un_count += 1

                    unit_ind.append(un_count)
                    channel_ticks.append(ch)

        # Visualization
        im = ax.imshow(psths_ch, vmax=vmax, aspect='auto', cmap=cmap, extent=[-pre*1000, post*1000, 0, len(psths_ch)])

        ax.set_yticks(unit_ind[::20]) 
        ax.set_yticklabels(channel_ticks[::-20])
        
        ax.set_ylabel('Channel')
        ax.set_xlabel('Time (ms)')
        ax.axvline(x=0, color='r', linestyle='--')  # Vertical line at time zero for stimulation

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Firing Rate')
        
        if save:
            plt.savefig(os.path.join(save_path, f'Probe{probeID}_ch_heatmap.eps'), dpi=600)

        return ax
    
    def units_distance_heatmap(self, probeID, run, pre = 0.5, post = 0.5, vmax = 20, contact = 4, save=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        cmap = sns.color_palette("crest", as_cmap=True).reversed()
        
        dist_dict = self.analysis.get_dists(contact = contact) # gets a dictionary of distances for each probe to the selected contact. 
        dists = dist_dict[f'probe{probeID}'] # ordered deep to superficial

        self.analysis.get_brain_regs()
        brainreg = self.analysis.brain_regs[f'probe{probeID}'] # ordered deep to superficial


        _units = self.units[self.units['probe'] == probeID].sort_values('ch', ascending=False)
        stimtimes = np.array(self.trials[self.trials.run == run].start_time)

        psths_ch = []
        distance_ticks = []
        un_count = 0
        unit_ind = []
        channel_ticks = []
        
        for ch in range(len(brainreg) - 1, -1, -1):  #iterates over a range of integers in reverse, starting from len(brainreg) - 1 and going down to 0
            matching_units = _units[_units.ch == ch]

            if matching_units.empty:
                psth, _, _ = psth_arr(np.array([0]), stimtimes, pre=pre, post=post, binsize=0.002)
                psths_ch.append(psth)
                un_count+=1
                unit_ind.append(un_count)
                current_distance = int(dists[ch])
                distance_ticks.append(current_distance)
                channel_ticks.append(ch)
            else:
                for _, unit in matching_units.iterrows():
                    spiketimes = np.array(unit['spike_times'])
                    psth, _, _ = psth_arr(spiketimes, stimtimes, pre=pre, post=post, binsize=0.002)
                    psths_ch.append(psth)
                    un_count+=1
                    unit_ind.append(un_count)
                    current_distance = int(dists[ch])
                    distance_ticks.append(current_distance)
                    channel_ticks.append(ch)

                # Determine the time bins
            pre_time_ms = -pre * 1000
            post_time_ms = post * 1000

        # Replace y-ticks with distance_ticks and adjust labels
        im = ax.imshow(psths_ch, vmax=vmax, aspect='auto', cmap=cmap, extent=[pre_time_ms, post_time_ms, 0, len(psths_ch)])
        
        """
        min_distance_idx = np.argmin(distance_ticks)
        # Create a range of indices for ticks centered around the minimum distance value
        tick_indices = np.arange(max(0, min_distance_idx - 5), min(len(unit_ind), min_distance_idx + 6))

        ax.set_ylabel('Distance from Stimulation')
        ax.set_yticks([unit_ind[idx] for idx in tick_indices])
        ax.set_yticklabels([distance_ticks[idx] for idx in tick_indices])
        """
        ax.set_yticks(unit_ind[::20]) 
        ax.set_yticklabels(distance_ticks[::-20]) # Reverse the tick labels
        
        
        ax.set_ylabel('Distance from Stimulation')
        ax.set_xlabel('Time (ms)')
        ax.axvline(x=0, color='r', linestyle='--')  # Vertical line at time zero for stimulation

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Firing Rate')
        if save:
            plt.savefig(os.path.join(self.path, f'Probe{probeID}dist_heatmap.eps'), dpi = 600)

        return ax, psths_ch, channel_ticks, distance_ticks, unit_ind 
        ## channel and distance ticks orders are switched so that they start with ch0... easier for indexing into brainregs
        ##Todo: distance calculations are a little wonky
    
    def calculate_modulation_index(self, psths_ch):
        psths_ch = np.array(psths_ch)
        baseline_window = range(0, 240)  # -500 to -2ms
        early_activation_window = range(251, 257)  # 0-12ms
        suppression_window = range(261, 326)  # 20-150ms
        late_activation_window = range(351, 451)  # 200-400ms

        def compute_modulation_index(response_window):
            baseline_rate = psths_ch[:, baseline_window].mean(axis=1)
            response_rate = psths_ch[:, response_window].mean(axis=1)
            
            # Check for zero rates to avoid division by zero
            zero_rate_indices = np.isclose(response_rate + baseline_rate, 0.0)
            
            mi = np.zeros_like(response_rate)
            
            # Calculate MI where the sum of response and baseline rates is not zero
            non_zero_rate_indices = ~zero_rate_indices
            mi[non_zero_rate_indices] = (response_rate[non_zero_rate_indices] - baseline_rate[non_zero_rate_indices]) / \
                                        (response_rate[non_zero_rate_indices] + baseline_rate[non_zero_rate_indices])
            
            # Set MI to zero where the sum of response and baseline rates is zero
            mi[zero_rate_indices] = 0.0

            return mi

        return {
            'early_activation': compute_modulation_index(early_activation_window),
            'suppression': compute_modulation_index(suppression_window),
            'late_activation': compute_modulation_index(late_activation_window),
        } 
    
    
    def plot_one_window(self, ax, rates, title, smoothing_sigma=5, line_color = '#2c3071', line_width = 4):
        """
        Plot a single window of smoothed firing rates on a given axis.
        """
        
        smoothed_rates = gaussian_filter1d(rates, smoothing_sigma)
        ax.plot(smoothed_rates, range(len(smoothed_rates)), color=line_color, linewidth=line_width)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlabel('Modulation Index')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlim([-1, 1])  # Setting x-axis limits
        ax.set_xticks([-1, 0, 1])  # Setting x-axis tick positions
    
    def plot_normalized_firing_rates_with_regions(self, axs, probeID, normalized_rates, channel_ticks, distance_ticks, unit_ind, smoothing_sigma=5):
        """
        Plot the firing rates using multiple subplots (axs), one for each window ('early_activation', 'suppression', 'late_activation').
        """
        windows = ['early_activation', 'suppression', 'late_activation']
        titles = [win.replace('_', ' ').title() for win in windows]

        # Getting unique regions and their middle ticks
        self.analysis.get_brain_regs()
        brainreg = self.analysis.brain_regs[f'probe{probeID}']  # ordered deep to superficial
        
        regions = [brainreg[ch] for ch in channel_ticks]

        # Find unique regions and compute their median position for labeling
        unique_regions = list(set(regions))
        median_ticks = {}
        for region in unique_regions:
            indices = [i for i, x in enumerate(regions) if x == region]
            median_ticks[region] = int(np.median(indices))
        
        # Find unique regions and compute their starting and ending indices for shading
        region_indices = {}
        for region in unique_regions:
            indices = [i for i, x in enumerate(regions) if x == region]
            region_indices[region] = (min(indices), max(indices))

        # Different shades of gray in cycles of 5
        gray_shades = [0.2, 0.4, 0.6, 0.8, 1.0] * (len(unique_regions) // 5 + 1)

        for idx, win in enumerate(windows):
            self.plot_one_window(axs[idx], normalized_rates[win], titles[idx], smoothing_sigma=smoothing_sigma)

            # Add shaded regions for each unique brain region with different shades of gray
            for i, (region, (start_idx, end_idx)) in enumerate(region_indices.items()):
                axs[idx].axhspan(start_idx, end_idx, facecolor=str(gray_shades[i]), alpha=0.5)

            # Remove box around the plot
            axs[idx].spines['top'].set_visible(False)
            axs[idx].spines['right'].set_visible(False)
            axs[idx].spines['bottom'].set_visible(False)
            axs[idx].spines['left'].set_visible(False)

        # Hide y-axis
        for ax in axs:
            ax.yaxis.set_visible(False)

        # Adding a secondary axis for the brain regions
        ax2 = axs[2].twinx()
        ax2.set_ylim(axs[2].get_ylim())
        ax2.set_yticks(list(median_ticks.values()))
        ax2.set_yticklabels(list(median_ticks.keys()), fontsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)


        plt.tight_layout()

    def integrated_heatmap_plot(self, probeID, run,  pre=0.5, post=0.5, vmax=20, contact=4, smoothing_sigma=5, save=True):
        # Create a figure
        fig = plt.figure(figsize=(20, 6))
        
        # Create a gridspec to customize subplot sizes
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 1, 1])  # 3x width for the heatmap, 1x for each firing rate plot
        
        # Generate the heatmap in the first subplot
        ax0 = plt.subplot(gs[0])
        ax, psths_ch, channel_ticks, distance_ticks, unit_ind = self.units_distance_heatmap(
            probeID, run, pre, post, vmax, contact, ax=ax0
        )
        # get the normalized firing rates for each epoch
        normalized_rates = self.calculate_modulation_index(psths_ch)
        # Generate the firing rate plots in the remaining subplots
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[3])
        
        self.plot_normalized_firing_rates_with_regions(
            [ax1, ax2, ax3], probeID, normalized_rates, channel_ticks, distance_ticks, unit_ind, smoothing_sigma
        )
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.path, f'Integrated_Plot_Probe{probeID}.eps'), dpi=600)
        return fig, [ax0, ax1, ax2, ax3]
    




# general functions (maybe to add to DLAB) 

def raster(times, triggers, pre=0.25, post=0.5, color = 'blue', 
            linewidth = 0.5, 
            labelsize = 8, 
            axis_labelsize = 2, 
            marker = 'o',
            marker_size = 5,  
            axes=None):
    
    '''
    times: list of spike times
    triggers: list of event times (e.g. stimulus onset)
    pre: time before event (in seconds)
    post: time after event (in seconds)
    color: color of the raster
    linewidth: width of the raster lines
    labelsize: size of the axis labels
    axis_labelsize: size of the axis labels
    marker: marker style  **NOTE '|' does not seem compatible with some seaborn styles. hm. 
    marker_size: size of the markers
    axes: matplotlib axes object
    '''
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

def stacked_rasters(units_df, triggers, window_start=-0.01, window_end=0.02, marker_size=1, c = 'blue', title = None, save = False, save_path = None):

    '''
    plot stacked raster where unique units are their own line... does collapse trials! 
    looks to plot by ccf_coordinates and brain regions, otherwise uses channels. 
    best if units_df represents a single probe. 


    units_df: dataframe with spike times, ch, and brain region
    triggers: list of event times (e.g. stimulus onset)
    window_start: time before event (in seconds)
    window_end: time after event (in seconds)
    marker_size: size of the markers
    c: color of the raster

    '''
    fig, ax1 = plt.subplots(figsize=(8, 10))
    #fig.patch.set_facecolor('white')
    #fig.patch.set_edgecolor('white')
    plot_spikes = []
    plot_units = []
    plot_colors = []

    if 'ccf_coordinates' in units_df.columns:
        # makes dv column from ccf coordinates (assumes ap, dv, ml structure) )
        units_df['dv'] = units_df['ccf_coordinates'].apply(lambda x: x[1] if isinstance(x, np.ndarray) and x.size > 1 else None)
        
        # for units with the same dv, provides an offset so they don't overlap
        grouped = units_df.groupby('dv')
        fraction_increment = 1 / (grouped['dv'].transform('count'))
        units_df['unique_dv'] = units_df['dv'] + fraction_increment * grouped.cumcount()
        units_df = units_df.sort_values(by = 'unique_dv')
        ccf = True
        ch = False
    else:
        # just uses channels if no DV
        grouped = units_df.groupby('ch')
        fraction_increment = 1 / (grouped['ch'].transform('count'))
        units_df['unique_ch'] = units_df['ch'] + fraction_increment * grouped.cumcount()
        units_df = units_df.sort_values(by = 'unique_ch')
        channels = units_df['ch'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, np.max(channels)))
        color_map = dict(zip(channels, colors))
        ch = True
        ccf = False

    # Create a color map for brain regions
    if 'brain_reg' in units_df.columns:
        brain_regions = units_df['brain_reg'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(brain_regions)))
        color_map = dict(zip(brain_regions, colors))
        brain_reg = True
    else: 
        brain_reg = False

    # Plotting loop
    for _, unit in units_df.iterrows():
        spikes = np.array(unit['spike_times'])
        
        if ccf: y = unit['unique_dv']
        if ch: y = unit['unique_ch']

        if brain_reg:
            brain_region = unit['brain_reg']
            color = color_map[brain_region]
        elif ch:
            color =color_map[unit['ch']]

        else: 
            color = c 

        for trigger in triggers:
            relative_spikes = spikes - trigger
            spikes_in_window = relative_spikes[(relative_spikes >= window_start) & (relative_spikes <= window_end)]
            plot_spikes.extend(spikes_in_window)
            plot_units.extend([y] * len(spikes_in_window))
            plot_colors.extend([color] * len(spikes_in_window))

    ax1.scatter(plot_spikes, plot_units, c=plot_colors, s=marker_size, marker = '|')
    ax1.set_xlabel('Time (s)')
    if ccf:
        ax1.set_ylabel('Depth (DV in CCF)')
        ax1.invert_yaxis()
    elif ch:
        ax1.set_ylabel('Ch')
    ax1.set_title(title)
    
    
    if brain_reg:
        # Create a secondary y-axis for brain region labels
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Ensure both y-axes have the same scale

        # Determine positions for brain region labels
        for brain_region in brain_regions:
            median_dv = units_df[units_df['brain_reg'] == brain_region]['dv'].median()
            ax2.text(window_end + 0.001, median_dv, brain_region, verticalalignment='center')

        ax2.set_yticks([])  # Hide the y-ticks on the secondary axis
        
    plt.tight_layout()
    if save == True:
        plt.savefig(save_path+'.eps', format='eps', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.savefig(save_path+'.png', format='png', bbox_inches='tight', facecolor=fig.get_facecolor())


