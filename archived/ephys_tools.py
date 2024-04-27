## set of ephys analysis functions designed to work with openephys data ()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,glob
import seaborn as sns
import pickle as pkl
from tqdm.notebook import tqdm as tqdm

from open_ephys.analysis import Session
import pendulum
import re


from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

from datetime import datetime
from dateutil.tz import tzlocal

from dlab.nwbtools import option234_positions,load_unit_data,make_spike_secs
from ccf_3D.tools import herbs_processing as hp

from pendulum import parse


## set of ephys analysis functions designed to work with openephys data ()
class Loader:
    def __init__(self, mouse, date):
        self.mouse = mouse
        self.date = date
        self.path = self.find_folder()
        self.check_for_nwb()


    #loading functions
    def find_folder(self):
        locations = [r'E:\\', r'C:\Users\hickm\Documents']
        for location in locations:
            search_pattern = os.path.join(location, f'*{self.mouse}*{self.date}*')
            matching_folders = glob.glob(search_pattern)
            if matching_folders:
                return matching_folders[0]
        return None
                

    def check_for_nwb(self):
        nwb_path = glob.glob(os.path.join(self.path,'*.nwb'))
        if nwb_path:
            if os.path.exists(nwb_path[0]):
                self.trials, self.units = self.load_nwb()
                try:
                    self.parameters = self.stim_dictionary()
                except:
                    print('Could not load parameter dictionary')
                    self.parameters = None
                self.processed = True
                print('NWB found. Trials and Units loaded')
        else:
            self.trials = None
            self.units = None
            self.processed = False 
            print("No .nwb files found in the specified directory.")

    def load_nwb(self, return_nwb = False):
        nwb_path = glob.glob(os.path.join(self.path,'*.nwb'))[0]
        io = NWBHDF5IO(nwb_path, 'r')
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        units = nwb.units.to_dataframe()
        
        if return_nwb == False:
            io.close()
            return trials, units
            
        else:
            return trials, units, nwb 
    
    def stim_dictionary(self):
        parameters = {}
        for run in self.trials.run.unique():
            amp = np.array(self.trials.loc[self.trials.run == run].amplitude)[0]
            pulse_width = np.array(self.trials.loc[self.trials.run == run].pulse_duration)[0]
            contacts = np.array(self.trials.loc[self.trials.run == run].contacts)[0]
            parameters[int(run)] = f'amp: {amp} ua, pw: {pulse_width} us, contacts: {contacts}'
        return parameters
    
class Preprocess:
    def __init__(self, Analysis):
        self.analysis = Analysis
        self.path = self.analysis.path
        self.mouse = self.analysis.mouse
        self.date = self.analysis.date
        self.stim_df = self.get_stim_df()
        
        
    
    def sort_dbs_runs(self, paths):
        new_names = []
        for path in paths:
            session_time_string = os.path.basename(path).split('_')[-1]
            if len(session_time_string.split('_')[-1]) < 2:
                hour = '0'+ session_time_string.split('_')[-1]
            else: hour = session_time_string.split('_')[-1]
            if len(session_time_string.split('-')[0]) < 2:
                minute = '0'+ session_time_string.split('-')[0]
            else: minute = session_time_string.split('-')[0]
            if len(session_time_string.split('-')[1]) < 2:
                second = '0'+ session_time_string.split('-')[1]
            else: second = session_time_string.split('-')[1]
            #         os.path.basename(path).split('-')[0]+'-'+
            new_names.extend([hour+'_'+minute+'_'+second])
        return np.array(paths)[np.argsort(new_names).astype(int)]
     
    def get_runs(self):
        recording = Analysis.OE(self.path)
        ## load events dataframe
        df = recording.events
        ## get timestamps for each run into a dictionary
        
        run = np.array(df.loc[df.line == 4].loc[df.state == 1].timestamp)
        stim = np.array(df.loc[df.line == 5].loc[df.state == 1].timestamp)
        print(f'# of Run_Triggers: {len(run)}')
        return run, stim
    
    def add_stims_to_run(self, run, stim):
        stim_dict = {}
        key_counter = 0
        # Get all the run timestamps except for the last run
        for i in range(len(run) - 1):
            run_start = run[i]
            run_end = run[i + 1]
            stim_for_run = [p for p in stim if run_start < p < run_end]
            if stim_for_run:  # Only add to dict if there are stims
                stim_dict[key_counter] = stim_for_run
                key_counter += 1
            else:
                print(f'Run Trigger # {i} not asscoiated with stim')

        # Last run
        last_run_start = run[-1]
        last_stim = [p for p in stim if last_run_start < p]
        if last_stim:  # Only add to dict if there are stims
            stim_dict[key_counter] = last_stim
        print(f'Run Triggers with Estim: {len(stim_dict)}')

        return stim_dict



    def parse_stim_csvs(self, stim_dict):
        formatted_date = pendulum.parse(self.date, strict = False).format("YYYYMMDD")
        csv_path = os.path.join(r'C:\Users\hickm\Documents\Stim_CSVs',f'{formatted_date}_{self.mouse}')

        extension = 'csv'
        result = glob.glob(os.path.join(csv_path, '*.{}'.format(extension)))
        sorted_dbs_runs = self.sort_dbs_runs(result)
        print(f'Number of Stim_CSVs: {len(sorted_dbs_runs)}')

        stim_df = pd.DataFrame()
        pd.read_csv(sorted_dbs_runs[0])
        for i, run in enumerate(sorted_dbs_runs):
            lil = pd.read_csv(run)
            lil['Run'] = i    
            #lil = pd.concat([lil] * int(lil.TrainQuantity)).sort_index().reset_index(drop=True)
            lil = pd.concat([lil] * int(len(stim_dict[i])))
            stim_df = pd.concat([stim_df,lil],ignore_index=True)

        concat_df = pd.concat([pd.read_csv(runs) for runs in sorted_dbs_runs])
        concat_df.to_csv(os.path.join(self.path, f'{self.mouse}.csv'))

        return stim_df
    
    def verify_trials(self, stim_df, stim_dict):
        trial_list = []
        for run, trials in stim_dict.items():
            for trial in trials:
                trial_list.append(trial)

        if len(trial_list) == len(stim_df):
            stim_df['stim_time'] = trial_list
            stim_df.to_csv(f'{self.mouse}_bytrial.csv')
            return stim_df
        else:
            print('Trials not equal to Dataframe Length, dumbass')
            print(len(trial_list))
            print(len(stim_df))

    def get_stim_df(self):
        run, stim = self.get_runs()
        stim_dict = self.add_stims_to_run(run, stim)
        stim_df = self.parse_stim_csvs(stim_dict)
        stim_df = self.verify_trials(stim_df, stim_dict)
        print('Successfully created stim_df)')
        return stim_df

class Raw:
    def __init__(self, Analysis):
        self.mouse = Analysis.mouse
        self.date = Analysis.date
        self.path = Analysis.path
        if Analysis.processed:
            self.units = Analysis.units

        
    def get_raw(self, probe, band = 'ap'):
    # takes in an open ephys recording object and a probe name and returns the continuous binary object
        recording = Analysis.OE(self.path)
        if band == 'ap': 
            if str(probe) == 'probeA':
                data = recording.continuous[1]
            elif str(probe) == 'probeB':
                data = recording.continuous[3]
            else:
                data = recording.continuous[5]
            return data

        elif band == 'lfp':
            if str(probe) == 'probeA':
                data = recording.continuous[1]
            elif str(probe) == 'probeB':
                data = recording.continuous[3]
            else:
                data = recording.continuous[5]
            return data
        else: 
            print('You got not bands. Get your paper up.')
            return None
        
    def get_chunk(self, probe, # can use binary_path output directly
              stim_times,
              band = 'ap',
              pre = 100, # time in ms
              post = 500, # time in ms
              chs = np.arange(0,200,1), # channels
              output = 'response' # 'response, 'pre/post', 'all'
              ):
        """
        Takes in a continuous binary object and a list of stimulation times and returns a chunk of the data
        """
        data = self.get_raw(probe = probe, band = band)
        sample_rate = data.metadata['sample_rate']
        pre_samps = int((pre/1000 * sample_rate))
        post_samps = int((post/1000 * sample_rate))
        total_samps = pre_samps + post_samps

        n_chs = len(chs)
        if output == 'response':
            response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))   
                chunk = data.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                    selected_channels = chs)
                chunk = chunk - np.median(chunk, axis = 0 )
                response[i,:,:] = chunk

            return response
        
        elif output == 'pre/post':
            pre_response = np.zeros((np.shape(stim_times)[0],pre_samps, n_chs))
            post_response = np.zeros((np.shape(stim_times)[0],post_samps, n_chs))
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))   

                pre_chunk = data.get_samples(start_sample_index = start_index, end_sample_index = stim, 
                                    selected_channels = np.arange(0,n_chs,1))
                post_chunk = data.get_samples(start_sample_index = stim, end_sample_index = end_index, 
                                    selected_channels = np.arange(0,n_chs,1))
                pre_chunk = pre_chunk - np.median(pre_chunk, axis = 0 )
                post_chunk = post_chunk - np.median(post_chunk, axis = 0 )
                pre_response[i,:,:] = pre_chunk
                post_response[i,:,:] = post_chunk
            return pre_response, post_response
        
        elif output == 'all':
            response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
            pre_response = np.zeros((np.shape(stim_times)[0],pre_samps, n_chs))
            post_response = np.zeros((np.shape(stim_times)[0],post_samps, n_chs))
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            
            for i, stim in enumerate(stim_indices):
                
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))   

                pre_chunk = data.get_samples(start_sample_index = start_index, end_sample_index = stim, 
                                    selected_channels = np.arange(0,n_chs,1))
                post_chunk = data.get_samples(start_sample_index = stim, end_sample_index = end_index, 
                                    selected_channels = np.arange(0,n_chs,1))
                
                chunk = data.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                    selected_channels = chs)
                pre_chunk = pre_chunk - np.median(pre_chunk, axis = 0 )
                post_chunk = post_chunk - np.median(post_chunk, axis = 0 )
                chunk = chunk - np.median(chunk, axis = 0 ) 
                pre_response[i,:,:] = pre_chunk
                post_response[i,:,:] = post_chunk
                response[i,:,:] = chunk
            return pre_response, post_response, response
        
    def plot_ap(self, probe, stim_times, 
                pre = 4, post = 20, 
                first_ch = 125, last_ch = 175, 
                title = '', 
                spike_overlay = False,
                n_trials = 10, spacing_mult = 350, 
                save = False, savepath = '', format ='png'):
        
        data = self.get_raw(probe = probe,band = 'ap')
        response = self.get_chunk(probe = probe, stim_times = stim_times, 
                                  pre = pre, post = post, 
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
        
        trial_subset = np.linspace(0,len(stim_times)-1, n_trials) #choose random subset of trials to plot 
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

#class NWB: 
class NWB_Tools:        
    
        def __init__(self, Analysis):
            self.analysis = Analysis
            self.mouse = self.analysis.mouse
            if self.analysis.processed:
                self.trials = self.analysis.trials
                self.units = self.analysis.units
            else: 
                self.stim_df = self.analysis.stim_df
            self.path = self.analysis.path


            
        
        def make_unitsdf(self, probes = ['probeA','probeB','probeC'], depths =  [2000, 2000, 2000], herbs = False, stimuli = None):
            """_summary_

            Args:
                probes (list, optional): _description_. Defaults to ['probeA','probeB','probeC'].
                depths (list, optional): _description_. Defaults to [2000, 2000, 2000].
                herbs (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            dfs = []
            for i, probe in enumerate(probes):
                
                probe_path = os.path.join(self.path, 'Record Node 105','Experiment1','recording1',
                            'continuous',f'Neuropix-PXI-104.{probe.capitilize()}-AP' )
                make_spike_secs(probe_path)
                df = load_unit_data(probe_path, probe_depth = depths[i], 
                                            spikes_filename = 'spike_secs.npy',
                                            probe_name = probe.strip('Probe'))
                cluster_info = pd.read_csv(os.path.join(probe_path, 'cluster_info.tsv'),sep ='\t')
                ch = np.array(cluster_info.ch)
                depth = np.array(cluster_info.depth)

                df['ch'] = ch
                df['depth'] = depth
                
                dfs.append(df)
            
            units_df = pd.concat(dfs)
            
            print('Line 440: units_df concatenated')
            
            #generate unique unit IDs
            new_unit_ids = []
            for unitid in units_df.index:
                uu = units_df.iloc[[unitid]]
                new_unit_ids.append("{}{}".format(str(list(uu["probe"])[0]), str(list(uu["unit_id"])[0])))
            units_df["unit_id"] = new_unit_ids

            print('Unique Unit IDs generated')

            if herbs:
                from ccf_3D.tools import herbs_processing as hp
                from ccf_3D.tools.metrics import distance
                stim_df = stimuli
                probe_list, stim = hp.load_herbs(self.mouse)
                probe_IDs = [probe.strip('probe') for probe in probes]
                probes_dist = {}
                stim_coords_ = hp.stim_coords(stim)
                most_used_contact = self.stim_df['contact_negative'].value_counts().idxmax()
                
                for ID, probe in zip(probe_IDs, probe_list):
                    vox = hp.neuropixel_coords(probe)
                    dist = distance(vox, stim_coords_[most_used_contact])
                    probes_dist[ID] = dist
                    mask = units_df['probe'] == ID
                    units_df.loc[mask, 'distance_from_stim'] = units_df.loc[mask, 'ch'].apply(lambda x: dist[x-1] if x-1 < len(dist) else None)
                    channel_structures_full,channel_structures_acronym = hp.get_channel_structures_herbs(probe)
                    units_df.loc[mask, 'channel_structures_full'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_full[x-1] if x-1 < len(channel_structures_full) else None)
                    units_df.loc[mask, 'channel_structures_acronym'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_acronym[x-1] if x-1 < len(channel_structures_acronym) else None)
            
            return units_df
        
        def prep_stim_df(self):   
            stimuli = self.stim_df
            stimuli['notes'] = ''
            stimuli['start_time'] = stimuli['stim_time']
            stimuli['end_time'] = stimuli['stim_time']+2
            return stimuli
        
        def assemble(self, subject, experimenter = 'jlh',
                    experiment_description = 'Electrical Brain Stimulation',
                    device_name = 'DenmanLab_EphysRig2',
                    lab = 'Denman Lab',
                    institution = 'University of Colorado',
                    keywords = ['neuropixels','mouse','electrical stimulation', 'orthogonal neuropixels']):
        
            nwbfile = NWBFile(self.mouse, 
                self.path, 
                datetime.now(tzlocal()),
                experimenter = experimenter,
                lab = lab,
                keywords = keywords,
                institution = institution,
                subject = subject,
                experiment_description = experiment_description,
                session_id = os.path.basename(self.path))
            return nwbfile
            
        def add_stim_epochs(self, nwbfile, stimuli):
            #add epochs
            nwbfile.add_epoch(stimuli.start_time.values[0],
                            stimuli.start_time.values[-1], 'stimulation_epoch')
            nwbfile.add_trial_column('train_duration', 'train duration (s)')
            nwbfile.add_trial_column('train_period', 'train period (s)')
            nwbfile.add_trial_column('train_quantity', 'train quantity')
            nwbfile.add_trial_column('shape', 'monophasic, biphasic or triphasic')
            nwbfile.add_trial_column('run', 'the run number')
            nwbfile.add_trial_column('pulse_duration', 'usecs')
            nwbfile.add_trial_column('pulse_number', 'event quantity')
            nwbfile.add_trial_column('event_period', 'milliseconds')
            nwbfile.add_trial_column('amplitude', 'amplitude in uA')
            nwbfile.add_trial_column('contacts', 'the stimulation contacts and polarities used on the stim electrode')
            nwbfile.add_trial_column('contact_negative', 'the negative (cathodal) contact for a trial')
            nwbfile.add_trial_column('contact_positive', 'the positive (anodal) contact used') 
            nwbfile.add_trial_column('polarity', 'bipolar or monopolar')
            nwbfile.add_trial_column('notes', 'general notes from recording')


            for i in range(len(stimuli)):    
                nwbfile.add_trial(start_time = stimuli.start_time[i],
                    stop_time = stimuli.end_time[i],
                    #parameter = str(stimuli.parameter[i]),
                    amplitude = stimuli.EventAmp1[i],
                    pulse_duration = stimuli.EventDur1[i],
                    shape = stimuli.EventType[i],
                    polarity = str(stimuli.polarity[i]),
                    run = stimuli.Run[i],
                    pulse_number = stimuli.EventQuantity[i],
                    event_period = stimuli.EventPeriod[i]/1e3,
                    train_duration = stimuli.TrainDur[i]/1e6,
                    train_period = stimuli.TrainPeriod[i]/1e6,
                    train_quantity = stimuli.TrainQuantity[i],
                    contacts = stimuli.comment[i],
                    contact_positive = stimuli.contact_positive[i],
                    contact_negative = stimuli.contact_negative[i],
                    notes = stimuli.notes[i])
                
        def add_electrodes(self, nwbfile, units_df, probes, device_name):
            probe_IDs = [probe.strip('Probe') for probe in probes]

            device = nwbfile.create_device(name = device_name)

            for i, probe in enumerate(probes):
                electrode_name = 'probe'+str(i)
                description = "Neuropixels1.0_"+probe_IDs[i]
                location = "near visual cortex"

                electrode_group = nwbfile.create_electrode_group(electrode_name,
                                                                description=description,
                                                                location=location,
                                                                device=device)
                
                #add channels to each probe
                for ch in range(option234_positions.shape[0]):
                    nwbfile.add_electrode(x=option234_positions[ch,0],y=0.,
                                        z=option234_positions[0,1],
                                        imp=0.0,location='none',
                                        filtering='high pass 300Hz',
                                        group=electrode_group)
            nwbfile.add_unit_column('probe', 'probe ID')
            nwbfile.add_unit_column('unit_id','cluster ID from KS2')
            nwbfile.add_unit_column('group', 'user label of good/mua')
            nwbfile.add_unit_column('depth', 'the depth of this unit from zpos and insertion depth')
            nwbfile.add_unit_column('xpos', 'the x position on probe')
            nwbfile.add_unit_column('zpos', 'the z position on probe')
            nwbfile.add_unit_column('no_spikes', 'total number of spikes across recording')
            nwbfile.add_unit_column('KSlabel', 'Kilosort label')
            nwbfile.add_unit_column('KSamplitude', 'Kilosort amplitude')
            nwbfile.add_unit_column('KScontamination', 'Kilosort ISI contamination')
            nwbfile.add_unit_column('template', 'Kilosort template')
            nwbfile.add_unit_column('ch', 'channel number')


            for i,unit_row in units_df[units_df.group != 'noise'].iterrows():
                nwbfile.add_unit(probe=str(unit_row.probe),
                                id = i,
                                unit_id = unit_row.unit_id,
                                spike_times=unit_row.times,
                                electrodes = np.where(unit_row.waveform_weights > 0)[0],
                                depth = unit_row.depth,
                                xpos= unit_row.xpos,
                                zpos= unit_row.zpos,
                                template= unit_row.template,
                                no_spikes = unit_row.no_spikes,
                                group= str(unit_row.group),
                                KSlabel= str(unit_row.KSlabel),
                                KSamplitude= unit_row.KSamplitude,
                                KScontamination= unit_row.KScontamination,
                                ch = unit_row.ch)
        
        def write_nwb(self, nwbfile):
            with NWBHDF5IO(os.path.join(self.path,f'{self.mouse}.nwb'), 'w') as io:
                io.write(nwbfile)     

        def make_nwb(self, subject, probes = ['probeA','probeB','probeC'], 
                    depths =  [2000, 2000, 2000], 
                    herbs = False,
                    experimenter = 'jlh',
                    experiment_description = 'Electrical Brain Stimulation',
                    device_name = 'DenmanLab_EphysRig2',
                    lab = 'Denman Lab',
                    institution = 'University of Colorado',
                    keywords = ['neuropixels','mouse','electrical stimulation', 'orthogonal neuropixels']):
            
            stimuli = self.prep_stim_df()
            print('Stimuli_df prepped')
            
            print('Beginning: loading units_DF')
            units_df = self.make_unitsdf(probes = probes, depths = depths, herbs = herbs, stimuli= stimuli)
            print('Units_df created')

            nwbfile = self.assemble(subject, experimenter = experimenter,
                    experiment_description =  experiment_description,
                    device_name = device_name,
                    lab = lab,
                    institution = institution,
                    keywords = keywords)
            print('NWBFile Assembled')
            self.add_stim_epochs(nwbfile, stimuli)
            print('Stim Epochs added')
            self.add_electrodes(nwbfile, units_df, probes, device_name)
            print('Electrodes added')
            self.write_nwb(nwbfile)
            print('NWB Written')

        def remake_nwb(self, herbs = True):
            # load nwb
            
            # parse nwb file
            
            # get all relevant information
            
            # remake nwb file
            
            pass
        
        def edit_nwb(self, herbs = True, probes = ['probeA','probeB','probeC'],_return = False):
            nwb_path = glob.glob(os.path.join(self.path,'*.nwb'))[0]
            io = NWBHDF5IO(nwb_path, 'r+')
            nwb = io.read()
            units_df = self.units
            stim_df = self.trials

            
            if herbs == True:
                from ccf_3D.tools import herbs_processing as hp
                from ccf_3D.tools.metrics import distance
                probe_list, stim = hp.load_herbs(self.mouse)
                probe_IDs = [probe.strip('probe') for probe in probes]
                probes_dist = {}
                stim_coords_ = hp.stim_coords(stim)
                most_used_contact = stim_df['contact_negative'].value_counts().idxmax()
                
                for ID, probe in zip(probe_IDs, probe_list):
                    vox = hp.neuropixel_coords(probe)
                    dist = distance(vox, stim_coords_[most_used_contact])
                    probes_dist[ID] = dist
                    mask = units_df['probe'] == ID
                    units_df.loc[mask, 'herbs_distance_from_stim'] = units_df.loc[mask, 'ch'].apply(lambda x: dist[x-1] if x-1 < len(dist) else None)
                    channel_structures_full,channel_structures_acronym = hp.get_channel_structures_herbs(probe)
                    units_df.loc[mask, 'channel_structures_full'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_full[x-1] if x-1 < len(channel_structures_full) else None)
                    units_df.loc[mask, 'channel_structures_acronym'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_acronym[x-1] if x-1 < len(channel_structures_acronym) else None)
                    

                brain_reg_full = list(units_df['channel_structures_full'])
                brain_reg_acronym = list(units_df['channel_structures_acronym'])  
                distance_from_stim = list(units_df['herbs_distance_from_stim'])
                #add columns to nwb
                nwb.units.add_column(name='brain_reg_full', 
                                    description='brain regions by ch from HERBs data', 
                                    data = brain_reg_full)
                nwb.units.add_column(name='brain_reg_acronym', 
                                    description='brain region acronyms by ch from HERBs data', 
                                    data = brain_reg_acronym)
                nwb.units.add_column(name='herbs_distance_from_stim', 
                                    description='distance from a ch to the most used contact of the stimulating electrode', 
                                    data = distance_from_stim)
                
                io.write(nwb)

            if _return == True:
                return nwb #as a writeable object   

class Analysis:
    def __init__(self, mouse, date):
        self.mouse = mouse
        self.date = date
        
        self.loader = Loader(self.mouse, self.date)
        self.path = self.loader.path
        self.processed = self.loader.processed
        
        if self.processed:
            self.trials = self.loader.trials
            self.units = self.loader.units
            self.parameters = self.loader.parameters
            self.get_probes()
        else: 
            self.preprocess = Preprocess(self)
            self.stim_df = self.preprocess.stim_df
        
        try:
            self.get_contacts()
        except:
            "Could not get_contacts, check trials dataframe"
        self.raw = Raw(self)
        self.nwb_tools = NWB_Tools(self)
        self.plotter = Plotter(self)




    @staticmethod    
    def OE(path):
        session = Session(path)
        recording = session.recordnodes[0].recordings[0]
        return recording
    
    def get_probes(self):
        # Extract unique probe names
        probe_IDs = self.units.probe.unique()
        # Format them as "probeA", "probeB", etc.
        probe_names = [f"probe{probe_ID}" for probe_ID in probe_IDs]
        self.probes = probe_names

    def get_contacts(self):
        if self.processed == False:
            df = self.stim_df
            strings = df['comment']
            
            contact_negative = []
            contact_positive = []
            polarity = []
            
            for string in strings:
                r_number = re.search(r'(\d+)r', string)
                r_value = int(r_number.group(1)) if r_number else None
                contact_negative.append(r_value)

                b_number = re.search(r'(\d+)b', string)
                b_value = int(b_number.group(1)) if b_number else 0
                contact_positive.append(b_value)

                if b_value == 0:
                    polarity.append("monopolar")
                else:
                    polarity.append("bipolar")
            
            df['contact_negative'] = contact_negative
            df['contact_positive'] = contact_positive
            df['polarity'] = polarity
            self.stim_df = df
        
        elif 'contact_negative' not in self.trials.columns:
            df = self.trials
            strings = df.contacts
            contact_negative = []
            contact_positive = []
            polarity = []
            
            for string in strings:
                r_number = re.search(r'(\d+)r', string)
                r_value = int(r_number.group(1)) if r_number else None
                contact_negative.append(r_value)

                b_number = re.search(r'(\d+)b', string)
                b_value = int(b_number.group(1)) if b_number else 0
                contact_positive.append(b_value)

                if b_value == 0:
                    polarity.append("monopolar")
                else:
                    polarity.append("bipolar")
            df['contact_negative'] = contact_negative
            df['contact_positive'] = contact_positive
            df['polarity'] = polarity
            self.trials = df
            print('Contacts added to trials dataframe')
        else:
            pass
    
    def get_zetas(self):
        # get zetascore with no window selected
        if os.path.exists(os.path.join(self.path, 'zetascores.pkl')):
            with open(os.path.join(self.path,'zetascores.pkl'), 'rb') as f:
                self.zetascore = pkl.load(f)
        else:
            import zetapy as zp
            zetascore = {}
            spiketimes = np.array(self.units.spike_times)
            for run in tqdm(self.trials.run.unique()):
                stim_time = np.array(self.trials.loc[self.trials.run==run].start_time)
                zetascore[run] = []
                for unit in range(len(spiketimes)):
                    unit_score = zp.getZeta(spiketimes[unit], stim_time)
                    zetascore[run].append(unit_score) 
            with open(os.path.join(self.path,'zetascores.pkl'), 'wb') as f:
                pkl.dump(zetascore, f)
            self.zetascore = zetascore
        
        # do it for 20ms window
        if os.path.exists(os.path.join(self.path, 'zetascores20ms.pkl')):
            with open(os.path.join(self.path,'zetascores20ms.pkl'), 'rb') as f:
                self.zetascore20ms = pkl.load(f)
        else:
            zetascore20ms = {}
            spiketimes = np.array(self.units.spike_times)
            for run in tqdm(self.trials.run.unique()):
                stim_time = np.array(self.trials.loc[self.trials.run == run].start_time)
                zetascore20ms[run] = []
                for i in range(len(spiketimes)):
                    unit_score = zp.getZeta(spiketimes[i],stim_time, dblUseMaxDur = 0.02)
                    zetascore20ms[run].append(unit_score)
            with open(os.path.join(self.path,'zetascores20ms.pkl'), 'wb') as f:
                pkl.dump(zetascore20ms, f)
            self.zetascore20ms = zetascore20ms
        
        # do it again for 300ms window
        if os.path.exists(os.path.join(self.path, 'zetascores300ms.pkl')):
            with open(os.path.join(self.path,'zetascores300ms.pkl'), 'rb') as f:
                self.zetascore300ms = pkl.load(f)
        else:
            zetascore300ms = {}
            spiketimes = np.array(self.units.spike_times)
            for run in tqdm(self.trials.run.unique()):
                stim_time = np.array(self.trials.loc[self.trials.run == run].start_time)
                zetascore300ms[run] = []
                for i in range(len(spiketimes)):
                    unit_score = zp.getZeta(spiketimes[i],stim_time, dblUseMaxDur = 0.3)
                    zetascore300ms[run].append(unit_score)
            with open(os.path.join(self.path,'zetascores300ms.pkl'), 'wb') as f:
                pkl.dump(zetascore300ms, f)
            self.zetascore300ms = zetascore300ms

    def assign_zeta_sig(self, zetas = None, plot = True):
        if zetas is None:
            zetas = self.zetascore300ms
        for run in self.trials.run.unique():
            p = []
            for unit in range(len(self.units.index)):
                p.append(zetas[run][unit][0])
            
            sig = ['sig' if z < 0.05 else 'non-sig' for z in p]
            self.units[f'r{run}'] = sig
        
        if plot:
            for run in self.trials.run.unique():
                p = []
                for unit in range(len(self.units.index)):
                    p.append(zetas[run][unit][0])
                num_sig = []
                for z in p:
                    if z < 0.05:
                        num_sig.append(z)
                print(f'Run: {run}, {self.parameters[run]}: {(len(num_sig)/len(p))*100}')
                perc_sig = (len(num_sig)/len(p))*100
                plt.bar(run, perc_sig)
            plt.title('Zetascore') 
            plt.xlabel('Run')
            plt.ylabel('Percent of Cells activated')
            tt = plt.xticks(range(len(self.parameters)))
    
    def get_electrode_coords(self):
        from ccf_3D.tools import herbs_processing as hp
        probes, stim = hp.load_herbs(mouse = self.mouse, probe_names = self.probes)
        electrode_coords = []
        for probe in probes:
            electrode_coords.append(hp.neuropixel_coords(probe))
        electrode_coords = np.array(electrode_coords)
        
        return electrode_coords

    def get_brain_regs(self):
        
        probes, stim = hp.load_herbs(mouse = self.mouse, probe_names = self.probes)
        brain_regs = []
        for probe in probes: 
            full, _ = hp.get_channel_structures_herbs(probe)
            brain_regs.append(full)
        brainreg_A = [brain_regs[0]]
        brainreg_B = [brain_regs[1]]
        brainreg_C = [brain_regs[2]]
        brainreg_A = brainreg_A[0]
        brainreg_B = brainreg_B[0]
        brainreg_C = brainreg_C[0]

        return brainreg_A, brainreg_B, brainreg_C

        



    


class Plotter():
    def __init__(self, Analysis):
        self.analysis = Analysis
        if self.analysis.processed == True:
            self.trials = Analysis.trials
            self.units = Analysis.units
            self.probes = Analysis.probes
            self.parameters = Analysis.parameters
            self.path = Analysis.path
            self.mouse = Analysis.mouse
            self.date = Analysis.date
            


    
    def summary_plot(self, save = True):
        from matplotlib.gridspec import GridSpec
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        import seaborn as sns

        '''
        First Row: Text details
        Second Row: Unit Quality Histograms and DataCube
        Final 2 Rows: Rasters for each unique run
        TODO: add LFP gamma power plot
        '''
        plt.style.use('dark_background')
        # Main grid layout with 4 rows
        main_gs = GridSpec(4, 1, height_ratios=[0.5, 3, 5, 5])

        # Create the main figure
        fig = plt.figure(figsize=(20, 30))

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
        gs_units_quality = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_upper_row[0, 0], wspace = 0.3, hspace = 0.7)
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
            row = index // 2
            col = index % 2
            ax_unit_quality = plt.subplot(gs_units_quality[row, col])
            subset = units_reset_index[units_reset_index['probe'] == probe]
            subset = units_reset_index[units_reset_index['group'] != 'nan']
            sns.histplot(data=subset, x='no_spikes', hue='group', bins=bins, element='bars', common_norm=False, palette='viridis', alpha=0.6, ax=ax_unit_quality)
            ax_unit_quality.set_title(f'Distribution of Spike Counts for Probe {probe}', fontsize = 13)
            ax_unit_quality.set_xlabel('Spike Counts', fontsize = 13)
            ax_unit_quality.set_ylabel('Number of Units', fontsize = 13)
            ax_unit_quality.set_ylim(0, y_max)
            ax_unit_quality.tick_params(labelsize=8, rotation =45)

        # Find the appropriate axes for the last subplot
        ax_last = plt.subplot(gs_units_quality[1, 1])
        # Plot the bar plot for unit group counts in the last subplot
        sns.countplot(data=self.units[self.units['group'] != 'nan'], x='probe', hue='group', palette='viridis', ax=ax_last)
        ax_last.set_title('Unit Group Counts by Probe', fontsize = 13)
        ax_last.set_xlabel('Probe', fontsize = 13)
        ax_last.set_ylabel('Count', fontsize = 13)
        ax_last.tick_params(labelsize=12, rotation =45)
        ax_last.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        


        ## trials 
        # Iterate through the polarities and plot the heatmaps

        for index, polarity in enumerate(polarities):
            ax2 = plt.subplot(gs_trials_data_cube[index, 0])
            subset = trials_with_existence[trials_with_existence['polarity'] == polarity]
            pivot_table = subset.pivot_table(values='exists', index='pulse_duration', columns='amplitude', fill_value=0)
            x_labels = [f"{x} ua" for x in pivot_table.columns]
            y_labels = [f"{y} us" for y in pivot_table.index]
            sns.heatmap(pivot_table, cmap=color_schemes[index], annot=False, cbar=False, linewidths=0.5, alpha=0.8, square=False, ax=ax2)
            ax2.set_title(f'Heatmap for {polarity} Polarity', fontsize=15)
            ax2.set_xlabel('Amplitude', fontsize=14)
            ax2.set_ylabel('Pulse Duration', fontsize=14)
            ax2.set_xticklabels(x_labels, fontsize=13, rotation=45) # Rotate x-axis tick labels
            ax2.set_yticklabels(y_labels, fontsize=13, rotation=45) # Rotate y-axis tick labels


        #################################################
        # Grid specification for the raster plot (2/3 height, divided into two columns)
        gs_raster = GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[2:])

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

    
            
    def units_distance_heatmap(self, brainreg, pre, post, probeID, run, dists, vmax, save=True):
        cmap = sns.color_palette("crest", as_cmap=True).reversed()
        
        _units = self.units[self.units['probe'] == probeID].sort_values('ch', ascending=False)
        stimtimes = np.array(self.trials[self.trials.run == run].start_time)

        psths_ch = []
        distance_ticks = []
        un_count = 0
        unit_ind = []
        channel_ticks = []
        
        for ch in range(len(brainreg) - 1, -1, -1):
            matching_units = _units[_units.ch == ch]
            num_units_ch = len(matching_units)
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
        print(len(distance_ticks))
        print(len(unit_ind))
        print(len(psths_ch))
        fig, ax = plt.subplots()
        im = ax.imshow(psths_ch, vmax=vmax, aspect='auto', cmap=cmap, extent=[pre_time_ms, post_time_ms, 0, len(psths_ch)])
        ax.set_yticks(unit_ind[::-40]) 
        ax.set_yticklabels(distance_ticks[::-40]) # Reverse the tick labels
        ax.set_ylabel('Distance from Stimulation')
        ax.set_xlabel('Time (ms)')
        ax.axvline(x=0, color='r', linestyle='--')  # Vertical line at time zero for stimulation

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Firing Rate')
        if save:
            plt.savefig(os.path.join(self.path, f'Probe{probeID}dist_heatmap.eps'), dpi = 600)

        return ax, psths_ch, channel_ticks[::-1], distance_ticks[::-1], unit_ind[::-1] 
        ## channel and distance ticks orders are switched so that they start with ch0... easier for indexing into brainregs
        ##Todo: distance calculations are a little wonky


    def calculate_normalized_firing_rates(self,psths_ch):
        psths_ch = np.array(psths_ch)
        baseline_window = range(0, 240) # -500 to -2ms
        early_activation_window = range(251, 257) # 0-12ms
        suppression_window = range(261, 326) # 20-150ms
        late_activation_window = range(351, 451) # 200-400ms

        baseline_rate = psths_ch[:, baseline_window].sum(axis=1) / len(baseline_window)
        early_activation_rate = psths_ch[:, early_activation_window].sum(axis=1) / len(early_activation_window)
        suppression_rate = psths_ch[:, suppression_window].sum(axis=1) / len(suppression_window)
        late_activation_rate = psths_ch[:, late_activation_window].sum(axis=1) / len(late_activation_window)

        return {
            'early_activation': early_activation_rate - baseline_rate,
            'suppression': suppression_rate - baseline_rate,
            'late_activation': late_activation_rate - baseline_rate,
        }

    

    def plot_normalized_firing_rates_with_regions(self,normalized_rates, channel_ticks, distance_ticks, unit_ind, brainreg, smoothing_sigma=5, save = True, probeID = 'A'):
        from scipy.ndimage.filters import gaussian_filter1d
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        windows = ['early_activation', 'suppression', 'late_activation']
        titles = [win.replace('_', ' ').title() for win in windows]
        brainreg = brainreg[::-1]
        
        # Getting unique regions and their middle ticks
        
        
        regions = []


        for ch in channel_ticks:
            regions.append(brainreg[ch])
           
        
        unique_regions = list(set(regions))

        reg_label = []

        unique_ticks = []
        for region in unique_regions:
            indices = [i for i, x in enumerate(regions) if x == region]
            unique_ticks.append(indices[0])
            unique_ticks.append(indices[-1])
            reg_label.append(region)
            reg_label.append(region)

        for idx, win in enumerate(windows):
            # Apply Gaussian smoothing to the firing rates
            smoothed_rates = gaussian_filter1d(normalized_rates[win], smoothing_sigma)
            
            axs[idx].plot(smoothed_rates, range(len(smoothed_rates)))
            axs[idx].set_title(titles[idx])
            axs[idx].set_xlabel('Normalized Firing Rate')
            axs[idx].invert_yaxis()

        axs[0].set_ylabel('Distance from Stimulation')
        axs[0].set_yticks(unit_ind[::40])
        axs[0].set_yticklabels(distance_ticks[::40])

        # Adding a secondary axis for the brain regions
        ax2 = axs[2].twinx()
        ax2.set_ylim(axs[2].get_ylim())
        ax2.set_yticks(unique_ticks)
        ax2.set_yticklabels(reg_label, fontsize=8)

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.path, f'Probe{probeID}_normalized_rates.eps'), dpi = 600)
        return fig, axs

    def plot_heatmap_normFR(self, probeID, brainreg, dists, run, pre=0.5, post=0.5, vmax=20, save=True):
        # Generate the initial heatmap and extract relevant info
        ax, psths_ch, channel_ticks, distance_ticks, unit_ind = self.units_distance_heatmap(brainreg=brainreg, 
                                                                                        pre=pre, 
                                                                                        post=post, 
                                                                                        probeID=probeID, 
                                                                                        run=run, 
                                                                                        dists=dists, 
                                                                                        vmax=vmax, 
                                                                                        save=False)
        
        # Calculate normalized firing rates based on the initial heatmap
        
        normalized_rates = self.calculate_normalized_firing_rates(psths_ch)
        
        # Plot normalized firing rates, adding labels for brain regions
        fig, axs = self.plot_normalized_firing_rates_with_regions(
            normalized_rates, channel_ticks, distance_ticks, unit_ind, brainreg, smoothing_sigma=5, save=True, probeID=probeID)




        
            




                
                
            

                

            











#! general tool and functions for paths and stuff










## general tools


        
## PSTH and Raster like stuff
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



