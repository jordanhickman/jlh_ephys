
#import dlab.nwbtools # this changed significantly so you have to figure out how to use it...

import os,glob
import pandas as pd
import numpy as np
import shutil

from pynwb import NWBHDF5IO
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

from datetime import datetime
from dateutil.tz import tzlocal

import pendulum
import re

import traceback
import logging

from dlab.generalephys import option234_positions

def make_spike_secs(probe_path):
    if os.path.isfile(os.path.join(probe_path, 'spike_seconds.npy')):
        spike_times = np.load(os.path.join(probe_path, 'spike_seconds.npy'))
        
    else:
        spike_samps = np.load(os.path.join(probe_path,'spike_times.npy')).flatten()        
        ts = np.load(os.path.join(probe_path,'timestamps.npy'))
        spike_times = ts[spike_samps]
        np.save(os.path.join(probe_path,'spike_seconds.npy'), spike_times)
    return spike_times

def load_unit_data(probe_path, probe_name, probe_depth):
    from random import sample
    spike_times = make_spike_secs(probe_path)
    
    site_positions  = np.load(os.path.join(probe_path,'channel_positions.npy'))               
    spike_clusters  = np.load(os.path.join(probe_path,'spike_clusters.npy')).flatten()
    spike_templates = np.load(os.path.join(probe_path,'spike_templates.npy'))
    templates       = np.load(os.path.join(probe_path,'templates.npy'))
    amplitudes      = np.load(os.path.join(probe_path,'amplitudes.npy'))
    cluster_info    = pd.read_csv(os.path.join(probe_path,'cluster_info.tsv'), delimiter='\t')
    
    
    weights        = np.zeros(site_positions.shape)
    
    mean_templates = []
    peak_templates = []

    all_weights    = []
    amps           = []
    times          = []
    ch             = []
    
    for unit_id in np.unique(spike_clusters):
        #get mean template for each unit
        all_templates, count    = np.unique(spike_templates[np.where(spike_clusters==unit_id)],return_counts=True)
        
        if len(all_templates) > 100:
            n_templates_to_subsample = 100
        else: 
            n_templates_to_subsample = len(all_templates)
        
        random_subsample_of_templates = templates[sample(list(all_templates),n_templates_to_subsample)]
        
        mean_template = np.mean(random_subsample_of_templates,axis=0)
        
        mean_templates.append(mean_template)
        
        if cluster_info is not None:
            best_ch = cluster_info[cluster_info.cluster_id == unit_id].ch.values[0].astype(int)
        else:
            best_ch = np.argmax((np.max(mean_template,axis=0) - np.min(mean_template,axis=0)))
        
        ch.append(best_ch)
        
        peak_wv = mean_template[:,best_ch-1]
        peak_templates.append(peak_wv)
        
        #Take weighted average of site positions where hweights is abs value of template for that channel
        #This gets us the x and y positions of the unit on the probe

        # for channel in range(len(mean_template.T)):
        #     weights[channel,:] = np.trapz(np.abs(mean_template.T[channel]))
    
        # # weights                /= weights.max()
        # weights[weights < 0.25] = 0 #Where weights are low, set to 0
        # x,y                     = np.average(site_positions,weights=weights,axis=0)
        # all_weights.append(weights)
        # xpos.append(x)
        # ypos.append(y)
        
        amps.append(amplitudes[:,0][spike_clusters==unit_id])
        times.append(spike_times[spike_clusters==unit_id])

    if cluster_info is not None:
        probe_data = cluster_info.copy()
        
    probe_data['probe']      = probe_name
    # probe_data['shank']    = np.floor(cluster_info['xcoords'].values / 205.).astype(int)
    probe_data['depth']      = np.array(site_positions[:,1][ch])*-1 + probe_depth
    probe_data['times']      = times
    probe_data['amplitudes'] = amps
    probe_data['template']   = mean_templates
    # probe_data['weights']  = all_weights
    probe_data['peak_wv']    = peak_templates
    probe_data['xpos']       = site_positions[:,0][probe_data['ch']]
    probe_data['ypos']       = site_positions[:,1][probe_data['ch']]
    probe_data['no_spikes']   = [len(i) for i in times]
    probe_data['KSamplitude'] = probe_data['Amplitude']
    probe_data['unit_id'] = probe_data['cluster_id']
    return probe_data

class NWB_Tools:        
        def __init__(self, analysis_obj):
            self.analysis = analysis_obj
            self.mouse = self.analysis.mouse
            if self.analysis.processed:
                self.trials = self.analysis.trials
                self.units = self.analysis.units
            else: 
                self.stim_df = self.analysis.stim_df
            self.path = self.analysis.path
        
        
        def pad_template(self, template):
            target_shape = (82, 384)
            padding_shape = tuple(target - current for target, current in zip(target_shape, template.shape))
            return np.pad(template, ((0, padding_shape[0]), (0, padding_shape[1])), 'constant')

        def check_and_pad_templates(self):
            unique_shapes = self.units_df['template'].apply(np.shape).unique()
            
            if len(unique_shapes) > 1:
                print(f"Multiple unique shapes found in 'template': {unique_shapes}. Padding to uniform shape.")
                self.units_df['template'] = self.units_df['template'].apply(self.pad_template)
            else:
                print(f"Only one unique shape found in 'template': {unique_shapes[0]}. No padding needed.")
        
        def make_unitsdf(self, probes = ['probeA','probeB','probeC'], depths =  [2000, 2000, 2000], herbs = False):
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
                probe_path = os.path.join(self.path, 'Record Node 105','experiment1','recording1',
                            'continuous',f'Neuropix-PXI-104.Probe{probe.strip("probe")}-AP' ) # ugh this can change...
                if os.path.exists(probe_path):
                    print('Appears to be recorded on OpenEphys')
                    df = load_unit_data(probe_path, 
                                        probe_depth = depths[i], 
                                        probe_name = probe.strip('probe'))
                else:
                    print('Appears to be recorded on SpikeGLX')
                    print('This is not yet supported')
                    #probe_path = glob.glob(os.path.join(self.path, f'*imec{i}*'))[0]
                    #df = load_unit_data(probe_path, probe_depth = depths[i],
                    #                   spikes_filename = 'spike_seconds_adjusted.npy',
                    #                   probe_name = probe.strip('probe'))
                
                dfs.append(df)
            
            units_df = pd.concat(dfs)
            
            print('Line 440: units_df concatenated')
            
            # Create new unique 'unit_id' by appending 'probe' and 'unit_id'
            units_df['ID'] = units_df['probe'].astype(str) + units_df['unit_id'].astype(str)

            print('Unique Unit IDs generated')

            if herbs:
                from ccf_3D.tools import herbs_processing as hp
                from ccf_3D.tools.metrics import distance
                stim_df = self.stimuli
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
            
            self.units_df = units_df
            
        
        def prep_stim_df(self):   
            stimuli = self.stim_df
            stimuli['notes'] = ''
            stimuli['start_time'] = stimuli['stim_time']
            stimuli['end_time'] = stimuli['stim_time']+2
            self.stimuli = stimuli
        
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
            self.nwbfile = nwbfile
            
        def add_stim_epochs(self, stimulator = 'AM'):
            #add epochs
            self.nwbfile.add_epoch(self.stimuli.start_time.values[0],
                            self.stimuli.start_time.values[-1], 'stimulation_epoch')
            self.nwbfile.add_trial_column('train_duration', 'train duration (s)')
            self.nwbfile.add_trial_column('train_period', 'train period (s)')
            self.nwbfile.add_trial_column('train_quantity', 'train quantity')
            self.nwbfile.add_trial_column('shape', 'monophasic, biphasic or triphasic')
            self.nwbfile.add_trial_column('run', 'the run number')
            self.nwbfile.add_trial_column('pulse_duration', 'usecs')
            self.nwbfile.add_trial_column('pulse_number', 'event quantity')
            self.nwbfile.add_trial_column('event_period', 'milliseconds')
            self.nwbfile.add_trial_column('amplitude', 'amplitude in uA')
            self.nwbfile.add_trial_column('contacts', 'the stimulation contacts and polarities used on the stim electrode')
            self.nwbfile.add_trial_column('contact_negative', 'the negative (cathodal) contact for a trial')
            self.nwbfile.add_trial_column('contact_positive', 'the positive (anodal) contact used') 
            self.nwbfile.add_trial_column('polarity', 'bipolar or monopolar')
            self.nwbfile.add_trial_column('notes', 'general notes from recording')


            if stimulator == 'AM':
                for _, row in self.stimuli.iterrows():    
                    self.nwbfile.add_trial(
                        start_time=row['start_time'],
                        stop_time=row['end_time'],
                        amplitude=row['EventAmp1'],
                        pulse_duration=row['EventDur1'],
                        shape=row['EventType'],
                        polarity=str(row['polarity']),
                        run=row['Run'],
                        pulse_number=row['EventQuantity'],
                        event_period=row['EventPeriod'] / 1e3,  # Convert to seconds, assuming original unit is milliseconds
                        train_duration=row['TrainDur'] / 1e6,  # Convert to seconds, assuming original unit is microseconds
                        train_period=row['TrainPeriod'] / 1e6,  # Convert to seconds, assuming original unit is microseconds
                        train_quantity=row['TrainQuantity'],
                        contacts=row['comment'],
                        contact_positive=row['contact_positive'],
                        contact_negative=row['contact_negative'],
                        notes=row['notes']
        )

            
            elif stimulator == 'STG5':
                for _, row in self.stimuli.iterrows():    
                    self.nwbfile.add_trial(
                        start_time=row['start_time'],
                        stop_time=row['end_time'],
                        amplitude=row['Amplitude'] / 1000,  # Divide by 1000 due to STG5 code error
                        pulse_duration=row['Pulse Duration'],
                        shape=str(row['Waveform']),
                        polarity=str(row['polarity']),
                        run=row['Run'],
                        pulse_number=1,
                        event_period=-1,  # Single pulses for now, given current limitations
                        train_duration=-1,
                        train_period=row['Time Between Trains'],
                        train_quantity=row['Total Trains'],
                        contacts=str(row['comment']),
                        contact_positive=row['contact_positive'],
                        contact_negative=row['contact_negative'],
                        notes=str(row['notes'])
                    )

                
        def add_electrodes(self, probes, device_name):
            probe_IDs = [probe.strip('Probe') for probe in probes]

            device = self.nwbfile.create_device(name = device_name, description = "Orthogonal neuropixel rig with 3 Neuropixels and electrical stimulator")

            for i, probe in enumerate(probes):
                electrode_name = 'probe'+str(i)
                description = "Neuropixels1.0_"+probe_IDs[i]
                location = "visual cortex"

                electrode_group = self.nwbfile.create_electrode_group(electrode_name,
                                                                description=description,
                                                                location=location,
                                                                device=device)
                
                #add channels to each probe
                for ch in range(option234_positions.shape[0]):
                    self.nwbfile.add_electrode(x=option234_positions[ch,0],y=0.,
                                        z=option234_positions[0,1],
                                        imp=0.0,location='none',
                                        filtering='high pass 300Hz',
                                        group=electrode_group)
            self.nwbfile.add_unit_column('probe', 'probe ID')
            self.nwbfile.add_unit_column('unit_id','cluster ID from KS2')
            self.nwbfile.add_unit_column('group', 'user label of good/mua')
            self.nwbfile.add_unit_column('depth', 'the depth of this unit from zpos and insertion depth')
            self.nwbfile.add_unit_column('xpos', 'the x position on probe')
            self.nwbfile.add_unit_column('zpos', 'the z position on probe')
            self.nwbfile.add_unit_column('no_spikes', 'total number of spikes across recording')
            self.nwbfile.add_unit_column('firing_rate', 'firing rate of unit')
            self.nwbfile.add_unit_column('KSlabel', 'Kilosort label')
            self.nwbfile.add_unit_column('KSamplitude', 'Kilosort amplitude')
            self.nwbfile.add_unit_column('KScontamination', 'Kilosort ISI contamination')
            self.nwbfile.add_unit_column('template', 'Kilosort template')
            self.nwbfile.add_unit_column('ch', 'channel number')


            for i,unit_row in self.units_df.iterrows():
                self.nwbfile.add_unit(probe=str(unit_row.probe),
                                id = i,
                                unit_id = unit_row.unit_id,
                                spike_times=unit_row.times,
                                electrodes = np.where(unit_row.waveform_weights > 0)[0],
                                depth = unit_row.depth,
                                xpos= unit_row.xpos,
                                zpos= unit_row.zpos,
                                template= unit_row.template,
                                no_spikes = unit_row.no_spikes,
                                firing_rate = unit_row.fr,
                                group= str(unit_row.group),
                                KSlabel= str(unit_row.KSlabel),
                                KSamplitude= unit_row.KSamplitude,
                                KScontamination= unit_row.ContamPct,
                                ch = unit_row.ch)
        
        def write_nwb(self):
            with NWBHDF5IO(os.path.join(self.path,f'{self.mouse}.nwb'), 'w') as io:
                io.write(self.nwbfile)
                io.close()

        def backup_nwb_to_synology(self):
            nwb_path = glob.glob(os.path.join(self.path, '*.nwb'))
            if not nwb_path:
                print("No .nwb files to backup.")
                return
            
            synology_path = r"\\denmanlab\s2\nwbs"
            
            for path in nwb_path:
                file_name = os.path.basename(path)
                destination = os.path.join(synology_path, file_name)
                
                if os.path.exists(destination):
                    print(f"{file_name} already backed up.")
                    continue
                    
                try:
                    shutil.copy(path, destination)
                    print(f"Successfully backed up {file_name} to Synology.")
                except Exception as e:
                    print(f"Failed to back up {file_name}: {e}")
        
        
        def make_nwb(self, subject, probes = ['probeA','probeB','probeC'], 
                    depths =  [2000, 2000, 2000], 
                    stimulator = 'AM',
                    herbs = False,
                    experimenter = 'jlh',
                    experiment_description = 'Electrical Brain Stimulation',
                    device_name = 'DenmanLab_EphysRig2',
                    lab = 'Denman Lab',
                    institution = 'University of Colorado',
                    keywords = ['neuropixels','mouse','electrical stimulation', 'orthogonal neuropixels']):
            
            self.prep_stim_df()
            print('Stimuli_df prepped')

            
            print('Beginning: loading units_DF')
            self.make_unitsdf(probes = probes, depths = depths, herbs = herbs)
            print('Units_df created')
            
            self.check_and_pad_templates()

            self.assemble(subject, experimenter = experimenter,
                    experiment_description =  experiment_description,
                    device_name = device_name,
                    lab = lab,
                    institution = institution,
                    keywords = keywords)
            print('NWBFile Assembled')

            self.add_stim_epochs(stimulator = stimulator)
            print('Stim Epochs added')
            
            self.add_electrodes(probes, device_name)
            print('Electrodes added')
            
            try:
                self.write_nwb()
                print('NWB Written')
            except Exception as e:
                logging.error(f"Could not write NWB. Error: {e}")
                traceback.print_exc()

            
            #self.backup_nwb_to_synology()

        def remake_nwb(self, stimuli_df, units_df, subject,
                       spatial = True,
                       experimenter = 'Hickman, Jordan',
                       lab = 'Denman Lab',
                       keywords = ['orthogonal', 'estim', 'electrical stimulation', 'multi-NPs', 'visual cortex'],
                       institution = 'University of Colorado Anschutz Medical Campus',
                       experiment_description = 'Electrical stimulation with orthogonal neuropixels in visual cortex',
                       device_name = 'DenmanLab_EphysRig2'
                       ):
            # before running this, make sure the dataframes are precisely what you'd like them to be

            # assemble
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
            self.nwbfile = nwbfile

            
            # stimuli epochs
            self.nwbfile.add_epoch(stimuli_df.start_time.values[0],
                            stimuli_df.start_time.values[-1], 'stimulation_epoch')
            self.nwbfile.add_trial_column('train_duration', 'train duration (s)')
            self.nwbfile.add_trial_column('train_period', 'train period (s)')
            self.nwbfile.add_trial_column('train_quantity', 'train quantity')
            self.nwbfile.add_trial_column('waveform_shape', 'monophasic, biphasic or triphasic')
            self.nwbfile.add_trial_column('run', 'the run number where each run generally represents a unique parameter space')
            self.nwbfile.add_trial_column('pulse_duration', 'usecs')
            self.nwbfile.add_trial_column('pulse_number', 'event quantity')
            self.nwbfile.add_trial_column('event_period', 'milliseconds')
            self.nwbfile.add_trial_column('amplitude', 'amplitude in uA')
            self.nwbfile.add_trial_column('contacts', 'the stimulation contacts and polarities used on the stim electrode')
            self.nwbfile.add_trial_column('contact_negative', 'the negative (cathodal) contact for a trial')
            self.nwbfile.add_trial_column('contact_positive', 'the positive (anodal) contact used')
            self.nwbfile.add_trial_column('polarity', 'bipolar or monopolar')
            self.nwbfile.add_trial_column('notes', 'general notes from recording')
            if spatial == True:
                self.nwbfile.add_trial_column('contact_negative_coord', 'the ccf coordinate for the cathodal contact')
                self.nwbfile.add_trial_column('contact_positive_coord', 'the ccf coordinate for the anodal contact')


            for i, trial in stimuli_df.iterrows():    
                if spatial == True: 
                    self.nwbfile.add_trial(start_time = trial.start_time,
                                            stop_time = trial.stop_time,
                                            amplitude = trial.amplitude,
                                            pulse_duration = trial.pulse_duration,
                                            waveform_shape = str(trial['shape']),
                                            polarity = str(trial.polarity),
                                            run = trial.run,
                                            pulse_number = trial.pulse_number,
                                            event_period = trial.event_period,
                                            train_duration = trial.train_duration,
                                            train_period = trial.train_period,
                                            train_quantity = trial.train_quantity,
                                            contacts = str(trial.contacts),
                                            contact_positive = trial.contact_positive,
                                            contact_negative = trial.contact_negative,
                                            contact_negative_coord = trial.contact_negative_coord,
                                            contact_positive_coord = trial.contact_positive_coord,
                                            notes = str(trial.notes)
                                            )
                else:
                    self.nwbfile.add_trial(start_time = trial.start_time,
                                            stop_time = trial.stop_time,
                                            amplitude = trial.amplitude,
                                            pulse_duration = trial.pulse_duration,
                                            waveform_shape = trial['shape'],
                                            polarity = str(trial.polarity),
                                            run = trial.run,
                                            pulse_number = trial.pulse_number,
                                            event_period = trial.event_period,
                                            train_duration = trial.train_duration,
                                            train_period = trial.train_period,
                                            train_quantity = trial.train_quantity,
                                            contacts = str(trial.contacts),
                                            contact_positive = trial.contact_positive,
                                            contact_negative = trial.contact_negative,
                                            notes = str(trial.notes)
                                            )
                    

            # electrodes
            probe_IDs = units_df.probe.unique()

            device = self.nwbfile.create_device(name = device_name, description = "Orthogonal neuropixel rig with 3 Neuropixels and electrical stimulator")

            for i, id in enumerate(probe_IDs):
                electrode_name = 'probe'+str(id)
                description = "Neuropixels1.0_"+id
                location = "visual cortex"

                electrode_group = self.nwbfile.create_electrode_group(electrode_name,
                                                                description=description,
                                                                location=location,
                                                                device=device)
                
                #add channels to each probe
                for ch in range(option234_positions.shape[0]):
                    self.nwbfile.add_electrode(x=option234_positions[ch,0],y=0.,
                                        z=option234_positions[0,1],
                                        imp=0.0,location='none',
                                        filtering='high pass 300Hz',
                                        group=electrode_group)
            self.nwbfile.add_unit_column('probe', 'probe ID')
            self.nwbfile.add_unit_column('unit_id','cluster ID from KS2')
            self.nwbfile.add_unit_column('group', 'user label of good/mua')
            self.nwbfile.add_unit_column('no_spikes', 'total number of spikes across recording')
            self.nwbfile.add_unit_column('KSlabel', 'Kilosort label')
            self.nwbfile.add_unit_column('KSamplitude', 'Kilosort amplitude')
            self.nwbfile.add_unit_column('KScontamination', 'Kilosort ISI contamination')
            self.nwbfile.add_unit_column('template', 'Kilosort template')
            self.nwbfile.add_unit_column('ch', 'channel number')

            if spatial == True:
        
                self.nwbfile.add_unit_column('brain_reg', 'brain region of ch unit was recorded on')
                self.nwbfile.add_unit_column('distance_from_stim', 'List of euclidean distances to each stimulation contact: use trials dataframe to identify appropriate stim contact')
                self.nwbfile.add_unit_column('ccf_coordinates', 'the ccf brain space coordinates (AP, DV, ML) for the channel the unit was recorded on')
            


            for i,unit_row in units_df[units_df.group != 'noise'].iterrows():
                if spatial == True:
                    self.nwbfile.add_unit(probe=str(unit_row.probe),
                                    id = i,
                                    unit_id = str(unit_row.unit_id),
                                    spike_times=unit_row.spike_times,
                                    template= unit_row.template,
                                    no_spikes = unit_row.no_spikes,
                                    group= str(unit_row.group),
                                    KSlabel= str(unit_row.KSlabel),
                                    KSamplitude= unit_row.KSamplitude,
                                    KScontamination= unit_row.KScontamination,
                                    ch = unit_row.ch,
                                    brain_reg = str(unit_row.brain_reg),
                                    distance_from_stim = unit_row.distance_from_stim,
                                    ccf_coordinates = unit_row.ccf_coordinates
                                    )
                else:
                    self.nwbfile.add_unit(probe=str(unit_row.probe),
                                    id = i,
                                    unit_id = str(unit_row.unit_id),
                                    spike_times=unit_row.spike_times,
                                    template= unit_row.template,
                                    no_spikes = unit_row.no_spikes,
                                    group= str(unit_row.group),
                                    KSlabel= str(unit_row.KSlabel),
                                    KSamplitude= unit_row.KSamplitude,
                                    KScontamination= unit_row.KScontamination,
                                    ch = unit_row.ch
                    )   
                
                    
                

            # write
            with NWBHDF5IO(os.path.join(self.path,f'{self.mouse}.nwb'), 'w') as io:
                io.write(self.nwbfile)
                io.close()
        
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
                return io, nwb #as a writeable object   