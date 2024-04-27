import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import re
from tqdm.notebook import tqdm as tqdm

from open_ephys.analysis import Session

from jlh_ephys.loader import Loader
from jlh_ephys.raw import Raw
from jlh_ephys.nwb_tools import NWB_Tools
from jlh_ephys.plotter import Plotter
from jlh_ephys.preprocess import Preprocess

from ccf_3D.tools import herbs_processing as hp
from ccf_3D.tools.metrics import distance


import logging

logging.basicConfig(level=logging.INFO)

class Analysis:
    def __init__(self, mouse, date, stim=True):
        self.mouse = mouse
        self.date = date
        self.error_info = []

        self.initialize_loader()
        if self.processed:
            self.load_processed_data()
        else:
            self.load_unprocessed_data()

        self.initialize_additional_classes()

    def initialize_loader(self):
        self.loader = Loader(self.mouse, self.date)
        self.path = self.loader.path
        self.processed = self.loader.processed

    def load_processed_data(self):
        self.trials = self.loader.trials
        self.units = self.loader.units
        #self.units.set_index('unit_id', inplace = True)
        self.parameters = self.loader.parameters
        self.get_probes()
        self.get_contacts() 

    def load_unprocessed_data(self):
        try:
            logging.info('Attempting to create stim-df')
            self.preprocess = Preprocess(self)
            self.stim_df = self.preprocess.stim_df
            self.get_contacts()
        except Exception as e:
            self.log_error("Could not create stim-df", e)

    def initialize_additional_classes(self):
        for class_name, class_obj in [ ("Raw", "Raw"), 
                                       ("NWB_Tools", "NWB_Tools"), 
                                       ("Plotter", "Plotter")]:
            self.try_initialize_class(class_name, class_obj)

    def try_initialize_class(self, class_name, class_obj):
        try:
            setattr(self, class_name.lower(), eval(f"{class_obj}(self)"))
        except Exception as e:
            self.log_error(f"Could not load {class_name} class", e)

    def log_error(self, message, exception_obj):
        error_info = {
            "error": str(exception_obj),
            "mouse": self.mouse,
            "date": self.date,
            "message": message
        }
        self.error_info.append(error_info)
        logging.error(f"{message}. An error occurred: {exception_obj}. Class partially initialized.")

    
    def get_probes(self):
        # Extract unique probe names
        probe_IDs = self.units.probe.unique()
        
        # Format them as "probeA", "probeB", etc., handling both 'A' and 'probeA' formats
        probe_names = [f"probe{probe_ID}" if not probe_ID.lower().startswith('probe') else probe_ID for probe_ID in probe_IDs]
        
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
    
    @staticmethod
    def load_coords_from_file(file_path):
        try:
            with open(file_path, 'rb') as file:
                return pkl.load(file)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error loading file: {e}")
        return None

    def get_electrode_coords(self):
        coords_path = os.path.join(self.path, 'final_coords', 'coords_dict.pkl')
        if os.path.exists(coords_path):
            probe_coords = self.load_coords_from_file(coords_path)
            if probe_coords:
                self.probe_coords = probe_coords
                self.stim_coords = probe_coords['stim']
                logging.info('Final adjusted coordinates loaded')
            else:
                logging.error('Failed to load final adjusted coordinates')
        else:
            try:
                probes, stim = hp.load_herbs(mouse=self.mouse, probe_names=self.probes)
                self.probe_coords = hp.multi_neuropixel_coords(probes)
                self.stim_coords = hp.stim_coords(stim)
            except Exception as e:
                logging.error(f"Could not load herbs: {e}")
                default_coords_path = r"E:\probe_preset\coords_dict.pkl"
                probe_coords = self.load_coords_from_file(default_coords_path)
                if probe_coords:
                    self.probe_coords = probe_coords
                    self.stim_coords = probe_coords['stim']
                    logging.info('Using default coordinates, you should probably adjust and verify them :)')
                else:
                    logging.error('Failed to load default coordinates')

    def get_dists(self, contact = 10):
        nn_contacts_labels = [6,11,3,14,1,16,2,15,5,12,4,13,7,10,8,9]
        contact = nn_contacts_labels.index(contact)
        if self.probe_coords is None:
            self.get_electrode_coords()
        distances = {}
        for probe, coords in self.probe_coords.items():
            if probe != 'stim':
                dist = distance(coords, self.probe_coords['stim'][contact])
                distances[probe] = dist
        self.distances = distances
        return distances


    def get_brain_regs(self):
        region_path = os.path.join(self.path, 'final_coords', 'regions_dict.pkl')
        if os.path.exists(region_path): # see if final brain regions have been saved
            brain_regs = self.load_coords_from_file(region_path)
            if brain_regs:
                self.brain_regs = brain_regs
                logging.info('Final adjusted brain regions loaded')
            else:
                logging.error('Failed to load final brain regions')
        else:
            try:
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
                brain_regs = {'probeA':brainreg_A, 'probeB':brainreg_B, 'probeC': brainreg_C}
                if brain_regs:
                    self.brain_regs = brain_regs
                    logging.info('Brain regions from Herbs loaded')
            
            except Exception as e:
                logging.error(f"Could not load herbs: {e}")
                default_regions_path = r"E:\probe_preset\regions_dict.pkl"
                brain_regs = self.load_coords_from_file(default_regions_path)
                if brain_regs:
                    self.brain_regs = brain_regs
                    logging.info('Using default brain_regions, you should probably adjust and verify them :)')

                else:
                    logging.error('Failed to load default brain_regions')

    def get_fs_rs(self, plot = False):
        from dlab import utils
        
        wave_ = []
        for i,template in self.units.template.items():
            wave_.append(utils.get_peak_waveform_from_template(np.array(template)))
        self.units['waveform']=wave_

        df = utils.classify_waveform_shape(self.units, plots=False, save_plots = False, kmeans=0)
        if plot == True:
            durations = np.array(df.waveform_duration)
            PTratio = np.array(df.waveform_PTratio)
            repolarizationslope = np.array(df.waveform_repolarizationslope)
            utils.plot_waveform_classification(durations, PTratio, repolarizationslope, df,save_plots=False)



