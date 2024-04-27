import os, glob
import numpy as np
from pynwb import NWBHDF5IO, NWBFile


class Loader:
    def __init__(self, mouse, date):
        self.mouse = mouse
        self.date = date
        self.path = self.find_folder()
        self.check_for_nwb()


    #loading functions
    def find_folder(self):
        locations = [r'E:\\', r'C:\Users\jordan\Documents', r'Z:\\']
        
        for location in locations:
            search_pattern = os.path.join(location, f'*{self.mouse}*{self.date}*')
            matching_folders = glob.glob(search_pattern)
            
            if matching_folders:
                return matching_folders[0]
        
        raise FileNotFoundError(f"No matching folder found for mouse {self.mouse} and date {self.date} in locations {locations}.")
                    

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