import os,glob
import numpy as np
import pandas as pd
import pendulum

from jlh_ephys.utils import OE

class Preprocess:
    def __init__(self, analysis_obj):
        self.analysis = analysis_obj
        self.path = self.analysis.path
        self.mouse = self.analysis.mouse
        self.date = self.analysis.date
        self.events_df = None
        try:
            self.get_stim_df()
        except:
            print('Could not create stim_df')
            self.stim_df = None
        
    
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
        if self.events_df is None:
            recording = OE(self.path)
            ## load events dataframe
            df = recording.events
        else:
            df = self.events_df
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
        self.stim_df = stim_df
        