import numpy as np
from open_ephys.analysis import Session

def OE(path):
        session = Session(path)
        recording = session.recordnodes[0].recordings[0]
        return recording

# need to add a polarity parameter checker to NWB file that parses the contacts column
def choose_stim_parameter(trials, amp=-100, pulse_number = 1, pulse_duration=100, polarity = 'bipolar', contact_pref = None, return_metadata = False):
    '''
    contact_pref: int, the contact index during the recording you want to load if multiple contacts were stimulated for a given parameter set 
    '''
    subset = stim_times = trials.loc[
        (trials['amplitude'] == amp) &
        (trials['pulse_number'] == pulse_number) &
        (trials['pulse_duration'] == pulse_duration) &
        (trials['polarity'] == polarity)]
    if len(subset.contact_positive.unique()) > 1 and contact_pref is None:
        print(f'These are the multiple unique cathodal {subset.contact_negative.unique()}')
        contact = subset.contact_negative.unique()[0]
        print(f'Returning the first one: {contact}')
        stim_times = np.array(subset.loc[subset.contact_negative == contact]['start_time'])
    elif len(subset.contact_positive.unique()) > 1 and contact_pref is not None:
        print(f'These are the multiple unique cathodal {subset.contact_negative.unique()}')
        contact = subset.contact_negative.unique()[contact_pref]
        print(f'Returning your contact preference indicated: {contact}')
        stim_times = np.array(subset.loc[subset.contact_negative == contact]['start_time'])
    else:  
        stim_times = np.array(subset['start_time'])

    if return_metadata:
        metadata = {'cat_contact': contact, 'all_contacts': {'anodal':subset.contact_positive, 'cathodal':subset.contact_negative}, 'run':subset.run.unique()} 
        return stim_times, metadata
    else:
        return stim_times
         
 
    

def stim_dictionary(trials):
    parameters = {}
    for run in trials.run.unique():
        amp = np.array(trials.loc[trials.run == run].amplitude)[0]
        pulse_width = np.array(trials.loc[trials.run == run].pulse_duration)[0]
        contacts = np.array(trials.loc[trials.run == run].contacts)[0]
        parameters[int(run)] = f'amp: {amp} ua, pw: {pulse_width} us, contacts: {contacts}'
    return parameters