import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,json,glob
from scipy.stats import linregress



def sort_dbs_runs(paths):
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

def getpaths_spikeglx(recording_folder):
    imec0_path = glob.glob(os.path.join(recording_folder,'*imec0'))[0]
    imec1_path = glob.glob(os.path.join(recording_folder,'*imec1'))[0]
    imec2_path = glob.glob(os.path.join(recording_folder,'*imec2'))[0]
    
    return imec0_path, imec1_path, imec2_path
def lfp_var_spikeglx(lfp_binpath):
    lfp = np.memmap(os.path.join(lfp_binpath), dtype=np.int16, mode='r')
    lfp=lfp.reshape(-1,385)
    one_sec = 10*2500
    start = int(len(lfp)/2)
    ten_sec = int(10*one_sec)
    chunk = lfp[start:start+ten_sec,0:300]
    var_list = []
    for i in range(300):
        mini = np.var(chunk[:,i])
        var_list.append(mini)
    var = np.array(var_list)
    return var



def rms_fxn(chunk):
    rms = np.sqrt(np.mean(chunk**2))
    return rms

def threshold_crossings(ap_filepath): # gives you a decent approximation of spike quality by channel 
    chunk = []
    ap = np.memmap(ap_filepath, dtype=np.int16, mode='r')
    ap=ap.reshape(-1,385)
    ap_gain=500
    for i in range(300):
        ch = ap[:,i]
        ch_chunk = []
        one_sec = 30000
        start = 100000
        interval = int(len(ch)/12)

        for i in range(10):
            mini_chunk = ch[start:start+one_sec]
            ch_chunk.append(mini_chunk)
            start += interval
        chunk.append(ch_chunk)

    chunk = np.array(chunk)
    mean_rms_list = []
    for i in range(len(chunk)):
        rms_list = []
        for x in range(len(chunk[i])):
            mini_rms = rms_fxn(chunk[i][x])
            rms_list.append(mini_rms)
        rms = np.array(rms_list)    
        mean_rms = np.mean(rms)
        mean_rms_list.append(mean_rms)
    threshold = [x * 2.5 for x in mean_rms_list] 
    crossings = []
    for ch in range(len(mean_rms_list)):
        ch_crossings = len(np.where(chunk[ch,:,:] > threshold[ch])[1])
        crossings.append(ch_crossings)
    return crossings



def jlh_prime(imec0,imec1,imec2,nidaq,stim_df): #text files with 1hz sync pulses

    imec0adjuster = imec0 - nidaq
    imec1adjuster = imec1 - nidaq
    imec2adjuster = imec2 - nidaq

    
    indexer = np.arange(0,len(imec0adjuster),1)
    slope, intercept, r, p, se = linregress(indexer,imec0adjuster)

    stim = np.array(stim_df.electrical_times_first)
    imec0_stim = []
    for i in range(len(stim)):
        adjuster = (slope*stim[i])+intercept
        adj_time = stim[i] + adjuster
        imec0_stim.append(adj_time)

  

    indexer = np.arange(0,len(imec1adjuster),1)
    slope, intercept, r, p, se = linregress(indexer,imec1adjuster)

    imec1_stim = []
    for i in range(len(stim)):
        adjuster = (slope*stim[i])+intercept
        adj_time = stim[i] + adjuster
        imec1_stim.append(adj_time)

    
    indexer = np.arange(0,len(imec2adjuster),1)
    slope, intercept, r, p, se = linregress(indexer,imec2adjuster)
    
    imec2_stim = []
    for i in range(len(stim)):
        adjuster = (slope*stim[i])+intercept
        adj_time = stim[i] + adjuster
        imec2_stim.append(adj_time)
    stim_df['imec0_stim'] = imec0_stim
    stim_df['imec1_stim'] = imec1_stim
    stim_df['imec2_stim'] = imec2_stim
    return stim_df

def load_lfp_spikeglx(lf_binpath): #
    lfp = np.memmap(lf_binpath, dtype=np.int16, mode='r')
    lfp=lfp.reshape(-1,385)
    return lfp


def lfp_plot_single(lfp, stim_times, ch, pre = 4, post = 20, lfp_gain = 250, label = '', save = False, savepath = '', format='png'):
   sample_rate = 2500
   pre_sample = int((pre/1000)*sample_rate)
   post_sample = int((post/1000)*sample_rate)
   time_window = np.linspace(-pre,post,(pre_sample + post_sample)) 
    
   response = np.zeros((np.shape(stim_times)[0],(pre_sample+post_sample),385))
   
   fig = plt.figure()
   for i,t in enumerate(stim_times):
         t = int(t * 2500) #lfp sampling rate
         chunk = lfp[t-pre_sample:t+post_sample,:]
         corrected_chunk = chunk-np.median(chunk,axis=0)
         corrected_chunk = corrected_chunk.T - np.median(corrected_chunk[:,280:300],axis=1) #median correct to just outside of the brain
         corrected_chunk=1e6*corrected_chunk.T/(512.*lfp_gain)
         response[i,:,:]=corrected_chunk
         plt.plot(time_window,corrected_chunk[:,ch],color = 'k', alpha=0.05)
   
   plt.plot(time_window,np.mean(response,axis=0)[:,ch],label=label)
   plt.ylim((-200,200))
   plt.gca().axvline(0,ls='--',color='k')
   plt.gca().axvline(0.2,ls='--',color='k')     
   leg = plt.legend(loc='lower right',fontsize = 'small')
   plt.ylabel('uV')
   plt.title('ch=' + str(ch))

   if save==True:
     plt.gcf().savefig(savepath,format=format,dpi=600)
   


   return fig 

def lfp_plot_allch(lfp, stim_times, pre = 4, post = 20, n_chs = 300, lfp_gain=250, title='', save = False, savepath='',format =  'png'):
   sample_rate = 2500
   pre_sample = int((pre/1000)*sample_rate)
   post_sample = int((post/1000)*sample_rate)
   time_window = np.linspace(-pre,post,(pre_sample + post_sample))   
    
   response = np.zeros((np.shape(stim_times)[0],(pre_sample+post_sample),385))
   for i,t in enumerate(stim_times):
        t = int(t * sample_rate) #lfp sampling rate
        chunk = lfp[t-pre_sample:t+post_sample,:]
        corrected_chunk = chunk-np.median(chunk,axis=0)
        corrected_chunk = corrected_chunk.T - np.median(corrected_chunk[:,n_chs-10:n_chs+10],axis=1) #median correct to just outside of the brain
        corrected_chunk=1e6*corrected_chunk.T/(512.*lfp_gain)
        response[i,:,:]=corrected_chunk
   mean_response = np.mean(response,axis=0).T[:n_chs]
   d = np.array(mean_response[::2,:] + mean_response[1::2])/2.
   fig = plt.figure(figsize=(8,24))
   
   for ch in range(int(n_chs/2)): plt.plot(time_window,d[ch]+ch*20)
   plt.ylim((-100,3000))
   plt.xlabel('time from stimulus onset (ms)')
   plt.ylabel('uV')
   plt.title(title)
   plt.gca().axvline(0,ls='--',color='r')
   if save==True:
       plt.gcf().savefig(savepath,format=format,dpi=600)
   

   return fig

def load_ap(ap_binpath):
    ap = np.memmap(ap_binpath, dtype=np.int16, mode='r')
    ap=ap.reshape(-1,385)
    return ap


def ap_plot_single(ap, stim_times, ch, pre = 4, post = 10, ap_gain = 500, label = '', save=False, savepath = '',format='png'):
   #variables
   sample_rate = 30000
   pre_sample = int((pre/1000)*sample_rate)
   post_sample = int((post/1000)*sample_rate)
   time_window = np.linspace(-pre,post,(pre_sample + post_sample))      
 
   fig = plt.figure()
    
   response = np.zeros((np.shape(stim_times)[0],(pre_sample+post_sample),385))
   for i,t in enumerate(stim_times):
        t = int(t * 30000) #ap sampling rate
        chunk = ap[t-pre_sample:t+post_sample,:]
        corrected_chunk = chunk-np.median(chunk,axis=0)
        #corrected_chunk = corrected_chunk.T - np.median(corrected_chunk[:,190:220],axis=1) #median correct to just outside of the brain
        corrected_chunk=1e6*corrected_chunk/(512.*ap_gain)
        response[i,:,:]=corrected_chunk
        plt.plot(time_window,corrected_chunk[:,ch],color = 'k', alpha=0.05)
    
   plt.plot(time_window,np.mean(response,axis=0)[:,ch],label=label)
   plt.ylim((-500,500))
   plt.gca().axvline(0,ls='--',color='k')
   plt.gca().axvline(0.00002,ls='--',color='k')
   leg = plt.legend(loc='lower right',fontsize = 'small')
   plt.ylabel('uV')
   plt.title('ch=' + str(ch))
   if save==True:
       plt.gcf().savefig(savepath,format=format,dpi=600)
   return fig 


def ap_plot_allch(ap, stim_times, pre = 4, post = 10, n_chs = 300, ap_gain = 500, title = '', n_stims = 15, spacing_mult = 100, save=False, savepath = '',format='png'):
   #variables
   sample_rate = 30000
   pre_sample = int((pre/1000)*sample_rate)
   post_sample = int((post/1000)*sample_rate)
   time_window = np.linspace(-pre,post,(pre_sample + post_sample))   
      
   plt.figure()
    
   response = np.zeros((np.shape(stim_times)[0],(pre_sample+post_sample),385))
   for i,t in enumerate(stim_times):
        t = int(t * 30000) #ap sampling rate
        chunk = ap[t-pre_sample:t+post_sample,:]
        corrected_chunk = chunk-np.median(chunk,axis=0)
        #corrected_chunk = corrected_chunk.T - np.median(corrected_chunk[:,190:220],axis=1) #median correct to just outside of the brain
        corrected_chunk=1e6*corrected_chunk/(512.*ap_gain)
        response[i,:,:]=corrected_chunk
       
    #mean_response = np.mean(response,axis=0).T[:n_chs]
   fig=plt.figure(figsize=(10,25))
   
   stim_sample = np.linspace(0,len(stim_times)-1,n_stims) #choose 8 to plot
   stim_sample = stim_sample.astype(int)
   for stim in stim_sample:
        for ch in range(0,n_chs,2): plt.plot(time_window,response[stim,:,ch]+ch*spacing_mult)
   
   plt.ylim((-500,32000))
   plt.xlabel('time from stimulus onset (ms)')
   plt.ylabel('uV')
   plt.title(title)
   plt.gca().axvline(0,ls='--',color='r')
   if save==True:
       plt.gcf().savefig(savepath,format=format,dpi=600)
   #plt.show()

   return fig 

def ap_heatmap(ap, stim_times, ch, pre = 4, post = 10, n_chs = 300, ap_gain = 500, label = '', n_stims = 8, save=False, savepath = '',format='png',vmax=200):
   #variables
   sample_rate = 30000
   pre_sample = int((pre/1000)*sample_rate)
   post_sample = int((post/1000)*sample_rate)
   time_window = np.linspace(-pre,post,(pre_sample + post_sample))      
 
   fig = plt.figure()
    
   response = np.zeros((np.shape(stim_times)[0],(pre_sample+post_sample),385))
   for i,t in enumerate(stim_times):
        t = int(t * 30000) #ap sampling rate
        chunk = ap[t-pre_sample:t+post_sample,:]
        corrected_chunk = chunk-np.median(chunk,axis=0)
        #corrected_chunk = corrected_chunk.T - np.median(corrected_chunk[:,190:220],axis=1) #median correct to just outside of the brain
        corrected_chunk=1e6*corrected_chunk/(512.*ap_gain)
        response[i,:,:]=corrected_chunk
   
   plt.imshow(np.absolute(response[:,:,ch]),vmax=vmax)
    
   #plt.plot(time_window,np.mean(response,axis=0)[:,ch],label=label)
   #plt.ylim((-500,500))
   #plt.gca().axvline(0,ls='--',color='k')
   #plt.gca().axvline(0.00002,ls='--',color='k')
   #leg = plt.legend(loc='lower right',fontsize = 'small')
   plt.ylabel('Trials')
   plt.title('ch=' + str(ch))
   if save==True:
       plt.gcf().savefig(savepath,format=format,dpi=600)
   return fig 


def sglx_nidaq_analog(bin_path, seconds=True):
    
    #Memory map the bin file and parse into binary lines
    mm = np.memmap(bin_path,dtype='int16')
    a0 = mm[0::9]
    a1 = mm[1::9]
    a2 = mm[2::9]
    a3 = mm[3::9]
    a4 = mm[4::9]
    a5 = mm[5::9]
    a6 = mm[6::9]
    a7 = mm[7::9]
    analog = {'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7}
    return analog