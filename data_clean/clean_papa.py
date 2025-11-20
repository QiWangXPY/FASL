import torch

from scipy import signal
from scipy import integrate

#new_x = signal.resample(x, 50)

import io
import numpy as np
import pandas as pd
import neurokit2 as nk
#x_train = torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa/x_train.pt')
#y_train = torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa/y_train.pt')
from tqdm import tqdm
import hrvanalysis as hrvana
from concurrent import futures
import os

import pickle

from scipy.stats import kurtosis, skew, entropy
from torch_ecg._preprocessors import Normalize
from scipy.signal import argrelmax, argrelmin

from scipy.integrate import trapz


import sys

stage_reverse = {5:3, 4:2, 3:2, 2:1, 1:1, 0:0}

#cnt_clean = np.zeros(4)

import glob
from scipy.interpolate import interp1d

mode = 'self'

#all_subjects = np.unique([i[-7:-3] for i in glob.glob("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/"+mode+"/raw/*.pt")])

#all_subjects = np.unique([i[-9:-3] for i in glob.glob("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/"+mode+"/raw/*.pt")])

#all_subjects = [i[-7:-3] for i in os.listdir('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/mesa/label')]
#all_subjects = [i[-9:-3] for i in os.listdir('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/cfs/label')]
all_subjects = [i for i in os.listdir('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/selfcollect/unlabel/ppg')]

hz = 100

nyquist_freq = hz/2  # 奈奎斯特频率
order = 8
cutoff_freq = 10  # Hz, 选择合适的截止频率
b, a = signal.butter(order, cutoff_freq / nyquist_freq, btype='lowpass', analog=False)
b_l, a_l = signal.butter(4, 0.5 / nyquist_freq, btype='lowpass', analog=False)

def extract_svri(single_waveform):
    """
    https://github.com/qiriro/PPG

    Args:
        single_waveform (np.array): input ppg segment
    Returns:
        svri (float): svri value
    """
    def __scale(data):
        data_max = max(data)
        data_min = min(data)
        return [(x - data_min) / (data_max - data_min) for x in data]
    max_index = np.argmax(single_waveform)
    single_waveform_scaled = __scale(single_waveform)
    return np.mean(single_waveform_scaled[max_index:]) / np.mean(single_waveform_scaled[:max_index])

def skewness_sqi(x, axis=0, bias=True, nan_policy='propagate'):
    """
    Computes ppg skewness using skew from scipy
    """
    return skew(x, axis, bias, nan_policy)

def compute_ipa(signal, fs):
    """
    Computes IPA by identifying the first dicrotic notch

    Args:
        signal(np.array): input ppg segment
        fs (int): ppg frequency
    Returns:
        ipa (float): IPA value
    """

    try:
        maxima_index = argrelmax(signal, order=fs // 5)[0]
        minima_index = argrelmin(signal, order=fs // 5)[0]
        
        
        
        single_beat = signal#[minima_index[0]:minima_index[1]]
        minima_beats = argrelmin(single_beat)[0]
        
        minima_index
        minima_beat = minima_beats[0]
        
        sys_values = single_beat[:minima_beat]
        dias_values = single_beat[minima_beat:]
        
        sys_x_values = np.linspace(0, len(sys_values) - 1, len(sys_values)) 
        dias_x_values = np.linspace(0, len(dias_values) - 1, len(dias_values)) 
        
        sys_phase = trapz(y=sys_values, x=sys_x_values)
        dias_phase = trapz(y=dias_values, x=dias_x_values)
        ipa = sys_phase/dias_phase

    
    except IndexError as e:
        ipa = 0 
        
    return ipa

def process_extract_features(file_id):#三个文件列表
#for file_id in tqdm.tqdm(all_subjects):
    
    
    x = torch.load("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/selfcollect/unlabel/ppg/"+str(file_id))
    x = x.reshape([-1])
    
    mean = torch.mean(x)
    std = torch.std(x)
    
    # 标准化变量
    std_x = (x - mean) / std
    std_x = signal.filtfilt(b, a, std_x)
   
    std_x = nk.ppg_clean(std_x, sampling_rate=hz)
    
    ppg_peaks_l = nk.ppg_findpeaks(-1 * std_x, sampling_rate=hz)
    ppg_peaks_h = nk.ppg_findpeaks(std_x, sampling_rate=hz)

    
    svri_li = list()
    sqi_li = list()
    ipa_li = list()
    
    for i in tqdm(range(len(ppg_peaks_h['PPG_Peaks']) - 1)):
        svri_li.append(extract_svri( -1 * std_x[ppg_peaks_h['PPG_Peaks'][i]:ppg_peaks_h['PPG_Peaks'][i+1]]))
        sqi_li.append(skewness_sqi( -1 * std_x[ppg_peaks_h['PPG_Peaks'][i]:ppg_peaks_h['PPG_Peaks'][i+1]]))
        ipa_li.append(compute_ipa( -1 * std_x[ppg_peaks_h['PPG_Peaks'][i]:ppg_peaks_h['PPG_Peaks'][i+1]], hz))

    with open('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/selfcollect/unlabel/papa/ipa/' + str(file_id), 'wb') as f:
        pickle.dump(ipa_li, f)
    with open('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/selfcollect/unlabel/papa/sqi/' + str(file_id), 'wb') as f:
        pickle.dump(sqi_li, f)
    with open('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/selfcollect/unlabel/papa/svri/' + str(file_id), 'wb') as f:
        pickle.dump(svri_li, f)
    
#already = np.unique([i[-7:-3] for i in os.listdir("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/mesa/papa/ipa")])

already = np.unique(os.listdir('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/selfcollect/unlabel/papa/ipa/'))

#with futures.ProcessPoolExecutor(max_workers=10) as pool:
for PID in all_subjects:
    temp = PID #"%04d" % PID
    if temp not in already:
        print(PID)
        #pool.submit(process_extract_features, PID)
        process_extract_features(PID)
            #sys.exit()