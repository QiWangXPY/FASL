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
import tqdm
import hrvanalysis as hrvana
from concurrent import futures
import os

import sys

stage_reverse = {5:3, 4:2, 3:2, 2:1, 1:1, 0:0}

#cnt_clean = np.zeros(4)

import glob
from scipy.interpolate import interp1d

#mode = 'train'
mode = 'cfs'

#all_subjects = np.unique([i[-7:-3] for i in glob.glob("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/"+mode+"/raw/*.pt")])
all_subjects = np.unique([i[-9:-3] for i in glob.glob("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/"+mode+"/raw/*.pt")])

hz = 128

nyquist_freq = hz/2  # 奈奎斯特频率
order = 8
cutoff_freq = 10  # Hz, 选择合适的截止频率
b, a = signal.butter(order, cutoff_freq / nyquist_freq, btype='lowpass', analog=False)
b_l, a_l = signal.butter(4, 0.5 / nyquist_freq, btype='lowpass', analog=False)

def process_extract_features(file_id):#三个文件列表
#for file_id in tqdm.tqdm(all_subjects):
    

    y_true = list()
    
    x = torch.load("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/"+mode+"/raw/x_"+str(file_id)+".pt")
    
    mean = torch.mean(x)
    std = torch.std(x)
    
    # 标准化变量
    std_x = (x - mean) / std
    std_x = signal.filtfilt(b, a, std_x)
    
    ppg_row = signal.resample(std_x, int(std_x.shape[0]/(hz/64)))
    
    ppg_peaks_l = nk.ppg_findpeaks(-1 * std_x, sampling_rate=hz)
    ppg_peaks_h = nk.ppg_findpeaks(std_x, sampling_rate=hz)
    
    hr_x = list()
    hr = list()
    bl_x = list()
    blood = list()
    
    for i in range(len(ppg_peaks_l['PPG_Peaks']) - 1):
        #heart rate
        hr.append(ppg_peaks_l['PPG_Peaks'][i+1] - ppg_peaks_l['PPG_Peaks'][i])
        hr_x.append(ppg_peaks_l['PPG_Peaks'][i])
    
    for i in range(len(ppg_peaks_h['PPG_Peaks']) - 1):
        blood.append(np.sum(-1*std_x[ppg_peaks_h['PPG_Peaks'][i]:ppg_peaks_h['PPG_Peaks'][i+1]]))
        bl_x.append(ppg_peaks_h['PPG_Peaks'][i])
    
    f_hr = interp1d(hr_x, hr, fill_value="extrapolate")
    f_bl = interp1d(bl_x, blood, fill_value="extrapolate")
    f_h = interp1d(ppg_peaks_h['PPG_Peaks'], std_x[ppg_peaks_h['PPG_Peaks']], fill_value="extrapolate")
    f_l = interp1d(ppg_peaks_l['PPG_Peaks'], std_x[ppg_peaks_l['PPG_Peaks']], fill_value="extrapolate")
    
    RR_list = f_hr([i for i in range(0, std_x.shape[0], int(hz / 2))])/hz
    blood_list = f_bl([i for i in range(0, std_x.shape[0], int(hz / 2))])/hz
    ba_list = signal.resample(signal.filtfilt(b_l, a_l, std_x), int(std_x.shape[0] / hz * 2))
    am_list = f_h([i for i in range(0, std_x.shape[0], int(hz / 2))]) - f_l([i for i in range(0, std_x.shape[0], int(hz / 2))])
    
    ppg_sig = np.concatenate(([RR_list], [am_list], [ba_list], [blood_list]), axis = 0)

    ppg_rrfea=list()
    ppg_amfea=list()
    ppg_bafea=list()
    ppg_blood=list()
    
    for i in tqdm.tqdm(range(0, int(std_x.shape[0]/hz * 2 - 40), 20)):
        if (len(ppg_blood)>=3664):
            break
        temp = hrvana.get_time_domain_features(RR_list[i:i+60] * 1000)
        ppg_rr = [temp['mean_nni'], temp['sdnn'], temp['sdsd']]
        temp = hrvana.get_frequency_domain_features(RR_list[i:i+60] * 1000)
        ppg_rr = ppg_rr + [temp['vlf'], temp['lf'], temp['hf'],
                        temp['lf_hf_ratio'], temp['total_power']]
        ppg_rrfea.append(ppg_rr)
        
        temp = hrvana.get_time_domain_features(am_list[i:i + 60] * 1000)
        ppg_rr = [temp['mean_nni'], temp['sdnn'], temp['sdsd']]
        temp = hrvana.get_frequency_domain_features(am_list[i:i + 60] * 1000)
        ppg_rr = ppg_rr + [temp['vlf'], temp['lf'], temp['hf'],
                        temp['lf_hf_ratio'], temp['total_power']]
        ppg_amfea.append(ppg_rr)

        temp = hrvana.get_time_domain_features(ba_list[i:i + 60] * 1000)
        ppg_rr = [temp['mean_nni'], temp['sdnn'], temp['sdsd']]
        temp = hrvana.get_frequency_domain_features(ba_list[i:i + 60] * 1000)
        ppg_rr = ppg_rr + [temp['vlf'], temp['lf'], temp['hf'],
                        temp['lf_hf_ratio'], temp['total_power']]
        ppg_bafea.append(ppg_rr)
        
        temp = hrvana.get_time_domain_features(blood_list[i:i + 60] * 1000)
        ppg_rr = [temp['mean_nni'], temp['sdnn'], temp['sdsd']]
        temp = hrvana.get_frequency_domain_features(blood_list[i:i + 60] * 1000)
        ppg_rr = ppg_rr + [temp['vlf'], temp['lf'], temp['hf'],
                        temp['lf_hf_ratio'], temp['total_power']]
        ppg_blood.append(ppg_rr)

    ppg_fea = np.concatenate((ppg_rrfea, ppg_amfea, ppg_bafea, ppg_blood), axis = 1)
    
    ppg_fea = np.transpose(ppg_fea, (1, 0))

    #torch.save(torch.tensor(ppg_row), '/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/'+mode+'/ppg/x_'+file_id+'.pt')
    torch.save(torch.tensor(ppg_sig), '/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/'+mode+'/sig_4/x_'+file_id+'.pt')
    torch.save(torch.tensor(ppg_fea), '/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/'+mode+'/fea_32/x_'+file_id+'.pt')
    
already = np.unique([i[-9:-3] for i in os.listdir("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/da300s/"+mode+"/sig_4/")])

#with futures.ProcessPoolExecutor(max_workers=10) as pool:
for PID in all_subjects:
    temp = PID #"%04d" % PID
    if temp not in already:
        print(PID)
        #pool.submit(process_extract_features, PID)
        process_extract_features(PID)
            #sys.exit()