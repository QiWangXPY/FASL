import matplotlib.pyplot as plt
import numpy as np
import tqdm

import os

import sys 
sys.path.append("../..") 

from source import utils
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.interval import Interval
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.mesa.mesa_psg_service import MesaPSGService
from source.mesa.mesa_time_based_service import MesaTimeBasedService
from source.mesa.metadata_service import MetadataService
from source.mesa.mesa_data_service import MesaDataService
import random

import torch
import pyedflib as pyedflib

all_files = MetadataService.get_all_files()
all_subjects = []
for file in all_files:
    file_id = file[-8:-4]
    all_subjects.append(file_id)
random.shuffle(all_subjects)
count = 0
spl='test'
for file_id in tqdm.tqdm(all_subjects):
    edf_file = pyedflib.EdfReader('/extern2/zgz/wq/sleep/sleep_classifiers' + '/data/mesa/polysomnography/edfs/mesa-sleep-' + file_id + '.edf')
    raw_labeled_sleep = MesaPSGService.load_raw(file_id)
    signal_labels = edf_file.getSignalLabels()
    ppg = edf_file.readSignal(signal_labels.index('Pleth'))
    y = list()
    x = list()
    freq = int(edf_file.getSampleFrequencies()[signal_labels.index('Pleth')])
    for i in raw_labeled_sleep[1:-1]:
        for start in range(int(i[1]),int(i[1]+i[2]),30):
            beg = start - 135
            end = start + 165
            if(len(ppg[beg*freq:end*freq]) == 300*freq):
                y.append([i[0], start, int(file_id)])
    beg = int(raw_labeled_sleep[1][1] * freq)
    end = int((raw_labeled_sleep[-2][1] + raw_labeled_sleep[-2][2]) * freq)
    
    x = ppg[beg : end]
    
    '''if not os.path.exists('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+spl):
        os.makedirs('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+spl)'''
    
    torch.save(torch.tensor(np.array(x)),'/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+spl+'/raw/x_'+str(file_id)+'.pt')
    torch.save(torch.tensor(np.array(y)),'/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+spl+'/raw/y_'+str(file_id)+'.pt')
    count+=1
    if(count>200):
        spl='train'