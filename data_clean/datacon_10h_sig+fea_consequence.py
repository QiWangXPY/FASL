import argparse

from copy import copy, deepcopy


from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

import torch.nn as nn
from torchsummary import summary

import sys, os
sys.path.append('./utilities')
import pandas as pd
import numpy as np
import random
import pickle
from scipy.signal import resample

from scipy.special import softmax
from tqdm import tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_path = './'
        
mode = 'train'
with open(data_path + 'cfs/'+mode+'_ref.pkl', 'rb') as f:
    ref_all = pickle.load(f)
with open(data_path + 'cfs/'+mode+'_std.pkl', 'rb') as f:
    std = pickle.load(f)
with open(data_path + 'cfs/'+mode+'_fea.pkl', 'rb') as f:
    fea = pickle.load(f)
with open(data_path + 'cfs/'+mode+'_sig.pkl', 'rb') as f:
    sig = pickle.load(f)
with open(data_path + 'cfs/'+mode+'_ppg.pkl', 'rb') as f:
    ppg = pickle.load(f)
'''
mode = mode + '_ahi'
with open(data_path + 'mesa/'+mode+'_label.pkl', 'rb') as f:
    test_label = pickle.load(f)
with open(data_path + 'mesa/'+mode+'_std.pkl', 'rb') as f:
    std = pickle.load(f)
with open(data_path + 'mesa/'+mode+'_fea.pkl', 'rb') as f:
    fea = pickle.load(f)
with open(data_path + 'mesa/'+mode+'_ref.pkl', 'rb') as f:
    ref_all = pickle.load(f)
'''

mask = torch.ones(32, dtype=torch.bool)
mask[[22, 30]] = False

def adjust_length_1D(tensor, target_length = 1200 * 1024):
    current_length = tensor.size(0)

    if current_length < target_length:
        # 如果长度不足，补零
        padding = target_length - current_length
        adjusted_tensor = torch.nn.functional.pad(tensor, (0, padding))
    else:
        # 如果长度超过，截断
        adjusted_tensor = tensor[:target_length]

    return adjusted_tensor
def adjust_length_2D(tensor, target_length = 1200 * 1024):
    
    current_length = tensor.size(1)

    if current_length < target_length:
        # 如果长度不足，补零
        padding = target_length - current_length
        adjusted_tensor = torch.nn.functional.pad(tensor, (0, padding, 0, 0))
    else:
        # 如果长度超过，截断
        adjusted_tensor = tensor[:, :target_length]

    return adjusted_tensor

#label = list()
for i in tqdm(ref_all):
    ref = ref_all[i]
    
    ppg_temp = ppg[ref]
    ppg_temp = (ppg_temp - std[2][ref][0:1])/std[2][ref][1:2]
    ppg_temp = torch.tensor( resample(ppg_temp, int(ppg_temp.shape[0]/64/30*1024)) )
    ppg[ref] = adjust_length_1D(ppg_temp, 1200 * 1024)
    
    sig_temp = sig[ref]
    sig_temp = (sig_temp - std[1][ref][:,0:1])/std[1][ref][:,1:2]
    sig_temp = adjust_length_2D(sig_temp, 1200 * 60)
    sig[ref] = sig_temp
    
    fea_temp = fea[ref]
    fea_temp = (fea_temp - std[0][ref][:,0:1])/std[0][ref][:,1:2]
    fea_temp = adjust_length_2D(fea_temp, 1199 * 3 + 1)
    fea[ref] = fea_temp
    

    #label.append( torch.tensor(test_label[test_label[:, 7] == i][:, 6].astype(float)) )

with open(data_path + 'cfs/10h_input/'+mode+'_ppg.pkl', 'wb') as f:
    pickle.dump(ppg, f)
with open(data_path + 'cfs/10h_input/'+mode+'_fea_conseq.pkl', 'wb') as f:
    pickle.dump(fea, f)
with open(data_path + 'cfs/10h_input/'+mode+'_sig_conseq.pkl', 'wb') as f:
    pickle.dump(sig, f)

'''

class Load_10h_Dataset(Dataset):
    def __init__(self, mode):
        super(Load_10h_Dataset, self).__init__()
        
        data_path = '../da300s/'
        
        self.mode = mode
        with open(data_path + 'mesa/'+mode+'_fea.pkl', 'rb') as f:
            self.fea = pickle.load(f)
        with open(data_path + 'mesa/'+mode+'_sig.pkl', 'rb') as f:
            self.sig = pickle.load(f)
        with open(data_path + 'mesa/'+mode+'_ppg.pkl', 'rb') as f:
            self.ppg = pickle.load(f)
        with open(data_path + 'mesa/'+mode+'_std.pkl', 'rb') as f:
            self.std = pickle.load(f)
        with open(data_path + 'mesa/'+mode+'_ref.pkl', 'rb') as f:
            self.ref = pickle.load(f)
        mode = mode + '_all'
        with open(data_path + 'mesa/'+mode+'_label.pkl', 'rb') as f:
            self.test_label = pickle.load(f)

        self.mask = torch.ones(32, dtype=torch.bool)
        self.mask[[22, 30]] = False

        label = list()
        for i in self.ref:
            ref = self.ref[i]
            
            ppg_temp = self.ppg[ref]
            ppg_temp = (ppg_temp - self.std[2][ref][0:1])/self.std[2][ref][1:2]
            ppg_temp = torch.tensor( resample(ppg_temp, int(ppg_temp.shape[0]/64/30*1024)) )
            self.ppg[ref] = self.adjust_length_1D(ppg_temp, 1200 * 1024)

            sig_temp = self.sig[ref]
            sig_temp = (sig_temp - self.std[1][ref][:,0:1])/self.std[1][ref][:,1:2]
            sig_temp = torch.tensor( resample(sig_temp, int(sig_temp.shape[1]/120*128), axis = 1) )
            sig_temp = self.adjust_length_2D(sig_temp, 600 * 128)
            sig_list = list()
            for j in range(0, sig_temp.size(1), 64):
                beg = j - 96
                end = j + 160
                beg = max(beg, 0)
                temp = sig_temp[:, beg: end]
                temp = self.adjust_length_2D(temp, 256)
                sig_list.append(temp.unsqueeze(1))
            self.sig[ref] = torch.concat(sig_list, axis = 1)

            fea_temp = self.fea[ref]
            fea_temp = (fea_temp - self.std[0][ref][:,0:1])/self.std[0][ref][:,1:2]
            filt = torch.zeros( fea_temp.shape[1] )
            for j in range(0, fea_temp.shape[1], 3):
                filt[j] = 1
            fea_temp = fea_temp[:, filt.bool()]
            fea_temp = self.adjust_length_2D(fea_temp, 1200)
            self.fea[ref] = fea_temp
            
            label.append( torch.tensor(self.test_label[self.test_label[:, 7] == i][:, 6].astype(float)) )
        self.test_label = label

        self.len = len(self.ppg)#2 * length#label.shape[0]

    def __getitem__(self, index):
        ppg = self.ppg[index]
        sig = self.sig[index]
        fea = self.fea[index][self.mask]
        label = self.test_label[index][:1200]
        
        return ppg.float(), sig.float(), fea.float(), label.float()
        
    def __len__(self):
        
        return self.len
        
    def adjust_length_1D(self, tensor, target_length = 1200 * 1024):
        
        current_length = tensor.size(0)
    
        if current_length < target_length:
            # 如果长度不足，补零
            padding = target_length - current_length
            adjusted_tensor = torch.nn.functional.pad(tensor, (0, padding))
        else:
            # 如果长度超过，截断
            adjusted_tensor = tensor[:target_length]
    
        return adjusted_tensor
    def adjust_length_2D(self, tensor, target_length = 1200 * 1024):
        
        current_length = tensor.size(1)
    
        if current_length < target_length:
            # 如果长度不足，补零
            padding = target_length - current_length
            adjusted_tensor = torch.nn.functional.pad(tensor, (0, padding, 0, 0))
        else:
            # 如果长度超过，截断
            adjusted_tensor = tensor[:, :target_length]
    
        return adjusted_tensor
'''