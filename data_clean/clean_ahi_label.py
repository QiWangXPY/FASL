import os
import re
import tqdm
import pickle
import numpy as np
import torch
from label_ahi import getLabels

mode = 'train'

ahi_path = 'data/mesa/polysomnography/annotations-events-profusion/'

already = np.unique([i[-7:-3] for i in os.listdir("../"+mode+"/fea_32/")]) 

label_index = os.listdir('data/mesa/polysomnography/annotations-events-profusion/')
resu = np.unique([re.findall(r'\d+', i)[0] for i in label_index])
already = np.intersect1d(resu, already)

test_label = list()

for i in tqdm.tqdm(already):
    y_raw = torch.load('../'+mode+'/raw/y_'+i+'.pt')
    label = y_raw[:, 0].tolist()
    
    ahi_label = getLabels(ahi_path + 'mesa-sleep-'+i+'-profusion.xml')
    
    beg_list = list()
    for j in range(len(label)-9):
        beg_item = list()
        beg_item.append(j * 30 * 64)
        beg_item.append((j+10) * 30 * 64)
        beg_item.append(j * 30)
        beg_item.append((j+10) * 30)
        beg_item.append(3 * j)
        beg_item.append(3 * (j+10)-2)
        
        if(np.any(ahi_label['ahi_labels'][y_raw[j, 1]+120:y_raw[j, 1]+150])):
            beg_item.append(1)
        else:
            beg_item.append(0)
        
        beg_item.append(i)
        
        beg_list.append(beg_item)


    test_label.append(np.array(beg_list))#[count_list.max(1)>=8])
test_label = np.concatenate(test_label, axis = 0)

'''
ref  :  mode_ref  id:index
label:  mode_label
ppg  :  test_ppgref[ind][int(test_label[i, 0]) : int(test_label[i, 1])].shape[0])
sig  :  test_sigref[ind][:,int(test_label[i, 2]) : int(test_label[i, 3])].shape[1])
fea  :  test_fearef[ind][:,int(test_label[i, 4]) : int(test_label[i, 5])].shape[1])

ind = test_ref[test_label[i, 7]]
ground_truth = test_ref[test_label[i, 6]]
'''

with open('mesa/'+mode+'_ahi_label.pkl', 'wb') as f:
    pickle.dump(test_label, f)
