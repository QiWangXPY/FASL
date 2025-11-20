import os
import tqdm
import pickle
import numpy as np
import torch

mode = 'train'
stage_reverse = {5:3, 4:2, 3:2, 2:1, 1:1, 0:0}
already = np.unique([i[-7:-3] for i in os.listdir("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/"+mode+"/fea_32/")])
test_ref = dict(zip(already, np.array(range(len(already)))))
test_fearef = list()
test_sigref = list()
test_ppgref = list()
test_label = list()

with open('mesa/'+mode+'_ref.pkl', 'rb') as f:
    ref = pickle.load(f)

for i in tqdm.tqdm(ref):
    
    test_fearef.append(torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+mode+'/fea_32/x_'+i+'.pt'))
    

'''
ref  :  mode_ref  id:index
label:  mode_label
ppg  :  test_ppgref[ind][int(test_label[i, 0]) : int(test_label[i, 1])].shape[0])
sig  :  test_sigref[ind][:,int(test_label[i, 2]) : int(test_label[i, 3])].shape[1])
fea  :  test_fearef[ind][:,int(test_label[i, 4]) : int(test_label[i, 5])].shape[1])

ind = test_ref[test_label[i, 7]]
ground_truth = test_ref[test_label[i, 6]]
'''

with open('mesa/'+mode+'_fea.pkl', 'wb') as f:
    pickle.dump(test_fearef, f)
