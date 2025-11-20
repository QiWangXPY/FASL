import os
import tqdm
import pickle
import numpy as np
import torch

mode = 'test'
stage_reverse = {5:3, 4:2, 3:2, 2:1, 1:1, 0:0}
already = np.unique([i[-7:-3] for i in os.listdir("/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/"+mode+"/fea_32/")])
test_ref = dict(zip(already, np.array(range(len(already)))))
test_fearef = list()
test_sigref = list()
test_ppgref = list()
test_label = list()

for i in tqdm.tqdm(already):
    
    test_fearef.append(torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+mode+'/fea_32/x_'+i+'.pt'))
    test_sigref.append(torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+mode+'/sig_4/x_'+i+'.pt'))
    test_ppgref.append(torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+mode+'/ppg/x_'+i+'.pt'))
    
    y_raw = torch.load('/extern2/zgz/wq/sleep/sleep_stage_ppg/clean_mesa_300s/'+mode+'/raw/y_'+i+'.pt')
    label = y_raw[:, 0].tolist()
    label = [stage_reverse[j] for j in label]

    
    count_list = list()
    for j in range(len(label)-9):
        temp = list()
        temp.append(label[j:j+10].count(0))
        temp.append(label[j:j+10].count(1))
        temp.append(label[j:j+10].count(2))
        temp.append(label[j:j+10].count(3))
        count_list.append(temp)
    count_list = np.array(count_list)

    beg_list = list()
    for j in range(len(label)-9):
        beg_item = list()
        beg_item.append(j * 30 * 64)
        beg_item.append((j+10) * 30 * 64)
        beg_item.append(j * 30)
        beg_item.append((j+10) * 30)
        beg_item.append(3 * j)
        beg_item.append(3 * (j+10)-2)
        beg_item.append(count_list[j].argmax())
        beg_item.append(i)
        
        beg_list.append(beg_item)

    test_label.append(np.array(beg_list)[count_list.max(1)>=8])
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

with open('mesa/'+mode+'_fea.pkl', 'wb') as f:
    pickle.dump(test_fearef, f)
with open('mesa/'+mode+'_sig.pkl', 'wb') as f:
    pickle.dump(test_sigref, f)
with open('mesa/'+mode+'_ppg.pkl', 'wb') as f:
    pickle.dump(test_ppgref, f)
with open('mesa/'+mode+'_ref.pkl', 'wb') as f:
    pickle.dump(test_ref, f)
with open('mesa/'+mode+'_label.pkl', 'wb') as f:
    pickle.dump(test_label, f)
'''with open('mesa/test_fea.pkl', 'rb') as f:
    test_fearef = pickle.load(f)'''
