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

from model import DenseNet, PPGNet
from utilities.tracker_utils import ClassificationTracker
from utilities.utils import setup_seed, Logger, print_args
from scipy.special import softmax
from tqdm import tqdm

from baseline_model_10h import PPG_10h, HeartRate_10h, HRV_10h, Hypnos_10h, Hypnos_10h_ablation, OurConLoss, ConFusionLoss, FourSignal_10h, Feature_10h, ALL_10h, OurPPG_10h

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

data_path = '../da300s/'

mode = 'all'
from scipy.signal import resample
import torch
import pickle
from torch.utils.data import Dataset

class Load_10h_Dataset(Dataset):
    def __init__(self, mode):
        super(Load_10h_Dataset, self).__init__()
        
        data_path = '../da300s/'
        
        self.mode = mode
        mode = mode + '_all'
        
        #with open(data_path + 'mesa/10h_input/'+mode+'_fea.pkl', 'rb') as f:
        #    self.fea = pickle.load(f)
        if(self.mode == 'train'):
            with open(data_path + 'mesa/10h_input/'+self.mode+'_fea_conseq_500.pkl', 'rb') as f:
                self.fea = pickle.load(f)
            #with open(data_path + 'mesa/10h_input/'+mode+'_sig.pkl', 'rb') as f:
            #    self.sig = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+self.mode+'_sig_conseq_500.pkl', 'rb') as f:#_500
                self.sig = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+mode+'_ppg_500.pkl', 'rb') as f:
                self.ppg = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+mode+'_label_frag_500.pkl', 'rb') as f:
                self.test_label = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+self.mode+'_ahi_label_500.pkl', 'rb') as f:
                self.ahi_label = pickle.load(f)
        else:
            with open(data_path + 'mesa/10h_input/'+self.mode+'_fea_conseq.pkl', 'rb') as f:
                self.fea = pickle.load(f)
            #with open(data_path + 'mesa/10h_input/'+mode+'_sig.pkl', 'rb') as f:
            #    self.sig = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+self.mode+'_sig_conseq.pkl', 'rb') as f:
                self.sig = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+mode+'_ppg.pkl', 'rb') as f:
                self.ppg = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+mode+'_label_frag.pkl', 'rb') as f:
                self.test_label = pickle.load(f)
            with open(data_path + 'mesa/10h_input/'+self.mode+'_ahi_label.pkl', 'rb') as f:
                self.ahi_label = pickle.load(f)

        '''self.mask = torch.ones(352, dtype=torch.bool)
        self.mask[66 : 77] = False
        self.mask[154 : 165] = False
        self.mask[242 : 253] = False
        self.mask[330 : 341] = False'''
        self.mask = torch.ones(32, dtype=torch.bool)
        self.mask[[6, 14, 22, 30]] = False
        
        del_list = list()
        for i in range(len(self.ahi_label)):
            if (self.ahi_label[i].shape[0])==0:
                del_list.append(i)
        
        for index in sorted(del_list, reverse=True):
            del self.ahi_label[index]
            del self.test_label[index]
            del self.fea[index]
            del self.sig[index]
            del self.ppg[index]
        
        self.train_len = len( self.fea )
        
        if(self.mode == 'train'):
            with open(data_path + 'mesa/10h_input/train_fea_conseq_1354.pkl', 'rb') as f:#1354
                self.fea = self.fea + pickle.load(f)
            with open(data_path + 'mesa/10h_input/train_sig_conseq_1354.pkl', 'rb') as f:
                self.sig = self.sig + pickle.load(f)
            with open(data_path + 'mesa/10h_input/train_all_ppg_1354.pkl', 'rb') as f:
                self.ppg = self.ppg + pickle.load(f)
            with open(data_path + 'mesa/10h_input/train_all_label_frag_1354.pkl', 'rb') as f:
                self.test_label = self.test_label + pickle.load(f)
            with open(data_path + 'mesa/10h_input/train_ahi_label_1354.pkl', 'rb') as f:
                self.ahi_label = self.ahi_label + pickle.load(f)

        del_list = list()
        for i in range(len(self.ahi_label)):
            if (self.ahi_label[i].shape[0])==0:
                del_list.append(i)
        
        for index in sorted(del_list, reverse=True):
            del self.ahi_label[index]
            del self.test_label[index]
            del self.fea[index]
            del self.sig[index]
            del self.ppg[index]

        self.len = len(self.ppg)#2 * length#label.shape[0]
        
    def __getitem__(self, index):
        ppg = self.ppg[index].unsqueeze(0)
        sig = self.sig[index]
        fea = self.fea[index][self.mask]
        label = self.test_label[index][0, :1200]
        
        #frag_label = self.test_label[index][1, :1200]
        
        frag_label = torch.tensor(0)
        if(self.mode == 'train' and index >= self.train_len):
            frag_label = torch.tensor(1)
        
        ahi_label = self.ahi_label[index][:1200]
        
        return ppg.float(), sig.float(), fea.float(), label.long(), frag_label.long(), ahi_label.long()
        
    def __len__(self):
        
        return self.len
    
def load_all():
    print("test labeled data:")
    test_loader = torch.utils.data.DataLoader(dataset=Load_10h_Dataset('test'), batch_size=1,
                                               shuffle=True, pin_memory=True,# drop_last=True,
                                               num_workers=0)
    print("train labeled data:")
    train_loader = torch.utils.data.DataLoader(dataset=Load_10h_Dataset('train'), batch_size=1,#train
                                               shuffle=True, pin_memory=True,#drop_last=True,
                                               num_workers=0)
    #train_loader = torch.utils.data.DataLoader(dataset=Loadone_Dataset('test'), batch_size=512,
    
    val_loader=test_loader
    return val_loader, test_loader, train_loader

#mesa_hrfea, mesa_acc, mesa_label
#RR * 8 am 3 ba 3
def main(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    class_counts = torch.tensor([4, 10, 1.5, 2.6])#label unbalence
    total_samples = class_counts.sum()
    weights = total_samples / class_counts  # 计算权重
    weights = weights / weights.sum() 
    criterion = nn.CrossEntropyLoss(weight=weights.float().to(device))
    con_criterion = OurConLoss()

    #class_counts = torch.tensor([4 * 7, 10 * 7, 1.5 * 7, 2.6 * 7, 4 * 2, 10 * 2, 1.5 * 2, 2.6 * 2])#label unbalence
    class_counts = torch.tensor([28020, 61441,  9535, 12901,  9637, 12615,  4427,  2732,   765, 19741, 476,  7820,  1459,  6680,   443,  3232])#
    total_samples = class_counts.sum()
    weights = total_samples / class_counts  # 计算权重
    weights = weights / weights.sum() 
    ei_criterion = nn.CrossEntropyLoss(weight=weights.float().to(device))
    
    class_counts = torch.tensor([14, 4])#label unbalence
    total_samples = class_counts.sum()
    weights = total_samples / class_counts  # 计算权重
    weights = weights / weights.sum() 
    frag_cri = nn.CrossEntropyLoss(weight=weights.float().to(device))

    sfmax = nn.Softmax(dim = 2)
    
    ahi_cri = nn.CrossEntropyLoss(weight=torch.tensor([0.22, 0.78]).float().to(device))
    log_root = os.path.join(r"tfboard", args.dataset, args.nn_type)
    tracker = ClassificationTracker(args, tensorboard_dir=log_root, master_kpi_path="./exp_results.csv")
    sys.stdout = Logger(tracker.tensorboard_path)

    print("Model: %s" % args.nn_type)
    print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
    print_args(args)

    tracker.copy_main_run_file(os.path.join(os.path.abspath(os.getcwd()), os.path.basename(__file__)))
    tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "models"))
    test_fold_gt = []  # this list can be used for hold out and CV
    test_fold_pred = []
    test_fold_prob = []
    test_fold_feature = []
    test_fold_idx = []
    for fold_num in np.arange(1):
        if mode == 'ppg':
            #model = Hypnos_10h()
            
            #model = OurPPG_10h()
            model = PPG_10h()
            #model = Hypnos_10h_ablation()
            #summary(model, (1, 64 * 300))  # print the model summary
        if mode == 'sig':
            #model = HeartRate_10h()
            model = FourSignal_10h()
            model.sig_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_sig_withoutseq/saved_models/fold_0_epoch_39.pth').sig_en.state_dict() )
            #summary(model, (4, 300))  # print the model summary
        if mode == 'fea':
            #model = HRV_10h()
            model = Feature_10h()
            model.fea_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_fea_withoutseq/saved_models/fold_0_epoch_34.pth').fea_en.state_dict() )
        if mode == 'all':
            model = ALL_10h()
            model = torch.load('tfboard/mesa/ResHybridAtt/our_10h_all_cnn_500_twoclasslayer/saved_models/fold_0_epoch_28.pth')
            
        if torch.cuda.is_available():
            model.cuda()
            #classifier.cuda()
        optims = {"SGD": torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
                  "ADAM": torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.01)
                  }
        optimizer = optims[args.optim]

        val_loader, test_loader, train_loader = load_all()

        print("Total training samples: %s" % train_loader.dataset.__len__())
        print("Total testing samples: %s" % test_loader.dataset.__len__())
        tracker.reset_best_eval_metrics_models()
        for epoch in range(args.epochs):
            # ***************** Training ************************
            # control the accumulated samples that added to gt and pred to calculate metrics at the last batch
            first_train_epoch = True
            model.train()

            for batch_idx, (ppg, sig, fea, label, frag_label, ahi_label) in enumerate(train_loader):
                
                if batch_idx > 0 and args.debug == 1:
                    continue

                y_frag = frag_label.squeeze(0)
                y = label.squeeze(0)
                ahi_label = ahi_label.squeeze(0)
                
                y = y #+ y_frag * 4 + ahi_label * 8

                y = y.long().to(device)
                
                if mode == 'ppg':
                    '''
                    con_ppg = torch.concat([ppg, ppg], axis = 0).to(device)
                    
                    outputs = model(con_ppg)
                    
                    con_feature = outputs[0]
                    train_loss = con_criterion(con_feature.permute(1, 0, 2))
                                        
                    ppg = con_ppg[0:1]
                    ahi_outputs = outputs[2][0:1]
                    outputs = (outputs[0][0:1], outputs[1][0:1])'''
                    ppg = ppg.to(device)
                    outputs = model(ppg)
                if mode == 'sig':
                    #sig = sig[0, 0]
                    sig = sig.to(device)
                    outputs = model(sig)
                    
                if mode == 'fea':
                    #fea = fea.permute([0, 2, 1])
                    fea = fea.to(device)
                    outputs = model(fea)

                if mode == 'all':
                    ppg = ppg.to(device)
                    sig = sig.to(device)
                    fea = fea.to(device)

                    '''
                    mask = torch.zeros([1, 1200, 1200])
                    mask[:, :, 960:] = 1
                    mask = mask[:, :, torch.randperm(1200)]
                    mask[:, :, y.shape[0]:] = 1
                    mask = mask == 1  
                    mask = mask.to(device)
                    '''
                    outputs = model(ppg, sig, fea)

                    #con_feature = outputs[0]
                    #train_loss = con_criterion(con_feature.permute(1, 0, 2)[:y.shape[0], :, :])
                
                #y_ahi = ahi_label.squeeze(0).to(device)
                # Forward pass
                
                if type(outputs) in (tuple, list):
                    
                    feature, outputs = outputs[0], outputs[1]# frag , outputs[2]

                    #frag = frag[0, :y.shape[0], :]
                else:
                    feature = outputs

                feature = feature[0, :y.shape[0], :]
                outputs = outputs[0, :y.shape[0], :]
                
                if(frag_label==1):
                    #value, predicted = torch.max(outputs[:, :].data, dim=1)
                    posi_list = list()
                    for i in range(4):
                        posi_list.append(list())

                    n = len(outputs)
                    low_k = int(n * 0.5)
                    entropy_i = - torch.sum(outputs * torch.log(outputs + 1e-10), dim=1)
                    low_k =  torch.kthvalue(entropy_i, low_k ).values
                    
                    prob_tea_i = outputs[entropy_i < low_k].clone()
                    prob_fea = feature[entropy_i < low_k]
                    
                    entropy_i = - torch.sum(prob_tea_i * torch.log(prob_tea_i + 1e-10), dim=1)
                    entropy_i = (entropy_i-entropy_i.min())/(entropy_i.max()-entropy_i.min())
                    for i in range(4):
                        for j in range(len(prob_tea_i)):
                            posi_list[i].append(prob_tea_i[j][i]-entropy_i[j])
                    posi_list = torch.clamp(torch.stack([torch.stack(i) for i in posi_list]), min=0).detach()

                    con_fea = torch.zeros([4, torch.sum(posi_list>0, axis = 1).min(), feature.shape[1]])
                    con_param = torch.zeros([4, torch.sum(posi_list>0, axis = 1).min()])

                    #train_loss = criterion( outputs[entropy_i < low_k, :], predicted[entropy_i < low_k] )
                    train_loss = 0
                    
                    if(con_fea.shape[1] != 0):
                        
                        #unsupervised loss
                        value, predicted = torch.max(prob_tea_i[:, :].data, dim=1)
    
                        #contract loss
                        for i in range(4):    
                            #lab_mean = (torch.sum(posi_list[i].unsqueeze(1) * feature, axis = 0)/torch.sum(posi_list[i])).detach()
                            #temp = (posi_list[i].unsqueeze(1) * feature + (1 - posi_list[i]).unsqueeze(1) * lab_mean)[posi_list[i]>0]

                            temp_fea = prob_fea[posi_list[i]>0]
                            temp_param = posi_list[i][posi_list[i]>0]
                            #temp = outputs[posi_list[i]>0]
                            
                            indices = torch.randperm(len(temp_param))[:con_fea.shape[1]]
                            con_fea[i] = temp_fea[indices]
                            con_param[i] = temp_param[indices]
                            
                        train_loss = train_loss + con_criterion(con_fea, con_param)/2
                if(frag_label==0):
                    train_loss = criterion( outputs[:, :], y ) #+ frag_cri( frag,  y_frag) #+ train_loss#+ criterion( ahi_outputs[:, :2], y_ahi )

                if(train_loss !=0 and not torch.isnan(train_loss)):
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                else:
                    continue

                _, predicted = torch.max(outputs[:, :].data, dim=1)
                
                #y = y % 4
                #predicted = predicted % 4
                
                if first_train_epoch:
                    epoch_gt = copy(y)
                    #epoch_gt = copy(y_ahi)
                    epoch_pred = copy(predicted)
                else:
                    epoch_gt = torch.cat([epoch_gt, y])
                    #epoch_gt = torch.cat([epoch_gt, y_ahi])
                    epoch_pred = torch.cat([epoch_pred, predicted])

                if batch_idx % args.log_interval == 0:
                    tracker.log_train_fold_epoch(epoch_gt.cpu().numpy(), epoch_pred.cpu().numpy(),
                                                 {'xent': train_loss.item()}, fold_num, len(train_loader), epoch,
                                                 batch_idx)
                first_train_epoch = False
            
            # ************** validation ******************
            print("testing start...")
            first_val_epoch = True
            num_val_samples = 0
            total_val_loss = 0
            val_fc3 = []
            model.eval()
            
            #classifier.eval()
            
            with torch.no_grad():
                for batch_idx, (ppg, sig, fea, label, frag_label, ahi_label) in enumerate(val_loader):
                    
                    if batch_idx > 0 and args.debug == 1:
                        continue

                    if mode == 'ppg':
                        ppg = ppg.to(device)
                        y_outputs = model(ppg)
    
                    if mode == 'sig':
                        #sig = sig[0, 0]
                        sig = sig.to(device)
                        y_outputs = model(sig)
                        
                    if mode == 'fea':
                        #fea = fea.permute([0, 2, 1])
                        fea = fea.to(device)
                        y_outputs = model(fea)

                    if mode == 'all':
                        ppg = ppg.to(device)
                        sig = sig.to(device)
                        fea = fea.to(device)
                        y_outputs = model(ppg, sig, fea)
                    
                    frag_label = frag_label.squeeze(0)
                    y = label.squeeze(0)
                    ahi_label = ahi_label.squeeze(0)
                    
                    y = y #+ frag_label * 4 + ahi_label * 8
    
                    y = y.long().to(device)

                    # Forward pass
                    
                    if type(y_outputs) in (tuple, list):
                        val_feature, y_val_pred = y_outputs[0], y_outputs[1]#y_frag_prob , y_outputs[2]
                    else:
                        y_val_pred = y_outputs
                        val_feature = y_outputs
                    
                    val_feature = val_feature[0, :y.shape[0], :]
                    y_val_pred = y_val_pred[0, :y.shape[0], :]
                    #y_val_pred = y_val_pred[:, 0, :]
                    #y_val_pred = y_val_pred[:, -1, :4]
                    #y_val_pred = y_val_pred[:, 4:]
                    
                    #y_val_pred = classifier( val_feature )[:, :4]
                    
                    if batch_idx < 5:
                        val_fc3.append(val_feature)
                    
                    val_loss = criterion(y_val_pred[:, :], y)
                    #val_loss = criterion(y_val_pred, y_ahi)
                    
                    total_val_loss += val_loss
                    _, y_val_pred = torch.max(y_val_pred[:, :].data, dim=1)

                    #y = y % 4
                    #y_val_pred = y_val_pred % 4
                    
                    num_val_samples += y.nelement()
                    if first_val_epoch:
                        val_epoch_gt = copy(y)
                        #val_epoch_gt = copy(y_ahi)
                        val_epoch_pred = copy(y_val_pred)
                    else:
                        val_epoch_gt = torch.cat([val_epoch_gt, y])
                        #val_epoch_gt = torch.cat([val_epoch_gt, y_ahi])
                        val_epoch_pred = torch.cat([val_epoch_pred, y_val_pred])
                    first_val_epoch = False
                mean_val_loss = total_val_loss / num_val_samples
                val_fc3 = torch.cat(val_fc3, dim=0).cpu()
                tracker.log_eval_fold_epoch(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                            {'mean_xent': mean_val_loss.cpu().numpy()}, fold_num, epoch, model)#classifier)
                if args.save_eval == 1:
                    tracker.save_test_analysis_visualisation_results(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                                                     val_fc3.cpu().numpy(), epoch, 'eval', fold_num=fold_num)
        # ************** test ******************
        # load the best
        print("testing start...")
        first_test_epoch = True
        num_test_samples = 0
        correct_test = 0
        total_test_loss = 0
        test_fc3_feature = []
        #test_idx_epoch_list = []
        # load the best validation model
        model = tracker.load_best_eval_model(model)
        model.eval()
        
        #classifier.eval()
        with torch.no_grad():
            for batch_idx, (ppg, sig, fea, label, frag_label, ahi_label) in enumerate(test_loader):
                
                if mode == 'ppg':
                    ppg = ppg.to(device)
                    y_outputs = model(ppg)

                if mode == 'sig':
                    #sig = sig[0, 0]
                    sig = sig.to(device)
                    y_outputs = model(sig)
                    
                if mode == 'fea':
                    #fea = fea.permute([0, 2, 1])
                    fea = fea.to(device)
                    y_outputs = model(fea)
                
                if mode == 'all':
                    ppg = ppg.to(device)
                    sig = sig.to(device)
                    fea = fea.to(device)
                    y_outputs = model(ppg, sig, fea)
                
                frag_label = frag_label.squeeze(0)
                y = label.squeeze(0)
                ahi_label = ahi_label.squeeze(0)
                
                y = y# + frag_label * 4 + ahi_label * 8

                y = y.long().to(device)

                # Forward pass
                if type(y_outputs) in (tuple, list):
                    test_feature, y_test_prob = y_outputs[0], y_outputs[1]#y_frag_prob , y_outputs[2]
                else:
                    y_test_prob = y_outputs
                    test_feature = y_outputs

                test_feature = test_feature[0, :y.shape[0], :]
                y_test_prob = y_test_prob[0, :y.shape[0], :]
                
                if batch_idx < 5:
                    test_fc3_feature.append(test_feature)
                
                test_loss = criterion(y_test_prob[:, :], y)
                #test_loss = criterion(y_test_prob, y_ahi)
                
                total_test_loss += test_loss
                _, y_test_pred = torch.max(y_test_prob[:, :].data, dim=1)

                #y = y % 4
                #y_test_pred = y_test_pred % 4
                
                num_test_samples += y.nelement()
                #num_test_samples += y_ahi.nelement()
                
                correct_test += y_test_pred.eq(y.data).sum().item()
                if first_test_epoch:
                    test_epoch_gt = copy(y)
                    #test_epoch_gt = copy(y_ahi)
                    
                    test_epoch_pred = copy(y_test_pred)
                    test_epoch_prob = copy(y_test_prob[:, :4])
                else:
                    test_epoch_gt = torch.cat([test_epoch_gt, y])
                    #test_epoch_gt = torch.cat([test_epoch_gt, y_ahi])
                    
                    test_epoch_pred = torch.cat([test_epoch_pred, y_test_pred])
                    test_epoch_prob = torch.cat([test_epoch_prob, y_test_prob[:, :4]])
                first_test_epoch = False
                #test_idx_epoch_list.append(test_idx)
            mean_test_loss = total_test_loss / num_test_samples
            test_fc3_feature = torch.cat(test_fc3_feature, dim=0).cpu()
            #test_idx_epoch_list = torch.cat(test_idx_epoch_list, dim=0).cpu()
            tracker.log_test_fold_epoch(fold_num, tracker.best_eval_epoch_idx, test_epoch_gt.cpu().numpy(),
                                        test_epoch_pred.cpu().numpy(),
                                        {'mean_xent': mean_test_loss.cpu().numpy()})
            test_fold_feature.append(test_fc3_feature.cpu().numpy())
            test_fold_gt.append(np.expand_dims(test_epoch_gt.cpu().numpy(), axis=1))
            test_fold_pred.append(np.expand_dims(test_epoch_pred.cpu().numpy(), axis=1))
            test_fold_prob.append(test_epoch_prob.cpu().numpy())
            #test_fold_idx.append(np.expand_dims(test_idx_epoch_list.cpu().numpy(), axis=1))
    #test_fold_idx = np.vstack(test_fold_idx).squeeze()
    test_fold_gt = np.vstack(test_fold_gt).squeeze()
    test_fold_pred = np.vstack(test_fold_pred).squeeze()
    test_fold_feature = np.vstack(test_fold_feature)
    test_fold_prob = softmax(np.vstack(test_fold_prob), axis=1)  # softmax over
    tracker.save_test_analysis_visualisation_results(test_fold_gt, test_fold_pred,
                                                         test_fold_feature, tracker.best_eval_epoch_idx, 'test')
    print("Finished!")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # specialised parameters
    parser.add_argument('--nn_type', type=str, default="ResHybridAtt",
                        help='define the neural network type')
    parser.add_argument('--att_on_modality', type=str, default="act", help="act, car, none")
    # general parameters for all models
    parser.add_argument('--optim', type=str, default="ADAM", help='optimisation')#ADAM
    parser.add_argument('--log_interval', type=int, default=100, help='interval to log metrics')
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes or labels')#4
    parser.add_argument('--momentum', type=float, default=0.9, help='opt momentum')
    parser.add_argument('--epochs', type=int, default=60, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug model')
    parser.add_argument('--dataset', type=str, default="mesa", help="apple, mesa, mesa_hr_statistic")
    parser.add_argument('--feature_type', type=str, default="all", help="all, hrv, hr, and so on")
    parser.add_argument('--seq_len', type=int, default=20, help="100, 50, 20 corresponding to 101, 51, and 21")
    parser.add_argument('--comments', type=str, default="", help="comments to append")
    parser.add_argument('--save_eval', type=int, default=0, help="not save the eval results")
    parser.add_argument('--seed', type=int, default=42, help="fix seed")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
