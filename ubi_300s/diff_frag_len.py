import argparse

from copy import copy

from torch.utils.data import Dataset
import torch
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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

data_path = '../da300s/'

mode = 'ppg'
class Loadone_Dataset(Dataset):
    def __init__(self, mode):
        super(Loadone_Dataset, self).__init__()
        
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

        mode = mode + '_exc'
        with open(data_path + 'mesa/'+mode+'_label.pkl', 'rb') as f:
            test_label = pickle.load(f)

        test_label = np.c_[test_label, np.zeros([len(test_label),1], dtype=int)]
        
        '''
        for i in range(test_label.shape[0]-1):
            if((test_label[i,7] == test_label[i+1,7]) and (test_label[i,6] != test_label[i+1,6])):
                test_label[i, 8] = 1
                test_label[i+1, 8] = 1
                
                test_label[i-1, 8] = 1
                test_label[i+2, 8] = 1a
        
                
                test_label[i-2, 8] = 1
                test_label[i+3, 8] = 1
                test_label[i-3, 8] = 1
                test_label[i+4, 8] = 1
        '''
        index = dict()
        
        for i in range(4):
            index[i] = list()
        for i in range(len(test_label)):
            index[int(test_label[i, 6])].append(test_label[i])
        length = len(index[0])
        for i in range(4):
            index[i] = np.array(index[i])
            length = min(length, len(index[i]))
        
        self.index = index

        self.mask = torch.ones(32, dtype=torch.bool)
        self.mask[[22, 30]] = False
        
        self.len = 8 * length#label.shape[0]

    def __getitem__(self, index):
        sleep_cla = index % 4
        
        while True:
            id = torch.randint(len(self.index[sleep_cla]), size=(1,))[0]
            ref = self.ref[self.index[sleep_cla][id][7]]
            
            ppg = self.ppg[ref][int(self.index[sleep_cla][id][0]): int(self.index[sleep_cla][id][1])]
            sig = self.sig[ref][:, int(self.index[sleep_cla][id][2]): int(self.index[sleep_cla][id][3])]
            fea = self.fea[ref][:, int(self.index[sleep_cla][id][4]): int(self.index[sleep_cla][id][5])]

            ppg = (ppg - self.std[2][ref][0:1])/self.std[2][ref][1:2]
            sig = (sig - self.std[1][ref][:,0:1])/self.std[1][ref][:,1:2]
            fea = (fea - self.std[0][ref][:,0:1])/self.std[0][ref][:,1:2]

            fea = fea[self.mask]

            ppg = ppg.unsqueeze(0)

            ppg_bef = self.ppg[ref][int(self.index[sleep_cla][id][0])-38400: int(self.index[sleep_cla][id][1])-38400]
            sig_bef = self.sig[ref][:, int(self.index[sleep_cla][id][2])-600: int(self.index[sleep_cla][id][3])-600]
            fea_bef = self.fea[ref][:, int(self.index[sleep_cla][id][4])-60: int(self.index[sleep_cla][id][5])-60]

            ppg_bef = (ppg_bef - self.std[2][ref][0:1])/self.std[2][ref][1:2]
            sig_bef = (sig_bef - self.std[1][ref][:,0:1])/self.std[1][ref][:,1:2]
            fea_bef = (fea_bef - self.std[0][ref][:,0:1])/self.std[0][ref][:,1:2]

            fea_bef = fea_bef[self.mask]

            ppg_bef = ppg_bef.unsqueeze(0)

            ppg_nex = self.ppg[ref][int(self.index[sleep_cla][id][0])+38400: int(self.index[sleep_cla][id][1])+38400]
            sig_nex = self.sig[ref][:, int(self.index[sleep_cla][id][2])+600: int(self.index[sleep_cla][id][3])+600]
            fea_nex = self.fea[ref][:, int(self.index[sleep_cla][id][4])+60: int(self.index[sleep_cla][id][5])+60]

            ppg_nex = (ppg_nex - self.std[2][ref][0:1])/self.std[2][ref][1:2]
            sig_nex = (sig_nex - self.std[1][ref][:,0:1])/self.std[1][ref][:,1:2]
            fea_nex = (fea_nex - self.std[0][ref][:,0:1])/self.std[0][ref][:,1:2]

            fea_nex = fea_nex[self.mask]

            ppg_nex = ppg_nex.unsqueeze(0)



            
            if(ppg.isnan().any() or sig.isnan().any() or fea.isnan().any()):
                continue
            if(ppg_bef.isnan().any() or sig_bef.isnan().any() or fea_bef.isnan().any()):
                continue
            if(ppg_nex.isnan().any() or sig_nex.isnan().any() or fea_nex.isnan().any()):
                continue
            if(ppg_bef.shape[2] != 19200 or sig_bef.shape[2] !=300 or fea_bef.shape[2] != 28):
                continue
            if(ppg_nex.shape[2] != 19200 or sig_nex.shape[2] !=300 or fea_nex.shape[2] != 28):
                continue
            else:
                break
                
        ppg = torch.concat([ppg_bef.unsqueeze(0), ppg.unsqueeze(0), ppg_nex.unsqueeze(0)], axis = 0)
        sig = torch.concat([sig_bef.unsqueeze(0), sig.unsqueeze(0), sig_nex.unsqueeze(0)], axis = 0)
        fea = torch.concat([fea_bef.unsqueeze(0), fea.unsqueeze(0), fea_nex.unsqueeze(0)], axis = 0)
        
        return ppg.float(), sig.float(), fea.float(), torch.tensor(sleep_cla)
    def __len__(self):
        return self.len
'''
class Loadone_Dataset(Dataset):

    def __init__(self, mode):
        super(Loadone_Dataset, self).__init__()

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
        
        #if mode == 'train':
        #    mode += '_all'
        #else:
        #    mode += '_all'
        #with open(data_path + 'mesa/'+mode+'_label.pkl', 'rb') as f:
        #    test_label = pickle.load(f)
        mode1 = mode + '_all'
        mode2 = mode + '_ahi'
        with open(data_path + 'mesa/'+mode1+'_label.pkl', 'rb') as f:
            test_label1 = pickle.load(f)
        with open(data_path + 'mesa/'+mode2+'_label.pkl', 'rb') as f:
            test_label2 = pickle.load(f)

        df1 = pd.DataFrame(test_label1)
        df2 = pd.DataFrame(test_label2)
        merged_df = pd.merge(df1, df2, on=[0,1,2,3,4,5,7], how='inner')
        test_label = np.array(merged_df)
        
        index = dict()
        
        for i in range(4):
            index[i] = list()
        for i in range(len(test_label)):
            index[int(test_label[i, 6])].append(test_label[i])
        length = len(index[0])
        for i in range(4):
            index[i] = np.array(index[i])
            length = min(length, len(index[i]))
        
        self.index = index

        self.mask = torch.ones(32, dtype=torch.bool)
        self.mask[[22, 30]] = False
        
        self.len = 8 * length#label.shape[0]

    def __getitem__(self, index):
        sleep_cla = index % 4
        
        while True:
            id = torch.randint(len(self.index[sleep_cla]), size=(1,))[0]
            ref = self.ref[self.index[sleep_cla][id][7]]
            
            ppg = self.ppg[ref][int(self.index[sleep_cla][id][0]): int(self.index[sleep_cla][id][1])]
            sig = self.sig[ref][:, int(self.index[sleep_cla][id][2]): int(self.index[sleep_cla][id][3])]
            fea = self.fea[ref][:, int(self.index[sleep_cla][id][4]): int(self.index[sleep_cla][id][5])]

            ppg = (ppg - self.std[2][ref][0:1])/self.std[2][ref][1:2]
            sig = (sig - self.std[1][ref][:,0:1])/self.std[1][ref][:,1:2]
            fea = (fea - self.std[0][ref][:,0:1])/self.std[0][ref][:,1:2]

            fea = fea[self.mask]

            ppg = ppg.unsqueeze(0)
            if(ppg.isnan().any() or sig.isnan().any() or fea.isnan().any()):
                continue
            else:
                break
            
        return ppg.float(), sig.float(), fea.float(), torch.tensor(sleep_cla), torch.tensor(int(self.index[sleep_cla][id][8]))
		

    def __len__(self):
        return self.len
'''

def load_all():
    print("test labeled data:")
    test_loader = torch.utils.data.DataLoader(dataset=Loadone_Dataset('test'), batch_size=512,
                                               shuffle=True, pin_memory=True,# drop_last=True,
                                               num_workers=0)
    print("train labeled data:")
    train_loader = torch.utils.data.DataLoader(dataset=Loadone_Dataset('train'), batch_size=512,
                                               shuffle=True, pin_memory=True,#drop_last=True,
                                               num_workers=0)
    val_loader=test_loader
    return val_loader, test_loader, train_loader

def ppg_model():
    # ppg 
    
    Hz = 128 // 2
    SEG_SEC = 300
    NUM_CLASSES = 6
    IN_CHAN = 1
    N_DENSE_BLOCK = 3
    
    model = DenseNet(
        input_size=Hz*SEG_SEC, in_channels=IN_CHAN,
        n_classes=NUM_CLASSES, skip_final_transition_blk=True,
        kernels=[
            [5] for _ in range(N_DENSE_BLOCK)
        ],
        layer_dilations=[
            [1] for _ in range(N_DENSE_BLOCK)
        ],
        channel_per_kernel=[
            #32 for _ in range(N_DENSE_BLOCK)
            32 for _ in range(N_DENSE_BLOCK)
        ],
        n_blocks=[
            2 for _ in range(N_DENSE_BLOCK)
        ],
        #low_conv_cfg=[32, 32, 'M'],
        low_conv_cfg=[16, 32, 'M'],
        #low_conv_kernels=[42, 21],
        low_conv_kernels=[21, 21, 21],
        #low_conv_strides=[5, 1],
        low_conv_strides=[5, 5, 1],
        low_conv_pooling_kernels=[2],
        transition_pooling=[2 for _ in range(N_DENSE_BLOCK)]
    )
    return model

def sig_model():
    Hz = 1
    SEG_SEC = 300
    NUM_CLASSES = 4
    IN_CHAN = 4
    N_DENSE_BLOCK = 4

    model = DenseNet(
        input_size=Hz*SEG_SEC, in_channels=IN_CHAN,
        n_classes=NUM_CLASSES, skip_final_transition_blk=True,
        kernels=[
            [5] for _ in range(N_DENSE_BLOCK)
        ],
        layer_dilations=[
            [1] for _ in range(N_DENSE_BLOCK)
        ],
        channel_per_kernel=[
            32 for _ in range(N_DENSE_BLOCK)
        ],
        n_blocks=[
            2 for _ in range(N_DENSE_BLOCK)
        ],
        #low_conv_cfg=[32, 32, 'M'],
        low_conv_cfg=[16, 4],# 'M'],
        #low_conv_kernels=[42, 21],
        low_conv_kernels=[21, 21],
        #low_conv_strides=[5, 1],
        low_conv_strides=[3, 1],
        low_conv_pooling_kernels=[2],
        transition_pooling=[2 for _ in range(N_DENSE_BLOCK)]
    )
    return model

def fea_model():
    Hz = 1
    SEG_SEC = 28
    NUM_CLASSES = 4
    IN_CHAN = 30
    N_DENSE_BLOCK = 3

    model = DenseNet(
        input_size=Hz*SEG_SEC, in_channels=IN_CHAN,
        n_classes=NUM_CLASSES, skip_final_transition_blk=True,
        kernels=[
            [5] for _ in range(N_DENSE_BLOCK)
        ],
        layer_dilations=[
            [1] for _ in range(N_DENSE_BLOCK)
        ],
        channel_per_kernel=[
            #32 for _ in range(N_DENSE_BLOCK)
            32 for _ in range(N_DENSE_BLOCK)
        ],
        n_blocks=[
            2 for _ in range(N_DENSE_BLOCK)
        ],
        #low_conv_cfg=[32, 32, 'M'],
        low_conv_cfg=[64, 30],# 'M'],
        #low_conv_kernels=[42, 21],
        low_conv_kernels=[21, 21],
        #low_conv_strides=[5, 1],
        low_conv_strides=[3, 1],
        low_conv_pooling_kernels=[2],
        transition_pooling=[2 for _ in range(N_DENSE_BLOCK)]
    )
    return model

#mesa_hrfea, mesa_acc, mesa_label
#RR * 8 am 3 ba 3
def main(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
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
            model = ppg_model()
            #model = PPGNet()
        if mode == 'sig':
            model = sig_model()
        if mode == 'fea':
            model = fea_model()
        if torch.cuda.is_available():
            model.cuda()
        optims = {"SGD": torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
                  "ADAM": torch.optim.Adam(model.parameters(), lr=args.lr)
                  }
        optimizer = optims[args.optim]
        
        if mode == 'ppg':
            summary(model, (1, 64 * 300))  # print the model summary
        if mode == 'sig':
            summary(model, (4, 300))  # print the model summary
        if mode == 'fea':
            summary(model, (30, 28))  # print the model summary
        
        val_loader, test_loader, train_loader = load_all()

        print("Total training samples: %s" % train_loader.dataset.__len__())
        print("Total testing samples: %s" % test_loader.dataset.__len__())
        tracker.reset_best_eval_metrics_models()
        for epoch in range(args.epochs):
            # ***************** Training ************************
            # control the accumulated samples that added to gt and pred to calculate metrics at the last batch
            first_train_epoch = True
            model.train()
            for batch_idx, (ppg, sig, fea, label, ahi) in enumerate(train_loader):
                
                if batch_idx > 0 and args.debug == 1:
                    continue
                ppg = ppg.to(device)
                sig = sig.to(device)
                fea = fea.to(device)
                y = label.to(device)
                y_ahi = ahi.to(device)
                                
                # Forward pass
                if mode == 'ppg':
                    outputs = model(ppg)
                if mode == 'sig':
                    outputs = model(sig)
                if mode == 'fea':
                    outputs = model(fea)
                
                
                if type(outputs) in (tuple, list):
                    feature, outputs = outputs[0], outputs[1]
                else:
                    feature = outputs
                
                '''train_loss = list()
                for i in range(outputs.shape[1]):
                    train_loss.append(criterion(outputs[:, i, :], y))
                '''
                # Backward and optimize
                train_loss = criterion(outputs[:, :4], y)
                train_loss_ahi = criterion(outputs[:, 4:], y_ahi)
                optimizer.zero_grad()
                
                '''if mode == 'ppg':
                    for i in range(outputs.shape[1]):
                        train_loss[i].backward(retain_graph=True)

                    outputs = outputs[:, 0, :]
                    train_loss = train_loss[0]
                else:'''
                train_loss.backward(retain_graph=True)
                train_loss_ahi.backward(retain_graph=True)
                #train_loss.backward(retain_graph=True)
                
                #if(epoch<=10):
                
                #id_loss.backward(retain_graph=True)
                
                optimizer.step()

                _, predicted = torch.max(outputs[:,4:].data, dim=1)
                
                if first_train_epoch:
                    #epoch_gt = copy(y)
                    epoch_gt = copy(y_ahi)
                    epoch_pred = copy(predicted)
                else:
                    #epoch_gt = torch.cat([epoch_gt, y])
                    epoch_gt = torch.cat([epoch_gt, y_ahi])
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
            with torch.no_grad():
                for batch_idx, (ppg, sig, fea, label, ahi) in enumerate(val_loader):
                    
                    if batch_idx > 0 and args.debug == 1:
                        continue
                    ppg = ppg.to(device)
                    sig = sig.to(device)
                    fea = fea.to(device)
                    y = label.to(device)
                    y_ahi = ahi.to(device)

                    if mode == 'ppg':
                        y_outputs = model(ppg)
                    if mode == 'sig':
                        y_outputs = model(sig)
                    if mode == 'fea':
                        y_outputs = model(fea)
                    
                    if type(y_outputs) in (tuple, list):
                        val_feature, y_val_pred = y_outputs[0], y_outputs[1]
                    else:
                        y_val_pred = y_outputs
                        val_feature = y_outputs

                    #y_val_pred = y_val_pred[:, 0, :]
                    #y_val_pred = y_val_pred[:, :4]
                    y_val_pred = y_val_pred[:, 4:]
                    
                    if batch_idx < 5:
                        val_fc3.append(val_feature)
                    #val_loss = criterion(y_val_pred, y)
                    val_loss = criterion(y_val_pred, y_ahi)
                    
                    total_val_loss += val_loss
                    _, y_val_pred = torch.max(y_val_pred.data, dim=1)
                    
                    num_val_samples += y.nelement()
                    if first_val_epoch:
                        #val_epoch_gt = copy(y)
                        val_epoch_gt = copy(y_ahi)
                        val_epoch_pred = copy(y_val_pred)
                    else:
                        #val_epoch_gt = torch.cat([val_epoch_gt, y])
                        val_epoch_gt = torch.cat([val_epoch_gt, y_ahi])
                        val_epoch_pred = torch.cat([val_epoch_pred, y_val_pred])
                    first_val_epoch = False
                mean_val_loss = total_val_loss / num_val_samples
                val_fc3 = torch.cat(val_fc3, dim=0).cpu()
                tracker.log_eval_fold_epoch(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                            {'mean_xent': mean_val_loss.cpu().numpy()}, fold_num, epoch, model)
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
        with torch.no_grad():
            for batch_idx, (ppg, sig, fea, label, ahi) in enumerate(test_loader):
                
                ppg = ppg.to(device)
                sig = sig.to(device)
                fea = fea.to(device)
                y = label.to(device)
                y_ahi = ahi.to(device)

                if mode == 'ppg':
                    y_outputs = model(ppg)                    
                if mode == 'sig':
                    y_outputs = model(sig)
                if mode == 'fea':
                    y_outputs = model(fea)
                
                if type(y_outputs) in (tuple, list):
                    test_feature, y_test_prob = y_outputs[0], y_outputs[1]
                else:
                    y_test_prob = y_outputs
                    test_feature = y_outputs

                #y_test_prob = y_test_prob[:, 0, :]
                #y_test_prob = y_test_prob[:, :4]
                y_test_prob = y_test_prob[:, 4:]
                
                if batch_idx < 5:
                    test_fc3_feature.append(test_feature)
                #test_loss = criterion(y_test_prob, y)
                test_loss = criterion(y_test_prob, y_ahi)
                
                total_test_loss += test_loss
                _, y_test_pred = torch.max(y_test_prob.data, dim=1)
                #num_test_samples += y.nelement()
                num_test_samples += y_ahi.nelement()
                
                correct_test += y_test_pred.eq(y_ahi.data).sum().item()
                if first_test_epoch:
                    #test_epoch_gt = copy(y)
                    test_epoch_gt = copy(y_ahi)
                    
                    test_epoch_pred = copy(y_test_pred)
                    test_epoch_prob = copy(y_test_prob)
                else:
                    #test_epoch_gt = torch.cat([test_epoch_gt, y])
                    test_epoch_gt = torch.cat([test_epoch_gt, y_ahi])
                    
                    test_epoch_pred = torch.cat([test_epoch_pred, y_test_pred])
                    test_epoch_prob = torch.cat([test_epoch_prob, y_test_prob])
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
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')#4
    parser.add_argument('--momentum', type=float, default=0.9, help='opt momentum')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
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
