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

from models import MODELS

from baseline_model_10h import PPG_10h, HeartRate_10h, HRV_10h, Hypnos_10h, Hypnos_10h_ablation, ConFusionLoss, FourSignal_10h,\
            Feature_10h, ALL_10h, OurPPG_10h, Spectrum_PPG

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

data_path = '../apnea/ApSense-main/processed/pickle_set/'

mode = 'apsense'#all, ppg

dataset = 'cfs' # mesa
from scipy.signal import resample
import torch
import pickle
from torch.utils.data import Dataset

class Load_10h_Dataset(Dataset):
    def __init__(self, mode):
        super(Load_10h_Dataset, self).__init__()
        
        self.mode = mode
        
        #with open(data_path + 'mesa/10h_input/'+mode+'_fea.pkl', 'rb') as f:
        #    self.fea = pickle.load(f)
        if(self.mode == 'train'):
            with open(data_path + dataset + '/mesa_by_subject_'+mode+'_200_ppg.pckl', 'rb') as f:#_x
                self.ppg = pickle.load(f)
            with open(data_path + dataset + '/mesa_by_subject_'+mode+'_200_y.pckl', 'rb') as f:
                self.ahi_label = pickle.load(f)
        else:
            with open(data_path + dataset + '/mesa_by_subject_'+mode+'_ppg.pckl', 'rb') as f:#_x
                self.ppg = pickle.load(f)
            with open(data_path + dataset + '/mesa_by_subject_'+mode+'_y.pckl', 'rb') as f:#_y
                self.ahi_label = pickle.load(f)
        
        self.ppg = np.concatenate(self.ppg)
        self.ahi_label = np.concatenate(self.ahi_label)

        self.ppg = self.ppg[self.ahi_label>=0]
        self.ahi_label = self.ahi_label[self.ahi_label>=0]

        self.ppg = torch.tensor(self.ppg).permute(0, 2, 1)
        self.ahi_label = torch.tensor(self.ahi_label)
        
        self.len = len(self.ppg)#2 * length#label.shape[0]
        
    def __getitem__(self, index):
        #ppg = self.ppg[index].permute(1, 0).unsqueeze(0)# usleep,1024
        ppg = self.ppg[index]

        ahi_label = self.ahi_label[index]
        
        return ppg.float(), ahi_label.long()
        
    def __len__(self):
        
        return self.len
    
def load_all():
    print("test labeled data:")
    test_loader = torch.utils.data.DataLoader(dataset=Load_10h_Dataset('test'), batch_size=500,
                                               shuffle=True, pin_memory=True,# drop_last=True,
                                               num_workers=0)
    print("train labeled data:")
    train_loader = torch.utils.data.DataLoader(dataset=Load_10h_Dataset('train'), batch_size=500,#train
                                               shuffle=True, pin_memory=True,#drop_last=True,
                                               num_workers=0)
    #train_loader = torch.utils.data.DataLoader(dataset=Loadone_Dataset('test'), batch_size=512,
    
    val_loader=test_loader
    return val_loader, test_loader, train_loader

#mesa_hrfea, mesa_acc, mesa_label
#RR * 8 am 3 ba 3
def main(args):
    #setup_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    class_counts = torch.tensor([4, 10, 1.5, 2.6])#label unbalence
    #class_counts = torch.tensor([46774., 48962., 15946., 14361.])#label unbalence
    total_samples = class_counts.sum()
    weights = total_samples / class_counts  # 计算权重
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.float().to(device))
    #criterion = nn.CrossEntropyLoss()
    con_criterion = ConFusionLoss()

    #class_counts = torch.tensor([4 * 7, 10 * 7, 1.5 * 7, 2.6 * 7, 4 * 2, 10 * 2, 1.5 * 2, 2.6 * 2])#label unbalence
    class_counts = torch.tensor([28020, 61441,  9535, 12901,  9637, 12615,  4427,  2732,   765, 19741, 476,  7820,  1459,  6680,   443,  3232])#
    total_samples = class_counts.sum()
    weights = total_samples / class_counts  # 计算权重
    weights = weights / weights.sum() 
    ei_criterion = nn.CrossEntropyLoss(weight=weights.float().to(device))
    
    #class_counts = torch.tensor([14, 4])#label unbalence
    class_counts = torch.tensor([20,  3.6])#torch.tensor([209643,  25493])#torch.tensor([170097,  44204])#label unbalence
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
            model = HeartRate_10h()
            #model = FourSignal_10h()
            #model.sig_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_sig_withoutseq/saved_models/fold_0_epoch_39.pth').sig_en.state_dict() )
            #summary(model, (4, 300))  # print the model summary
        if mode == 'fea':
            model = HRV_10h()
            #model = Feature_10h()
            #model.fea_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_fea_withoutseq/saved_models/fold_0_epoch_34.pth').fea_en.state_dict() )
        if mode == 'hypnos':
            model = Hypnos_10h()
        if mode == 'usleep':
            model_params = {
                'input_shape': [int(0.5 * 30 * 2 ** 10), 32, 1], #[int(ds_train.fsTime * ds_train.window), ds_train.nSpace, ds_train.nChannels], 
                'num_classes': 4, #len(events)
                'num_outputs': 30 * 2 ** 10/30,#ds_train.window // ds_train.prediction_resolution,
                'depth': 9,
                'init_filter_num': 16,
                'filter_increment_factor': 2 ** (1 / 3),
                'max_pool_size': (2, 2),
                'kernel_size': (16, 3)
            }
            model = Spectrum_PPG(**model_params)
        if mode == 'apsense':
            model = MODELS["DSepST15Net_no_branch"](num_channels=7, winsize=60)
        if mode == 'all':
            model = ALL_10h()
            #model.ppg_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_ppg_withoutseq/saved_models/fold_0_epoch_13.pth').ppg_en.state_dict() )
            #model.sig_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_sig_withoutseq/saved_models/fold_0_epoch_39.pth').sig_en.state_dict() )
            #model.fea_en.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_fea_withoutseq/saved_models/fold_0_epoch_34.pth').fea_en.state_dict() )

            
            #our_10h_all_cnn_500/saved_models
            #model.sequence.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_all_cnn_384fea_withoutloadparam/saved_models/fold_0_epoch_13.pth').sequence.state_dict() )
            
            #model = torch.load('tfboard/mesa/ResHybridAtt/our_10h_all_cnn_500/saved_models/fold_0_epoch_42.pth')
            
            #model.softmax = nn.Softmax(dim = 2)
            
            #model.classifier.load_state_dict( torch.load('tfboard/mesa/ResHybridAtt/our_10h_all_cnn_384fea_withoutloadparam/saved_models/fold_0_epoch_13.pth').classifier.state_dict() )
        #model = nn.DataParallel( model )
        
        #classifier = nn.DataParallel( ThreeClassifier(6) )
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

            '''
            if(mode == 'all'):
                if(epoch<4):
                    for param in model.ppg_en.parameters():
                        param.requires_grad = False
                    for param in model.sig_en.parameters():
                        param.requires_grad = False
                    for param in model.fea_en.parameters():
                        param.requires_grad = False
                    for param in model.sequence.parameters():
                        param.requires_grad = False
                    for param in model.classifier.parameters():
                        param.requires_grad = False
                else:
                    for param in model.ppg_en.parameters():
                        param.requires_grad = True
                    for param in model.sig_en.parameters():
                        param.requires_grad = True
                    for param in model.fea_en.parameters():
                        param.requires_grad = True
                    for param in model.sequence.parameters():
                        param.requires_grad = True
                    for param in model.classifier.parameters():
                        param.requires_grad = True
            '''
            #model.eval()
            #classifier.train()
            '''
            for batch_idx, (ppg, sig, fea, label, frag_label, ahi_label) in enumerate(val_loader):
                y = label.squeeze(0).to(device)
                if mode == 'all':
                    ppg = ppg.to(device)
                    sig = sig.to(device)
                    fea = fea.to(device)

                    outputs = model(ppg, sig, fea)
                # Forward pass
                
                if type(outputs) in (tuple, list):
                    feature, outputs, frag = outputs[0], outputs[1], outputs[2]

                    frag = frag[0, :y.shape[0], :]
                else:
                    feature = outputs
                outputs = outputs[0, :y.shape[0], :]

                value, predicted = torch.max(outputs[:, :].data, dim=1)
                
                n = len(outputs)
                top_k = int(n * 0.6)
                
                top_k =  torch.kthvalue(value, n - top_k + 1).values
                
                
                
                train_loss = criterion( outputs[value > top_k, :], predicted[value > top_k] ) 
                    
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
    		'''
            for batch_idx, (ppg, ahi_label) in enumerate(train_loader):
                
                if batch_idx > 0 and args.debug == 1:
                    continue
                
                ahi_label = ahi_label.to(device)
                ppg = ppg.to(device)
                
                outputs = model(ppg)
                
                if type(outputs) in (tuple, list):
                    #lab = outputs[2].detach()
                    #lab = softmax( lab * 2 )
                    
                    #feature, outputs, outputs_osa = outputs[0], outputs[1], outputs[2]# frag , outputs[2]
                    if mode =='hypnos':
                        feature, outputs, outputs_osa = outputs[0], outputs[1],  outputs[2] # frag , outputs[2]
                    else:
                        feature, outputs = outputs[0], outputs[1] # frag , outputs[2]

                    #frag = frag[0, :y.shape[0], :]
                else:
                    feature = outputs
                    
                #lab = lab[0, :y.shape[0], :]
                
                #hypnos OSA
                #outputs_osa = outputs_osa[0, :y.shape[0], :]

                
                
                '''
                #if(frag_label == 1):
                    #value, predicted = torch.max(outputs[:, :].data, dim=1)
                    
                    n = len(outputs)
                    top_k = int(n * 0.15)
                    
                    top_k =  torch.kthvalue(value, n - top_k + 1).values
                    
                    
                    #train_loss = criterion( outputs[value > top_k, :], predicted[value > top_k] ) 
                    
                    #train_loss = criterion( outputs[ :, :], predicted[:] ) 
                #    train_loss = criterion( outputs[:, :], lab )
                #else:
                #    train_loss = criterion( outputs[:, :], y ) #+ frag_cri( frag,  y_frag) #+ train_loss#+ criterion( ahi_outputs[:, :2], y_ahi )
                '''
                
                #ahi_outputs = ahi_outputs[0, :y_ahi.shape[0], :]
                
                if mode == 'ppg':
                    train_loss = criterion( outputs[:, :], y ) #+ train_loss + ahi_cri( ahi_outputs[:, :2], y_ahi )
                elif mode == 'hypnos':
                    train_loss = criterion(outputs, y) + frag_cri(outputs_osa, ahi_label) + con_criterion(feature[:, :y.shape[0], :].permute(1, 0, 2))
                else:
                    #train_loss = criterion( outputs[:, :], y ) #+ frag_cri( frag,  y_frag) #+ train_loss#+ criterion( ahi_outputs[:, :2], y_ahi )

                    train_loss = frag_cri( outputs, ahi_label )
                    
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs[:, :].data, dim=1)
                
                #y = y % 4
                #predicted = predicted % 4
                
                if first_train_epoch:
                    #epoch_gt = copy(y)
                    epoch_gt = copy(ahi_label)
                    epoch_pred = copy(predicted)
                else:
                    #epoch_gt = torch.cat([epoch_gt, y])
                    epoch_gt = torch.cat([epoch_gt, ahi_label])
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
                for batch_idx, (ppg, ahi_label) in enumerate(val_loader):
                    
                    if batch_idx > 0 and args.debug == 1:
                        continue

                    ahi_label = ahi_label.to(device)
                    ppg = ppg.to(device)
                    
                    y_outputs = model(ppg)
                        
                    # Forward pass
                    
                    if type(y_outputs) in (tuple, list):
                        if mode =='hypnos':
                            val_feature, y_val_pred, y_frag_prob = y_outputs[0], y_outputs[1],  y_outputs[2] # frag , outputs[2]
                        else:
                            val_feature, y_val_pred = y_outputs[0], y_outputs[1]#y_frag_prob , y_outputs[2]
                    else:
                        y_val_pred = y_outputs
                        val_feature = y_outputs
                    

                    #y_val_pred = y_val_pred[:, 0, :]
                    #y_val_pred = y_val_pred[:, -1, :4]
                    #y_val_pred = y_val_pred[:, 4:]
                    
                    #y_val_pred = classifier( val_feature )[:, :4]
                    
                    if batch_idx < 5:
                        val_fc3.append(val_feature)
                    
                    #val_loss = criterion(y_val_pred[:, :], y)
                    val_loss = frag_cri(y_val_pred, ahi_label)
                    
                    total_val_loss += val_loss
                    _, y_val_pred = torch.max(y_val_pred[:, :].data, dim=1)

                    #y = y % 4
                    #y_val_pred = y_val_pred % 4
                    
                    num_val_samples += ahi_label.nelement()
                    if first_val_epoch:
                        #val_epoch_gt = copy(y)
                        val_epoch_gt = copy(ahi_label)
                        val_epoch_pred = copy(y_val_pred)
                    else:
                        #val_epoch_gt = torch.cat([val_epoch_gt, y])
                        val_epoch_gt = torch.cat([val_epoch_gt, ahi_label])
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
            for batch_idx, (ppg, ahi_label) in enumerate(test_loader):
                
                ahi_label = ahi_label.to(device)
                ppg = ppg.to(device)

                y_outputs = model(ppg)
                
                # Forward pass
                if type(y_outputs) in (tuple, list):
                    if mode =='hypnos':
                        test_feature, y_test_prob, y_frag_prob = y_outputs[0], y_outputs[1],  y_outputs[2] # frag , outputs[2]
                    else:
                        test_feature, y_test_prob = y_outputs[0], y_outputs[1]#y_frag_prob , y_outputs[2]
                else:
                    y_test_prob = y_outputs
                    test_feature = y_outputs

                
                if batch_idx < 5:
                    test_fc3_feature.append(test_feature)
                
                #test_loss = criterion(y_test_prob[:, :], y)
                test_loss = frag_cri(y_test_prob, ahi_label)
                
                total_test_loss += test_loss
                _, y_test_pred = torch.max(y_test_prob[:, :].data, dim=1)

                #y = y % 4
                #y_test_pred = y_test_pred % 4
                
                #num_test_samples += y.nelement()
                num_test_samples += ahi_label.nelement()
                
                #correct_test += y_test_pred.eq(y.data).sum().item()
                correct_test += y_test_pred.eq(ahi_label.data).sum().item()
                
                if first_test_epoch:
                    #test_epoch_gt = copy(y)
                    test_epoch_gt = copy(ahi_label)
                    
                    test_epoch_pred = copy(y_test_pred)
                    test_epoch_prob = copy(y_test_prob[:, :4])
                else:
                    #test_epoch_gt = torch.cat([test_epoch_gt, y])
                    test_epoch_gt = torch.cat([test_epoch_gt, ahi_label])
                    
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
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes or labels')#4
    parser.add_argument('--momentum', type=float, default=0.9, help='opt momentum')
    parser.add_argument('--epochs', type=int, default=60, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
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
