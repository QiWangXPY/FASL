from __future__ import print_function

import sys
import argparse
import time
import math
import numpy as np
import pandas as pd
import torch.optim as optim
import os
import glob

from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn

from model import DenseNet
#from deepsense_model import MyUTDmodel
#from attnsense_model import MyUTDmodel
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
import pickle

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=25,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--iterative_epochs', type=int, default=5,
                        help='number of iterative training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='MyUTDmodel')
    parser.add_argument('--dataset', type=str, default='UTD-MHAD',
                        choices=['USC-HAR', 'UTD-MHAD', 'ours'], help='dataset')
    parser.add_argument('--method', type=str, default='iterative',
                        choices=['iterative', 'finetune'], help='dataset')
    parser.add_argument('--num_class', type=int, default=2,#27,
                        help='num_class')
    parser.add_argument('--num_train_basic', type=int, default=1,#[600,500,400,300,200,100]*2
                        help='num_train_basic')
    parser.add_argument('--num_test_basic', type=int, default=8,#[600,500,400,300,200,100]
                        help='num_test_basic')
    parser.add_argument('--label_rate', type=int, default=5,#[600,500,400,300,200,100]
                        help='label_rate')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='./save/Cosmo/UTD-MHAD_models/Cosmo_UTD-MHAD_MyUTDmodel_label_',
                        help='path to pre-trained model')
    parser.add_argument('--trial', type=int, default='1',
                        help='id for recording multiple runs')
    parser.add_argument('--guide_flag', type=int, default='1',
                        help='id for recording multiple runs')
    parser.add_argument('--cuda', type=str, default='7',
                        help='cuda id')
    opt = parser.parse_args()

    # set the path according to the environment
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        if mode == 'train':
            mode += '_frag'
        else:
            mode += '_exc'
        with open(data_path + 'mesa/'+mode+'_label.pkl', 'rb') as f:
            test_label = pickle.load(f)
            
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
            ref = self.ref[self.index[sleep_cla][id][-1]]
            
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
            
        return ppg.float(), sig.float(), fea.float(), torch.tensor(sleep_cla)
		

    def __len__(self):
        return self.len

def set_loader(opt):
    
    val_li = list()
    #load labeled train and test data
    print("train labeled data:")
    mode = 'train'
    train_dataset = Loadone_Dataset(mode)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)#, drop_last=True)
    
    print("test data:")
    mode = 'test'
    test_dataset = Loadone_Dataset(mode)    
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    val_li.append(val_loader)

    return train_loader, val_li

def set_model(opt):

    # ppg 
    criterion = torch.nn.CrossEntropyLoss()
    
    #model = ppg_model()
    #model = sig_model()
    model = fea_model()
    
    
    ## load pretrained feature encoders
    '''
    ckpt_path = opt.ckpt + str(opt.label_rate) + '_lr_0.01_decay_0.9_bsz_32_temp_0.07_trial_False_epoch_300/last.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)'''
    
    model = model.cuda()
    criterion = criterion.cuda()

    return model, criterion#classifier,


def train(train_loader, model,# classifier,
          criterion, optimizer, epoch, opt):
    """one epoch training"""
   
    '''if ( int(epoch / opt.iterative_epochs) % 2 ) == 0:   
        model.train()
        classifier.eval() 
    else: 
        model.eval()
        classifier.train() 
    '''
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (ppg, sig, fea, labels) in enumerate(train_loader):
        
        #input_data1 = torch.unsqueeze(ppg, 1)# ppg
        #input_data1 = sig
        input_data1 = fea
        
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]

        # compute loss
        output = model(input_data1)
        
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc, _ = accuracy(output, labels, topk=(1, 1))#5
        top1.update(acc[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model,# classifier,
             criterion, opt):
    """validation"""
    model.eval()
    #classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))
    label_list = []
    pred_list = []

    with torch.no_grad():
        end = time.time()
        for idx, (ppg, sig, fea, labels) in enumerate(val_loader):
            
            #input_data1 = torch.unsqueeze(ppg, 1) #ppg
            #input_data1 = sig
            input_data1 = fea
        
            if torch.cuda.is_available():
                input_data1 = input_data1.cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(input_data1)
            loss = criterion(output, labels)

            # calculate and store confusion matrix
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # update metric
            losses.update(loss.item(), bsz)
            acc, _ = accuracy(output, labels, topk=(1, 1))#5
            top1.update(acc[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    F1score_test = f1_score(label_list, pred_list, average="macro") # macro sees all class with the same importance

    print(' * Acc@1 {top1.avg:.3f}\t'
        'F1-score {F1score_test:.3f}\t'.format(top1=top1, F1score_test=F1score_test))

    return losses.avg, top1.avg, confusion, F1score_test
    
def save_model(model,# classifier,
               optimizer, opt, epoch):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    ckpt_path = opt.ckpt + str(opt.label_rate) + '_lr_0.01_decay_0.9_bsz_32_temp_0.07_trial_False_epoch_'+str(epoch)+'/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_file = os.path.join(
    ckpt_path, '{epoch}.pth'.format(epoch="model"))
    '''torch.save(state, save_file)
    state = {
        'opt': opt,
        'model': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    save_file = os.path.join(
    ckpt_path, '{epoch}.pth'.format(epoch="classifier"))'''
    torch.save(state, save_file)
    del state

def ppg_model():
    # ppg 
    criterion = torch.nn.CrossEntropyLoss()
    
    Hz = 128 // 2
    SEG_SEC = 300
    NUM_CLASSES = 4
    IN_CHAN = 1
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
            #32 for _ in range(N_DENSE_BLOCK)
            32 for _ in range(N_DENSE_BLOCK)
        ],
        n_blocks=[
            2 for _ in range(N_DENSE_BLOCK)
        ],
        #low_conv_cfg=[32, 32, 'M'],
        low_conv_cfg=[16, 32, 64, 'M'],
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

def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    opt = parse_option()
    result_record = list()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    test_len = 1
    for i in range(test_len):
        result_record.append(np.zeros((opt.trial, 3)))
    
    for trial_id in range(opt.trial):

        # build data loader
        train_loader, val_loader = set_loader(opt)

        # build model and criterion
        
        model,  criterion = set_model(opt)
        
        # build optimizer for feature extractor and classifier
        optimizer = optim.SGD([ 
                    {'params': model.parameters(), 'lr': 1e-4}],   # 0
                    #{'params': classifier.parameters(), 'lr': opt.learning_rate}],
                    momentum=opt.momentum,
                    weight_decay=opt.weight_decay)

        #record_acc = np.zeros(opt.epochs)
        best_acc = np.zeros(test_len)
        val_li = np.zeros(test_len)
        F1_li = np.zeros(test_len)
        
        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model,# classifier, 
                              criterion, optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            # eval for one epoch
            for i in range(test_len):
                loss, val_acc, con, val_F1score = validate(val_loader[i], model,# classifier,
                                                                 criterion, opt)
                val_li[i] = val_acc
                if val_li[i] > best_acc[i]:
                    best_acc[i] = val_li[i]
                    F1_li[i] = val_F1score
                    confusion = con
            '''if epoch%10==0:
                save_model(model,# classifier,
                  optimizer, opt, epoch)'''
        #   record_acc[epoch-1] = val_acc
        for i in range(test_len):
            result_record[i][trial_id, 0] = best_acc[i]
            result_record[i][trial_id, 1] = val_li[i]
            result_record[i][trial_id, 2] = F1_li[i]
    save_model(model,# classifier,
              optimizer, opt, opt.epochs)
    for i in range(test_len):
        print(str(i)+"best accuracy:", np.mean(result_record[i][:, 0]))
        print("std accuracy:", np.std(result_record[i][:, 0]))
        print("val accuracy:", np.mean(result_record[i][:, 1]))
        print("std accuracy:", np.std(result_record[i][:, 1]))
        print("F1 accuracy:", np.mean(result_record[i][:, 2]))
        print("std accuracy:", np.std(result_record[i][:, 2]))
        '''
        print('best accuracy: {:.2f}'.format(best_acc))
        print('last accuracy: {:.3f}'.format(val_li))
        print('final F1:{:.3f}'.format(F1_li))
        print("confusion_result_labelrate_{:,}_{}:".format(opt.label_rate, opt.method))

        print("iterative epoch:", opt.iterative_epochs)
 
    if not os.path.exists('record'):
            os.makedirs('record')
        ''' 
        np.savetxt("./record/{}_confusion_result_labelrate_{:,}_epoch{}_{}.txt".format('fea', opt.label_rate, opt.iterative_epochs, trial_id), confusion)
        #np.savetxt("./record/confusion_record_acc_labelrate_{:,}_epoch_{}_{}.txt".format(opt.label_rate, opt.iterative_epochs, trial_id), record_acc)


if __name__ == '__main__':
    main()
