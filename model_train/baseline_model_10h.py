import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import stochastic_depth
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 定义线性层用于生成查询、键和值
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x 的形状为 [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # 计算查询、键和值
        Q = self.query_layer(x)  # 形状 [batch_size, seq_len, output_dim]
        K = self.key_layer(x)    # 形状 [batch_size, seq_len, output_dim]
        V = self.value_layer(x)  # 形状 [batch_size, seq_len, output_dim]

        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2))  # 形状 [batch_size, seq_len, seq_len]
        scores = scores / (self.output_dim ** 0.5)  # 缩放

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # 形状 [batch_size, seq_len, seq_len]

        # 计算加权和值
        output = torch.bmm(attention_weights, V)  # 形状 [batch_size, seq_len, output_dim]

        return attention_weights, output

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    # w_2(relu(w_1(x)+b1))+b2
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # w_1,b1
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Hypnos_10h_ablation(nn.Module):
    #input shape : [batch, 1, 1200 * 1024], [batch, 1, 10h * 60min * 2 * 1024 ppg]
    def __init__(self):
        super(Hypnos_10h_ablation, self).__init__()

        en_channel = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        #encoder
        self.encoder_1 = self.ppg_encoder(en_channel[0], en_channel[0])
        self.encoder_2 = self.ppg_encoder(en_channel[1], en_channel[1])
        self.encoder_3 = self.ppg_encoder(en_channel[2], en_channel[2])
        self.encoder_4 = self.ppg_encoder(en_channel[3], en_channel[3])
        self.encoder_5 = self.ppg_encoder(en_channel[4], en_channel[4])
        self.encoder_6 = self.ppg_encoder(en_channel[5], en_channel[5])
        self.encoder_7 = self.ppg_encoder(en_channel[6], en_channel[6])
        self.encoder_8 = self.ppg_encoder(en_channel[7], en_channel[7])
        
        #sequence
        self.position_encoding_1 = PositionalEncoding(1200, 0.1)
        self.seq_attn_1 = nn.MultiheadAttention(1200, 1, dropout=0.1, batch_first=True)
        self.norm_11 = nn.LayerNorm(normalized_shape=(1024) )
        self.forward_1 = PositionwiseFeedForward(1024, 1024)
        self.norm_12 = nn.LayerNorm(normalized_shape=(1024) )
        
        self.seq_attn_2 = nn.MultiheadAttention(1200, 1, dropout=0.1, batch_first=True)
        self.norm_21 = nn.LayerNorm(normalized_shape=( 1024) )
        self.forward_2 = PositionwiseFeedForward(1024, 1024)
        self.norm_22 = nn.LayerNorm(normalized_shape=( 1024) )
        
        #classifier
        self.classifier = nn.Conv1d(in_channels=1024, out_channels=4, kernel_size=1, dilation=1)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, heart_10h):        
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_1(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_2(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_3(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_4(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_5(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_6(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_7(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_8(heart_10h)], axis = 1))

        heart_10h = heart_10h.permute([0, 2, 1]).reshape([heart_10h.shape[0], 1200, -1]).permute([0, 2, 1])
        #heart_10h = heart_10h.reshape([heart_10h.shape[0], 1200, -1])
        
        #sequence
        heart_10h_pre = self.position_encoding_1(heart_10h)
        heart_10h, _ = self.seq_attn_1(heart_10h_pre, heart_10h_pre, heart_10h_pre)
        heart_10h_pre = self.norm_11( (heart_10h_pre + heart_10h).permute([0, 2, 1]) )
        heart_10h = self.forward_1(heart_10h_pre)
        
        heart_10h_pre = self.norm_12(heart_10h + heart_10h_pre).permute([0, 2, 1])
        
        heart_10h, _ = self.seq_attn_2(heart_10h_pre, heart_10h_pre, heart_10h_pre)
        heart_10h_pre = self.norm_21( (heart_10h_pre + heart_10h).permute([0, 2, 1]) )
        heart_10h = self.forward_2(heart_10h_pre)
        
        heart_10h = self.norm_22( heart_10h + heart_10h_pre )

        fea = heart_10h
        heart_10h = heart_10h.permute([0, 2, 1])
        #classifier
        heart_10h = self.softmax( self.classifier(heart_10h) ).permute([0, 2, 1])
        
        return fea, heart_10h
    def ppg_encoder(self, in_channel, out_channel):
        layer = nn.Sequential(
                nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                #nn.MaxPool1d(kernel_size=2, stride=2)
            )
        return layer
    def ppg_sequence(self):
        layer = nn.Sequential(
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=3, dilation=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=6, dilation=2),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=12, dilation=4),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=24, dilation=8),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=48, dilation=16),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
                nn.Dropout(p = 0.2)
                #nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
                #nn.LeakyReLU(negative_slope=0.15),
            )
        return layer

class Hypnos_10h(nn.Module):
    #input shape : [batch, 1, 1200 * 1024], [batch, 1, 10h * 60min * 2 * 1024 ppg]
    def __init__(self):
        super(Hypnos_10h, self).__init__()

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        #encoder
        self.sig_en = sig_encoder(1200)
        #sequence
        self.position_encoding_1 = PositionalEncoding(1200, 0.2)
        self.seq_attn_1 = nn.MultiheadAttention(1200, 1, dropout=0.2, batch_first=True)
        self.norm_11 = nn.LayerNorm(normalized_shape=(1200, 256) )
        self.forward_1 = PositionwiseFeedForward(256, 256)
        self.norm_12 = nn.LayerNorm(normalized_shape=(1200, 256) )

        
        self.seq_attn_2 = nn.MultiheadAttention(1200, 1, dropout=0.2, batch_first=True)
        self.norm_21 = nn.LayerNorm(normalized_shape=(1200, 256) )
        self.forward_2 = PositionwiseFeedForward(256, 256)
        self.norm_22 = nn.LayerNorm(normalized_shape=(1200, 256) )
        
        #classifier
        self.expert_1 = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(negative_slope=0.15),
            )
        self.expert_2 = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(negative_slope=0.15),
            )
        self.expert_share = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(negative_slope=0.15),
            )
        self.gate = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Softmax(dim = 2)
            )
        self.classifier_1 = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Linear(64, 4),
                nn.Softmax(dim = 2)
            )
        self.classifier_2 = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Linear(64, 2),
                nn.Softmax(dim = 2)
            )
        
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self, heart_10h):
        #encode
        heart_10h = self.sig_en(heart_10h)
        
        heart_10h = heart_10h.permute([0, 2, 1]).reshape([heart_10h.shape[0], 1200, -1])

        fea = heart_10h
        
        #sequence
        heart_10h_pre = self.position_encoding_1(heart_10h.permute([0, 2, 1]))
        heart_10h, _ = self.seq_attn_1(heart_10h_pre, heart_10h_pre, heart_10h_pre)
        heart_10h_pre = self.norm_11( (heart_10h_pre + heart_10h).permute([0, 2, 1]) )
        heart_10h = self.forward_1(heart_10h_pre)


        
        #heart_10h = self.norm_12(heart_10h + heart_10h_pre)#.permute([0, 2, 1])
        heart_10h_pre = self.norm_12(heart_10h + heart_10h_pre).permute([0, 2, 1])
        heart_10h, _ = self.seq_attn_2(heart_10h_pre, heart_10h_pre, heart_10h_pre)
        heart_10h_pre = self.norm_21( (heart_10h_pre + heart_10h).permute([0, 2, 1]) )
        heart_10h = self.forward_2(heart_10h_pre)
        
        heart_10h = self.norm_22( heart_10h + heart_10h_pre )
        
        #classifier
        
        #fea = heart_10h
        
        heart_1 = self.expert_1(heart_10h)
        heart_2 = self.expert_2(heart_10h)
        heart_share = self.expert_share(heart_10h)
        heart_gate = self.gate(heart_10h)

        heart_1 = torch.concat([heart_1, heart_share], axis = 2) + heart_gate
        heart_2 = torch.concat([heart_2, heart_share], axis = 2) + heart_gate

        heart_10h = self.classifier_1[:3](heart_1)
        lab = heart_10h
        heart_10h = self.softmax(lab)
        OSA_10h = self.classifier_2(heart_2)
        
        return fea, heart_10h, OSA_10h, lab

'''class Hypnos_10h(nn.Module):
    def ppg_encoder(self, in_channel, out_channel, norm_shape):
        layer = nn.Sequential(
                nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LayerNorm(normalized_shape=(norm_shape) ),
                nn.LeakyReLU(negative_slope=0.15),

                nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
'''
'''nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
        nn.LeakyReLU(negative_slope=0.15),
        nn.MaxPool1d(kernel_size=2, stride=2)''''''
        return layer
    def ppg_sequence(self):
        layer = nn.Sequential(
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=3, dilation=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=6, dilation=2),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=12, dilation=4),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=24, dilation=8),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=48, dilation=16),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
                nn.Dropout(p = 0.2)
                #nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
                #nn.LeakyReLU(negative_slope=0.15),
            )
        return layer

    #input shape : [batch, 1, 1200 * 1024], [batch, 1, 10h * 60min * 2 * 1024 ppg]
    def __init__(self):
        super(Hypnos_10h, self).__init__()

        en_channel = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        norm_shape = [1200*1024, 1200*512, 1200*256, 1200*128, 1200*64, 1200*32, 1200*16, 1200*8]
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        #encoder
        self.encoder_1 = self.ppg_encoder(en_channel[0], en_channel[1], norm_shape[0])
        self.encoder_2 = self.ppg_encoder(en_channel[1], en_channel[2], norm_shape[1])
        self.encoder_3 = self.ppg_encoder(en_channel[2], en_channel[3], norm_shape[2])
        self.encoder_4 = self.ppg_encoder(en_channel[3], en_channel[4], norm_shape[3])
        self.encoder_5 = self.ppg_encoder(en_channel[4], en_channel[5], norm_shape[4])
        self.encoder_6 = self.ppg_encoder(en_channel[5], en_channel[6], norm_shape[5])
        self.encoder_7 = self.ppg_encoder(en_channel[6], en_channel[7], norm_shape[6])
        self.encoder_8 = self.ppg_encoder(en_channel[7], en_channel[8], norm_shape[7])

        #sequence
        self.position_encoding_1 = PositionalEncoding(1200, 0.2)
        self.seq_attn_1 = nn.MultiheadAttention(1200, 1, dropout=0.2, batch_first=True)
        self.norm_11 = nn.LayerNorm(normalized_shape=(1200, 1024) )
        self.forward_1 = PositionwiseFeedForward(1024, 1024)
        self.norm_12 = nn.LayerNorm(normalized_shape=(1200, 1024) )

        
        self.seq_attn_2 = nn.MultiheadAttention(1200, 1, dropout=0.2, batch_first=True)
        self.norm_21 = nn.LayerNorm(normalized_shape=(1200, 1024) )
        self.forward_2 = PositionwiseFeedForward(1024, 1024)
        self.norm_22 = nn.LayerNorm(normalized_shape=(1200, 1024) )
        
        #classifier
        self.expert_1 = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(negative_slope=0.15),
            )
        self.expert_2 = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(negative_slope=0.15),
            )
        self.expert_share = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(negative_slope=0.15),
            )
        self.gate = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Softmax(dim = 2)
            )
        self.classifier_1 = nn.Sequential(
                nn.Linear(512, 128),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Linear(128, 4),
                nn.Softmax(dim = 2)
            )
        self.classifier_2 = nn.Sequential(
                nn.Linear(512, 128),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Linear(128, 2),
                nn.Softmax(dim = 2)
            )
        
        
    def forward(self, heart_10h):
        #encode
        heart_10h = self.encoder_1(heart_10h)
        heart_10h = self.encoder_2(heart_10h)
        heart_10h = self.encoder_3(heart_10h)
        heart_10h = self.encoder_4(heart_10h)
        heart_10h = self.encoder_5(heart_10h)
        heart_10h = self.encoder_6(heart_10h)
        #heart_10h = self.encoder_7(heart_10h)
        #heart_10h = self.encoder_8(heart_10h)
        
        heart_10h = heart_10h.permute([0, 2, 1]).reshape([heart_10h.shape[0], 1200, -1])

        #sequence
        heart_10h_pre = self.position_encoding_1(heart_10h.permute([0, 2, 1]))
        heart_10h, _ = self.seq_attn_1(heart_10h_pre, heart_10h_pre, heart_10h_pre)
        heart_10h_pre = self.norm_11( (heart_10h_pre + heart_10h).permute([0, 2, 1]) )
        heart_10h = self.forward_1(heart_10h_pre)


        
        #heart_10h = self.norm_12(heart_10h + heart_10h_pre)#.permute([0, 2, 1])
        heart_10h_pre = self.norm_12(heart_10h + heart_10h_pre).permute([0, 2, 1])
        heart_10h, _ = self.seq_attn_2(heart_10h_pre, heart_10h_pre, heart_10h_pre)
        heart_10h_pre = self.norm_21( (heart_10h_pre + heart_10h).permute([0, 2, 1]) )
        heart_10h = self.forward_2(heart_10h_pre)
        
        heart_10h = self.norm_22( heart_10h + heart_10h_pre )
        
        #classifier
        fea = heart_10h
        
        heart_1 = self.expert_1(heart_10h)
        heart_2 = self.expert_2(heart_10h)
        heart_share = self.expert_share(heart_10h)
        heart_gate = self.gate(heart_10h)

        heart_1 = torch.concat([heart_1, heart_share], axis = 2) + heart_gate
        heart_2 = torch.concat([heart_2, heart_share], axis = 2) + heart_gate
        
        

        heart_10h = self.classifier_1(heart_1)
        OSA_10h = self.classifier_2(heart_2)
        
        return fea, heart_10h, OSA_10h
'''
class PPG_10h(nn.Module):
    #input shape : [batch, 1, 1200 * 1024], [batch, 1, 10h * 60min * 2 * 1024 ppg]
    def __init__(self):
        super(PPG_10h, self).__init__()

        en_channel = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        #encoder
        self.encoder_1 = self.ppg_encoder(en_channel[0], en_channel[0])
        self.encoder_2 = self.ppg_encoder(en_channel[1], en_channel[1])
        self.encoder_3 = self.ppg_encoder(en_channel[2], en_channel[2])
        self.encoder_4 = self.ppg_encoder(en_channel[3], en_channel[3])
        self.encoder_5 = self.ppg_encoder(en_channel[4], en_channel[4])
        self.encoder_6 = self.ppg_encoder(en_channel[5], en_channel[5])
        self.encoder_7 = self.ppg_encoder(en_channel[6], en_channel[6])
        self.encoder_8 = self.ppg_encoder(en_channel[7], en_channel[7])
        
        #sequence
        self.seq_linear = nn.Linear(1024, 128)
        self.seq_1 = self.ppg_sequence()
        self.seq_2 = self.ppg_sequence()
        
        #classifier
        self.classifier = nn.Conv1d(in_channels=128, out_channels=4, kernel_size=1, dilation=1)#4
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, heart_10h):        
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_1(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_2(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_3(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_4(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_5(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_6(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_7(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_8(heart_10h)], axis = 1))

        heart_10h = heart_10h.permute([0, 2, 1]).reshape([heart_10h.shape[0], 1200, -1])
        
        #fea = heart_10h
        #sequence
        heart_10h = self.seq_linear(heart_10h)
        heart_10h = heart_10h.permute([0, 2, 1])
        
        heart_temp = self.seq_1( heart_10h )
        heart_10h = heart_10h + heart_temp
        heart_temp = self.seq_2( heart_10h )
        heart_10h = heart_10h + heart_temp


        fea = heart_10h.permute(0, 2, 1)
        
        #classifier
        lab = self.classifier(heart_10h) 
        heart_10h = self.softmax(  lab ).permute([0, 2, 1])
        
        return fea, heart_10h, lab
    def ppg_encoder(self, in_channel, out_channel):
        layer = nn.Sequential(
                nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
                nn.LeakyReLU(negative_slope=0.15),
                #nn.MaxPool1d(kernel_size=2, stride=2)
            )
        return layer
    def ppg_sequence(self):
        layer = nn.Sequential(
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=3, dilation=1),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=6, dilation=2),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=12, dilation=4),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=24, dilation=8),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=48, dilation=16),
                nn.LeakyReLU(negative_slope=0.15),
                nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
                nn.Dropout(p = 0.2)
                #nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
                #nn.LeakyReLU(negative_slope=0.15),
            )
        return layer

class HeartRate_10h(nn.Module):
    #input shape : [1, 72000]
    #input shape : [1200, 256], [10h * 60min * 2 , 256] ( 64hz heartrate at central, and nearly heartrate)
    def __init__(self):
        super(HeartRate_10h, self).__init__()

        self.conv1d = nn.Conv1d(in_channels = 1, out_channels = 8, kernel_size = 1)
        
        #encoder
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        en_channel = [8, 16, 32]
        self.encoder_1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = en_channel[0], out_channels = en_channel[0], kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = en_channel[0], out_channels = en_channel[0], kernel_size = 3, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = en_channel[1], out_channels = en_channel[1], kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = en_channel[1], out_channels = en_channel[1], kernel_size = 3, padding=1)
        )
        self.encoder_3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = en_channel[2], out_channels = en_channel[2], kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = en_channel[2], out_channels = en_channel[2], kernel_size = 3, padding=1)
        )
        self.dense = nn.Linear(2048, 128)

        #sequence
        self.sequence_1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=6, dilation=2),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=12, dilation=4),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=24, dilation=8),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=48, dilation=16),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
            nn.Dropout(p = 0.2)
        )
        self.sequence_2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=6, dilation=2),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=12, dilation=4),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=24, dilation=8),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=48, dilation=16),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 7, padding=96, dilation=32),
            nn.Dropout(p = 0.2)
        )

        #classifier
        self.classifier = nn.Conv1d(in_channels=128, out_channels=4, kernel_size=1, dilation=1)

        self.softmax = nn.Softmax(dim = 2)

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
    def forward(self, heart_10h):
        heart_list = list()
        for j in range(0, heart_10h.size(1), 60):
            beg = j - 96
            end = j + 160
            beg = max(beg, 0)
            temp = heart_10h[:, beg: end]
            temp = self.adjust_length_2D(temp, 256)
            heart_list.append(temp.unsqueeze(1))
        heart_10h = torch.concat(heart_list, axis = 1).squeeze(0)

        
        heart_10h = self.conv1d(heart_10h.unsqueeze(1))

        #encoder
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_1(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_2(heart_10h)], axis = 1))
        heart_10h = self.max_pool(torch.concat([heart_10h, self.encoder_3(heart_10h)], axis = 1))
        heart_10h = self.dense(heart_10h.reshape([heart_10h.shape[0], -1])).unsqueeze(0)

        fea = heart_10h
        #sequence
        heart_10h = heart_10h.permute(0, 2, 1)
        heart_10h = self.sequence_1(heart_10h) + heart_10h
        heart_10h = self.sequence_1(heart_10h) + heart_10h

        #fea = heart_10h.permute([0, 2, 1])
        #classifier
        heart_10h = self.classifier(heart_10h)
        lab = heart_10h.permute(0, 2, 1)

        heart_10h = self.softmax(lab)
        
        return fea, heart_10h, lab

class HRV_10h(nn.Module):
    #input shape : [batch, 1200, 30], [batch, 10h * 60min *2, 30 feature number]
    #input shape : [batch, 28, 1200]
    def __init__(self):
        super(HRV_10h, self).__init__()

        #encoder
        self.encoder = nn.Sequential(
            nn.Linear(308, 512),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(64, 16),
            nn.LeakyReLU(negative_slope=0.15),
        )

        #sequence
        self.lstm1 = nn.LSTM(16, 16, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        self.relu = nn.LeakyReLU(negative_slope=0.15)
        self.drop = nn.Dropout(p = 0.2)
        
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(64, 4),
            nn.Dropout(p = 0.2),
            
        )
        self.softmax = nn.Softmax(dim = 2)

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

    def forward(self, heart_10h):
        heart_list = list()
        for j in range(0, heart_10h.size(1), 1):
            beg = j - 5
            end = j + 6
            beg = max(beg, 0)
            temp = heart_10h[:, beg: end, :].reshape([1, -1])
            temp = self.adjust_length_2D(temp, 308)
            heart_list.append(temp)
        heart_10h = torch.concat(heart_list, axis = 0).unsqueeze(0)
        
        #encoder
        heart_10h = self.encoder(heart_10h)

        fea = heart_10h
        
        #sequence
        heart_10h, _ = self.lstm1(heart_10h)
        heart_10h, _ = self.lstm2( self.relu(heart_10h) )
        heart_10h, _ = self.lstm3( self.relu(heart_10h) )
        heart_10h = self.drop(heart_10h)

        lab = self.classifier(heart_10h)
        #classifier
        heart_10h = self.softmax( lab )

        
        return fea, heart_10h, lab

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, weight_decay=0.0,
                 residual=True, stochastic_depth=True, activation='gelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.residual = residual
        self.stochastic_depth = stochastic_depth
        self.activation = activation
        self.proj = None
        
        if residual and in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same')
        
        self.weight_decay = weight_decay

    def forward(self, x):
        x_ = self.conv(x)

        if self.activation == 'gelu':
            x_ = F.gelu(x_)
        else:
            raise ValueError("Unsupported activation function")

        x_ = self.bn(x_)
        
        if self.residual:
            if self.proj is not None:
                x = self.proj(x)

            if self.stochastic_depth:
                survival_prob = 0.9
                return stochastic_depth(x + x_, 0.9)
            else:
                return x + x_
        else:
            return x_

class Spectrum_PPG(nn.Module):
    def __init__(self, input_shape, num_classes, num_outputs, depth=None,
                 init_filter_num=8, filter_increment_factor=2 ** (1 / 3),
                 kernel_size=(16, 1), max_pool_size=(2, 1), activation='gelu',
                 output_layer='sigmoid', weight_decay=0.0,
                 residual=False, stochastic_depth=False):
        
        super(Spectrum_PPG, self).__init__()

        if depth is None:
            depth = self.determine_depth(input_shape[0], max_pool_size[0])

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.depth = depth
        self.init_filter_num = init_filter_num
        self.filter_increment_factor = filter_increment_factor
        self.kernel_size = kernel_size
        self.max_pool_size = max_pool_size
        self.activation = activation
        self.output_layer = output_layer
        self.weight_decay = weight_decay
        self.residual = residual
        self.stochastic_depth = stochastic_depth

        # Zero padding
        zeros_to_add = int(2 ** (np.ceil(np.log2(input_shape[0]))) - input_shape[0])
        self.padding = (zeros_to_add // 2, 0) if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2) else (0, 0)

        features = init_filter_num 
        self.features_list = []
        self.kernel_size_list = []
        self.max_pool_size_list = []
        self.encoder_layers = nn.ModuleList()
        self.enbn_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.kernel_sizes = [[16, 3], [16, 3], [16, 3], [16, 2], [16, 1], [16, 1], [16, 1], [16, 1], [16, 1]]
        
        # Encoder
        for i in range(depth):
            self.features_list.append(features)
            self.kernel_size_list.append(kernel_size)
            self.max_pool_size_list.append(max_pool_size)

            # Feature extractor
            in_c = 1 if i==0 else int(features)
            self.encoder_layers.append(ConvBlock(in_channels=in_c, 
                                                 out_channels=int(features),
                                                 kernel_size=kernel_size,
                                                 weight_decay=weight_decay,
                                                 residual=residual,
                                                 stochastic_depth=stochastic_depth,
                                                 activation=activation))
            
            in_c = int(features)
            features *= filter_increment_factor

            padding = self.calculate_padding(max_pool_size)

            # Convolution and Batch Normalization
            self.encoder_layers.append(nn.Conv2d(in_c, int(features), 
                                                  kernel_size=max_pool_size, 
                                                  stride=max_pool_size, 
                                                  padding=padding))
            self.enbn_layers.append(nn.BatchNorm2d(int(features)))

            # Update kernel_size and max_pool_size
            kernel_size = self.kernel_sizes[i]
            if i>3:
                max_pool_size = (max_pool_size[0], 1)
        
        self.features_list.append(features)
        self.kernel_size_list.append(kernel_size)
        self.max_pool_size_list.append(max_pool_size)
        
        # Middle part
        self.middle_conv = ConvBlock(in_channels=int(features), 
                                      out_channels=int(features), 
                                      kernel_size=kernel_size,
                                      weight_decay=weight_decay,
                                      residual=residual,
                                      stochastic_depth=stochastic_depth,
                                      activation=activation)
        # Decoder
        for count, i in enumerate(reversed(range(depth))):
            padding = self.calculate_padding(self.max_pool_size_list[i + 1])
            self.decoder_layers.append(nn.ConvTranspose2d(int(self.features_list[i+1]),
                                                           int(self.features_list[i]),
                                                           kernel_size=self.max_pool_size_list[i],
                                                           stride=self.max_pool_size_list[i],
                                                           padding=padding))
            self.decoder_layers.append(nn.BatchNorm2d(int(self.features_list[i])))
            
            self.decoder_layers.append(ConvBlock(in_channels=2 * int(self.features_list[i]),
                                                 out_channels=int(self.features_list[i]),
                                                 kernel_size=self.kernel_size_list[i], activation=activation,
                       weight_decay=weight_decay, residual=residual, stochastic_depth=stochastic_depth))
        #classifier
        self.class_1 = nn.Conv1d(512, self.init_filter_num, kernel_size=1, padding='same')

        #linear
        self.linear_1 = nn.Conv1d(self.init_filter_num, self.num_classes, kernel_size=1, padding='same')
        self.linear_2 = nn.Linear(self.num_classes, self.num_classes)
    def forward(self, x):
        # Zero padding
        if self.padding[0] > 0:
            x = F.pad(x, (0, 0, self.padding[0], self.padding[0]))
        
        skips = []
        # Encoder
        for i in range(self.depth):
            x = self.encoder_layers[2 * i](x)  # Conv block
            skips.append(x)  # Save skip connection
            x = self.encoder_layers[2 * i + 1](x)  # Conv + BN
            x = self.enbn_layers[i](x)

        # Middle part
        x = self.middle_conv(x)
        
        # Decoder
        for i in range(self.depth):
            x = self.decoder_layers[3 * i](x)  # Transpose conv + BN
            x = self.decoder_layers[3 * i + 1](x)
            x = torch.cat((skips[self.depth - 1 - i], x), dim=1)  # Concatenate with skip connection
            x = self.decoder_layers[3 * i + 2](x)  # Conv block

        # Cut-off zero-padded segment
        if (self.padding[0] > 0):
            x = x[:, :, self.padding[0]:-self.padding[0], :]
        x = x.permute(0, 2, 1, 3)         
        # Reshape
        x = x.reshape(x.size(0), x.size(1), -1)  # Reshape for Conv1D
        
        # Non-linear activation
        x = self.class_1(x.permute(0, 2, 1))
        
        if self.input_shape[0] // self.num_outputs > 0:
            x = F.avg_pool1d(x, kernel_size=int(x.size(2) // self.num_outputs) )  # Average pooling if needed
        # Final layers
        fea = x.permute(0, 2, 1)
        
        x = self.linear_1(x)
        x = self.linear_2(x.permute(0, 2, 1))
        lab = x
        x = nn.Softmax(dim = 2)(x)
        #x = nn.Sigmoid()()
        
        
        
        return fea, x, lab

    def determine_depth(self, temporal_shape, temporal_max_pool_size):
        depth = 0
        while temporal_shape % 2 == 0:
            depth += 1
            temporal_shape /= round(temporal_max_pool_size)
        depth -= 1
        return depth

    def calculate_padding(self, kernel_size):
        # Calculate padding for 'same' equivalent
        padding = []
        for ks in kernel_size:
            pad = (ks - 1) // 2  # Calculate padding
            padding.append(pad)
        return tuple(padding)











## contrastive fusion loss with SupCon format: https://arxiv.org/pdf/2004.11362.pdf
class OurConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(OurConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, param, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:#特征维度拉平
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)#btz*btz,32*32单位矩阵

        contrast_count = features.shape[1]#9
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
        
        contrast_param = torch.cat(torch.unbind(param, dim=1), dim=0)# 额外参数：change to [n_views*bsz, 1]
        contrast_param = contrast_param.unsqueeze(1) * contrast_param.unsqueeze(0)
        #contrast_param = contrast_param * contrast_param
        
        #[32*9,128]，unbind，按contrast拆分后，按batch组合，与下头mask重复堆叠对应
        contrast_feature = F.normalize(contrast_feature, dim = 1)#行归一化
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count#9

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)#[32*9,32*9],[batch*线性组合，batch*线性组合]
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)# positive index,行列分别重复9次，[288*288]
        # print(mask.shape)#[1151, 1152] (btz*9)
        
        # mask-out self-contrast cases
        
        logits_mask = torch.scatter(#288*288
            torch.ones_like(mask),#所有元素均为1
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),#19*9，288
            0
        )#dig to 0, others to 1 (negative samples)
        
        mask = mask * logits_mask#positive samples except itself
        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask #exp(z_i * z_a / T)自乘，正样本做和
        
        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))#特征值，减去exp(特征/温度)按行做和
        log_prob = log_prob * contrast_param
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#sup_out
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class ConFusionLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConFusionLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:#特征维度拉平
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]#负样本

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)#btz*btz,32*32单位矩阵

        contrast_count = features.shape[1]#9
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
        #[32*9,128]，unbind，按contrast拆分后，按batch组合，与下头mask重复堆叠对应
        contrast_feature = F.normalize(contrast_feature, dim = 1)#行归一化

        anchor_feature = contrast_feature
        anchor_count = contrast_count#9

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)#[32*9,32*9],[batch*线性组合，batch*线性组合]
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)# positive index,行列分别重复9次，[288*288]
        # print(mask.shape)#[1151, 1152] (btz*9)
        
        # mask-out self-contrast cases
        
        logits_mask = torch.scatter(#288*288
            torch.ones_like(mask),#所有元素均为1
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),#19*9，288
            0
        )#dig to 0, others to 1 (negative samples)
        
        mask = mask * logits_mask#positive samples except itself
        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask #exp(z_i * z_a / T)自乘，正样本做和
        
        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))#特征值，减去exp(特征/温度)按行做和
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#sup_out
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss



#our model

'''
def en_block(in_channel, out_channel, norm_shape, pool_shape, pad):
    layer = nn.Sequential(
            nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, padding=pad),
            nn.LayerNorm(normalized_shape=(norm_shape) ),
            nn.LeakyReLU(negative_slope=0.15),

            nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding=1),
            nn.LeakyReLU(negative_slope=0.15),
            nn.MaxPool1d(kernel_size=pool_shape, stride=pool_shape),
        )
    return layer
'''
'''
class en_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_shape, pool_shape, kernel_size, pad):
        super(en_block, self).__init__()

        #encoder
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, padding=pad),
            nn.LayerNorm(normalized_shape=(norm_shape) ),
            nn.LeakyReLU(negative_slope=0.15),

            nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = kernel_size, padding=pad),
            nn.LeakyReLU(negative_slope=0.15),
            
        )
        #self.res = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, dilation=1)
        
        self.AvgPool1d = nn.MaxPool1d(kernel_size=pool_shape, stride=pool_shape)
    def forward(self, x):
        x = self.AvgPool1d(torch.concat([x, self.layer(x)], axis = 1))
        #x = self.AvgPool1d(self.res(x) + self.layer(x))

        return x
'''

class en_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_shape, pool_shape, kernel_size, pad):
        super(en_block, self).__init__()

        #encoder
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, padding=pad),
            nn.LayerNorm(normalized_shape=(norm_shape) ),
            nn.LeakyReLU(negative_slope=0.15),

            nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = kernel_size, padding=pad),
            nn.LeakyReLU(negative_slope=0.15),
            
        )
        #self.res = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, dilation=1)
        
        self.AvgPool1d = nn.MaxPool1d(kernel_size=pool_shape, stride=pool_shape)
    def forward(self, x):
        x = self.AvgPool1d(torch.concat([x, self.layer(x)], axis = 1))
        #x = self.AvgPool1d(self.res(x) + self.layer(x))

        return x

class en_block_2D(nn.Module):
    def __init__(self, in_channel, out_channel, norm_shape, pool_shape, kernel_size, pad):
        super(en_block_2D, self).__init__()

        #encoder
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, padding=pad),
            nn.LayerNorm(normalized_shape=(norm_shape) ),
            nn.LeakyReLU(negative_slope=0.15),

            nn.Conv1d(in_channels = out_channel, out_channels = out_channel, kernel_size = kernel_size, padding=pad),
            nn.LeakyReLU(negative_slope=0.15),
            
        )
        #self.res = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, dilation=1)
        
        #self.AvgPool1d = nn.MaxPool1d(kernel_size=pool_shape, stride=pool_shape)
        self.AvgPool1d = nn.AvgPool2d(kernel_size=(2, pool_shape), stride=(2, pool_shape))
    def forward(self, x):
        x = self.AvgPool1d(torch.concat([x, self.layer(x)], axis = 1))
        #x = self.AvgPool1d(self.res(x) + self.layer(x))

        return x
    
class cnn_seq_block(nn.Module):
    #[1, 768, 1200]
    def __init__(self, channel):
        super(cnn_seq_block, self).__init__()
        
        self.cnn_block1_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=3, dilation=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.15)
        #self.cnn_block1_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=12, dilation=4)
        self.cnn_block1_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=6, dilation=2)
        self.norm1 = nn.LayerNorm(normalized_shape=(1200) )
        
        
        #self.cnn_block2_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=48, dilation=16)
        self.cnn_block2_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=12, dilation=4)
        self.relu2 = nn.LeakyReLU(negative_slope=0.15)
        #self.cnn_block2_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=96, dilation=32)
        self.cnn_block2_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=24, dilation=8)
        self.norm2 = nn.LayerNorm(normalized_shape=(1200) )
        
        self.cnn_block3_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=48, dilation=16)
        #self.cnn_block3_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=36, dilation=12)
        self.relu3 = nn.LeakyReLU(negative_slope=0.15)
        self.cnn_block3_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=96, dilation=32)
        #self.cnn_block3_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=48, dilation=16)
        self.norm3 = nn.LayerNorm(normalized_shape=(1200) )
        '''
        #self.cnn_block4_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=72, dilation=24)
        self.cnn_block4_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=60, dilation=20)
        self.relu4 = nn.LeakyReLU(negative_slope=0.15)
        #self.cnn_block4_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=96, dilation=32)
        self.cnn_block4_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=72, dilation=24)
        self.norm4 = nn.LayerNorm(normalized_shape=(1200) )

        self.cnn_block5_1 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=84, dilation=28)
        self.relu5 = nn.LeakyReLU(negative_slope=0.15)
        self.cnn_block5_2 = nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = 7, padding=96, dilation=32)
        self.norm5 = nn.LayerNorm(normalized_shape=(1200) )
        '''
        self.drop = nn.Dropout(p = 0.2)
    def forward(self, x):
        x_pre = self.cnn_block1_1( x )
        x = self.relu1(x + x_pre)
        x_pre = self.cnn_block1_2( x )
        x = self.norm1(x + x_pre)

        
        x_pre = self.cnn_block2_1( x )
        x = self.relu2(x + x_pre)
        x_pre = self.cnn_block2_2( x )
        x = self.norm2(x + x_pre) 
        
        
        
        x_pre = self.cnn_block3_1( x )
        x = self.relu3(x + x_pre)
        x_pre = self.cnn_block3_2( x )
        x = self.norm3(x + x_pre) 
        
        '''
        x_pre = self.cnn_block4_1( x )
        x = self.relu4(x + x_pre)
        x_pre = self.cnn_block4_2( x )
        x = self.norm4(x + x_pre)

        x_pre = self.cnn_block5_1( x )
        x = self.relu5(x + x_pre)
        x_pre = self.cnn_block5_2( x )
        x = self.norm5(x + x_pre)
        '''
        x = self.drop(x)
        return x


class lstm_seq_block(nn.Module):
    #[1, 1200, 768]
    def __init__(self):
        super(lstm_seq_block, self).__init__()
        
        self.lstm1_1 = nn.LSTM(768, 384, bidirectional=True, batch_first=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.15)
        self.lstm1_2 = nn.LSTM(768, 384, bidirectional=True, batch_first=True)
        self.norm1 = nn.LayerNorm(normalized_shape=(768) )
        
        self.lstm2_1 = nn.LSTM(768, 384, bidirectional=True, batch_first=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.15)
        self.lstm2_2 = nn.LSTM(768, 384, bidirectional=True, batch_first=True)
        self.norm2 = nn.LayerNorm(normalized_shape=(768) )
            
        self.drop = nn.Dropout(p = 0.2)

    def forward(self, x):
        x_pre, _ = self.lstm1_1(x)
        x = self.relu1(x + x_pre)
        
        x_pre, _ = self.lstm1_2( x )
        x = self.norm1(x + x_pre)
        
        x_pre, _ = self.lstm2_1( x )
        x = self.relu2(x + x_pre)
        
        x_pre, _ = self.lstm2_2( x )
        x = self.norm2(x + x_pre) 
        
        x = self.drop(x)
        return x

class attn_seq_block(nn.Module):
    #[1, 1200, 768]
    #[:, :, mask = True]
    def __init__(self, channel):
        super(attn_seq_block, self).__init__()
        
        self.position_encoding_1 = PositionalEncoding(channel, 0.1)
        self.seq_attn_1 = nn.MultiheadAttention(channel, 1, dropout=0.1, batch_first=True, )
        self.norm_11 = nn.LayerNorm(normalized_shape=(channel) )
        self.forward_1 = PositionwiseFeedForward(channel, channel)
        self.norm_12 = nn.LayerNorm(normalized_shape=(channel) )
        
        self.seq_attn_2 = nn.MultiheadAttention(channel, 1, dropout=0.1, batch_first=True)
        self.norm_21 = nn.LayerNorm(normalized_shape=( channel) )
        self.forward_2 = PositionwiseFeedForward(channel, channel)
        self.norm_22 = nn.LayerNorm(normalized_shape=( channel) )

        self.drop = nn.Dropout(p = 0.2)
    def forward(self, x, mask = None):

        '''
        mask = torch.zeros([1, 1200, 1200])
        mask[:, :, 960:] = 1
        mask = mask[:, :, torch.randperm(1200)]
        mask[:, :, len:] = 1
        mask = mask == 1'''
        
        x_pre = self.position_encoding_1(x)
        x, _ = self.seq_attn_1( x_pre, x_pre, x_pre, attn_mask = mask)
        x_pre = self.norm_11( (x_pre + x) )
        x = self.forward_1(x_pre)
        
        x_pre = self.norm_12(x + x_pre)
        
        x, _ = self.seq_attn_2(x_pre, x_pre, x_pre, attn_mask = mask)
        x_pre = self.norm_21( (x_pre + x) )
        x = self.forward_2(x_pre)
        
        x = self.norm_22( x + x_pre )

        x = self.drop(x)
        return x

class attn_seq_block_local(nn.Module):
    #[1, 1200, 768]
    #[:, :, mask = True]
    def __init__(self, channel):
        super(attn_seq_block_local, self).__init__()

        self.channel = channel
        
        self.position_encoding_1 = PositionalEncoding(channel, 0.1)
        self.seq_attn_1 = nn.MultiheadAttention(channel, 1, dropout=0.1, batch_first=True, )
        self.norm_11 = nn.LayerNorm(normalized_shape=(channel) )
        self.forward_1 = PositionwiseFeedForward(channel, channel)
        self.norm_12 = nn.LayerNorm(normalized_shape=(channel) )
        
        self.seq_attn_2 = nn.MultiheadAttention(channel, 1, dropout=0.1, batch_first=True)
        self.norm_21 = nn.LayerNorm(normalized_shape=( channel) )
        self.forward_2 = PositionwiseFeedForward(channel, channel)
        self.norm_22 = nn.LayerNorm(normalized_shape=( channel) )

        self.drop = nn.Dropout(p = 0.2)
    def forward(self, x, mask = None):

        #x = F.pad(x, (0, 0, 25, 25), mode='constant', value=0)
        
        #x = x.reshape([25, 50, -1])
        
        '''
        mask = torch.zeros([1, 1200, 1200])
        mask[:, :, 960:] = 1
        mask = mask[:, :, torch.randperm(1200)]
        mask[:, :, len:] = 1
        mask = mask == 1'''
        
        x_pre = self.position_encoding_1(x)
        
        #x_pre = x_pre.reshape([25, 50, self.channel])
        
        x, _ = self.seq_attn_1( x_pre, x_pre, x_pre, attn_mask = mask)
        x_pre = self.norm_11( (x_pre + x) )
        x = self.forward_1(x_pre)
        x_pre = self.norm_12(x + x_pre)

        #x_pre = x_pre.reshape([1, 1250, self.channel])[:, 25:1225, :]
        #x_pre = x_pre.reshape([24, 50, self.channel])
        
        
        x, _ = self.seq_attn_2(x_pre, x_pre, x_pre, attn_mask = mask)
        x_pre = self.norm_21( (x_pre + x) )
        x = self.forward_2(x_pre)
        
        x = self.norm_22( x + x_pre )

        x = self.drop(x)
        
        return x.reshape([1, 1200, -1])

def fea_encoder(time_len):
    
    #en_channel = [28, 128]#, 128]#, 192]
    #ou_channel = [100, 128]#, 128]#, 64]
    #en_channel = [28, 64]
    #ou_channel = [36, 192]
    #en_channel = [28, 64]
    #ou_channel = [36, 64]
    en_channel = [28, 64, 128]#, 192]#, 192]
    ou_channel = [36, 64, 128]#, 64]#, 64]
    #en_channel = [28, 32, 32]
    #ou_channel = [36, 32, 32]
    norm_shape = [time_len * 3, time_len * 3, time_len * 3, time_len * 3, time_len * 3]
    norm_shape = [int(x) for x in norm_shape]
    pool_shape = [1, 1, 3]#1,
    #pool_shape = [1, 3, 3]
    
    #import torch.nn.functional as F
    layer = nn.Sequential(
            en_block(en_channel[0], ou_channel[0], norm_shape[0], pool_shape[0], 33, 16),
            en_block(en_channel[1], ou_channel[1], norm_shape[1], pool_shape[1], 33, 16),
            en_block(en_channel[2], ou_channel[2], norm_shape[2], pool_shape[2], 17, 8),
            #en_block(en_channel[3], ou_channel[3], norm_shape[3], pool_shape[3], 17, 8),
            #en_block(en_channel[4], ou_channel[4], norm_shape[4], pool_shape[4], 17, 8),
            nn.Dropout(p = 0.1)
    )
    return layer

def sig_encoder(time_len):
    '''en_channel = [4,  16, 32, 64, 128, 192]
    ou_channel = [12, 16, 32, 64, 64, 64]'''
    #en_channel = [4,  64, 128]
    #ou_channel = [60, 64, 128]
    #en_channel = [4,  32, 128]
    #ou_channel = [28, 96, 128]
    #en_channel = [4,  32, 64]
    #ou_channel = [28, 32, 64]
    en_channel = [4,  16, 64, 128]
    ou_channel = [12, 48, 64, 128]
    #en_channel = [4, 8, 16, 32]
    #ou_channel = [12, 24, 48, 32]
    #en_channel = [4,  16, 64, 128, 192]
    #ou_channel = [12, 48, 64, 64, 64]
    #en_channel = [4,  16, 64, 96, 128, 192]
    #ou_channel = [12, 48, 32, 32, 64, 64]
    norm_shape = [time_len*60, time_len*60/2, time_len*60/2/2, time_len*60/2/2/3]
    #norm_shape = [time_len*60, time_len*60/6, time_len*60/3/2/2, time_len*60/4/5/3]
    #norm_shape = [time_len*60, time_len*60/2, time_len*60/4, time_len*60/12]
    #norm_shape = [1200*60, 1200*60, 1200*60/2, 1200*60/4, 1200*60/12]
    norm_shape = [int(x) for x in norm_shape]
    #pool_shape = [6, 10]
    pool_shape = [2, 2, 3, 5]
    #pool_shape = [3, 4, 5]
    #pool_shape = [1, 2, 2, 3]
    #pool_shape = [1, 2, 2, 3, 5]

    layer = nn.Sequential(
            en_block(en_channel[0], ou_channel[0], norm_shape[0], pool_shape[0], 65, 32),#
            en_block(en_channel[1], ou_channel[1], norm_shape[1], pool_shape[1], 65, 32),
            en_block(en_channel[2], ou_channel[2], norm_shape[2], pool_shape[2], 33, 16),#33
            en_block(en_channel[3], ou_channel[3], norm_shape[3], pool_shape[3], 33, 16),
            #en_block(en_channel[4], ou_channel[4], norm_shape[4], pool_shape[4], 33, 16),
            #en_block(en_channel[5], ou_channel[5], norm_shape[5], pool_shape[5], 33, 16),
            nn.Dropout(p = 0.1)
            #en_block(en_channel[4], ou_channel[4], norm_shape[4], pool_shape[4], 1),
            #en_block(en_channel[5], ou_channel[5], norm_shape[5], pool_shape[5], 1),
    )
    
    return layer


def ppg_encoder(time_len):
    #norm_shape = [1200*1024, 1200*512, 1200*256, 1200*128, 1200*64, 1200*32, 1200*16, 1200*8]
    #pool_shape = [2, 2, 2, 2, 2, 2, 2, 2]
    #en_channel = [1, 4, 16, 64, 128]
    #ou_channel = [3, 12, 48, 64, 128]
    #en_channel = [1, 16, 64, 128]
    #ou_channel = [15, 48, 64, 128]
    #en_channel = [1, 8, 32, 128]
    #ou_channel = [7, 24, 96, 128]
    #en_channel = [1, 8, 32, 64]
    #ou_channel = [7, 24, 32, 64]
    en_channel = [1, 4, 16, 64, 128]
    ou_channel = [3, 12, 48, 64, 128]
    #en_channel = [1, 8, 16, 32, 32]
    #ou_channel = [15, 24, 48, 32, 32]
    #en_channel = [1, 4, 16, 64, 128, 192]#, 192]
    #ou_channel = [3, 12, 48, 64, 64, 64]#, 64]
    norm_shape = [time_len*1024, time_len*1024/4, time_len*1024/4/4, time_len*1024/4/4/4, time_len*1024/4/4/4/2]#, time_len*2]#, time_len*2]
    #norm_shape = [time_len*1024, time_len*1024/2, time_len*1024/2/2, time_len*1024/2/2/4, time_len*1024/2/2/4/4]#, time_len*2]#, time_len*2]
    norm_shape = [int(x) for x in norm_shape]
    #pool_shape = [4, 4, 4, 4, 4, 2, 2]
    pool_shape = [4, 4, 4, 2, 2]
    #pool_shape = [4, 8, 8, 4]
    #pool_shape = [4, 4, 4, 4 * 4]#, 2, 2]
    
    layer = nn.Sequential(
            en_block(en_channel[0], ou_channel[0], norm_shape[0], pool_shape[0], 129, 64),
            en_block(en_channel[1], ou_channel[1], norm_shape[1], pool_shape[1], 129, 64),
            en_block(en_channel[2], ou_channel[2], norm_shape[2], pool_shape[2], 65, 32),
            en_block(en_channel[3], ou_channel[3], norm_shape[3], pool_shape[3], 65, 32),
            en_block(en_channel[4], ou_channel[4], norm_shape[4], pool_shape[4], 33, 16),
            #en_block(en_channel[5], ou_channel[5], norm_shape[5], pool_shape[5], 17, 8),
            #en_block(en_channel[6], ou_channel[6], norm_shape[6], pool_shape[6], 17, 8),
            nn.Dropout(p = 0.1),
            #en_block(en_channel[5], ou_channel[5], norm_shape[5], pool_shape[5], 17, 8),
            #en_block(en_channel[6], en_channel[6], norm_shape[6], pool_shape[6], 1),
            #en_block(en_channel[7], en_channel[7], norm_shape[7], pool_shape[7], 1),
    )
    return layer

class FourSignal_10h(nn.Module):
    def __init__(self):
        super(FourSignal_10h, self).__init__()

        #encoder
        self.sig_en = sig_encoder(1200)
        
        #sequence
        self.sequence_1 = cnn_seq_block( 256 )
        #self.sequence_2 = self.seq_block( 256 )

        #classifier
        self.classifier = nn.Conv1d(in_channels=256, out_channels=4, kernel_size=1, dilation=1)
    def forward(self, sig):
        #encoder
        
        sig = self.sig_en(sig)
        #sequence
        '''
        heart_10h = sig
        
        heart_10h = self.sequence_1(heart_10h) + heart_10h
        heart_10h = self.sequence_1(heart_10h) + heart_10h
        '''
        #heart_10h = self.sequence(sig.permute([0, 2, 1])).permute([0, 2, 1])#, lstm&attn
        heart_10h = self.sequence_1(sig)#, lstm&attn
        
        fea = heart_10h.permute([0, 2, 1])
        #classifier
        heart_10h = self.classifier(heart_10h)
        heart_10h = heart_10h.permute([0, 2, 1])
        return fea, heart_10h

class Feature_10h(nn.Module):
    def __init__(self):
        super(Feature_10h, self).__init__()

        #encoder
        self.fea_en = fea_encoder()

        
        #sequence
        
        self.sequence = attn_seq_block(256)

        #classifier
        self.classifier = nn.Conv1d(in_channels=256, out_channels=4, kernel_size=1, dilation=1)
    def forward(self, fea):
        fea = F.pad(fea, (1, 1, 0, 0), mode='constant', value=0)
        
        #encoder
        fea = self.fea_en(fea)
        #sequence
        '''
        heart_10h = torch.concat([ppg, sig, fea], axis = 1)
        
        heart_10h = self.sequence_1(heart_10h) + heart_10h
        heart_10h = self.sequence_1(heart_10h) + heart_10h

        fea = heart_10h.permute([0, 2, 1])
        '''
        heart_10h = self.sequence(fea.permute([0, 2, 1])).permute([0, 2, 1])
        
        fea = heart_10h.permute([0, 2, 1])
        #classifier
        heart_10h = self.classifier(heart_10h)
        heart_10h = heart_10h.permute(0, 2, 1)
        return fea, heart_10h

class OurPPG_10h(nn.Module):
    def __init__(self):
        super(OurPPG_10h, self).__init__()

        #encoder
        self.ppg_en = ppg_encoder()

        self.ppg_linear = nn.Linear(256, 64)
        
        #sequence
        self.sequence = cnn_seq_block(192)
        
        #self.sequence_1 = seq_block(256)
        #self.sequence_2 = seq_block(256)

        #classifier
        self.classifier = nn.Conv1d(in_channels=192, out_channels=8, kernel_size=1, dilation=1)
    def forward(self, ppg):


        #encoder
        ppg = self.ppg_en(ppg)
        ppg = self.ppg_linear(ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1200, -1])).permute([0, 2, 1])
        
        #sequence
        heart_10h = ppg#torch.concat([ppg, sig, fea], axis = 1)
        
        #heart_10h = self.sequence_1(heart_10h) + heart_10h
        heart_10h = self.sequence(heart_10h)
        #heart_10h = self.sequence_1(heart_10h) + heart_10h

        fea = heart_10h.permute([0, 2, 1])
        #classifier
        heart_10h = self.classifier(heart_10h)
        heart_10h = heart_10h.permute(0, 2, 1)
        
        return fea, heart_10h

'''
class ALL_10h(nn.Module):
    def __init__(self):
        super(ALL_10h, self).__init__()

        
        self.time_len = 1200
        
        #encoder
        self.fea_en = fea_encoder(self.time_len)
        self.sig_en = sig_encoder(self.time_len)
        self.ppg_en = ppg_encoder(self.time_len)

        #self.ppg_linear = nn.Linear(512, 16)
        #self.fea_linear = nn.Linear(96, 16)
        #self.sig_linear = nn.Linear(80, 16)
        
        #self.ppg_linear = nn.Linear(1024, 128)
        
        #self.linear = nn.Linear(768, 384)
        
        #self.ppg_linear = nn.Linear(512, 128)
        
        #sequence
        #self.sequence_1 = self.seq_block(768)
        #self.sequence_2 = self.seq_block(768)

        #self.sequence = cnn_seq_block(32)#256, 768
        
        #self.sequence = lstm_seq_block()
        #self.sequence = attn_seq_block()
        
        #classifier
        #self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=4, kernel_size=1, dilation=1)#256, 768               64
        
        self.classifier_1 = nn.Conv1d(in_channels=16, out_channels=2, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_2 = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1, dilation=1)
        
        #self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_2 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_3 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1, dilation=1)

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(2)#64
        #self.batch_norm2 = nn.BatchNorm1d(2)#4
        
        #self.batch_norm1 = nn.BatchNorm1d(256)#64
        #self.batch_norm2 = nn.BatchNorm1d(64)
        #self.batch_norm3 = nn.BatchNorm1d(4)

        #self.batch_norm1 = nn.BatchNorm1d(4)
        
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, ppg, sig, fea, mask = None):
        
        #ppg = F.pad(ppg, (25 * 1024, 25 * 1024), mode='constant', value=0)
        #sig = F.pad(sig, (25 * 60, 25 * 60), mode='constant', value=0)
        #fea = F.pad(fea, (25 * 3, 25 * 3), mode='constant', value=0)

        #encoder
        ppg = self.ppg_en(ppg)
        #ppg = ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1200, -1]).permute([0, 2, 1])
        
        sig = self.sig_en(sig)
        
        fea = F.pad(fea, (1, 1), mode='constant', value=0)
        fea = self.fea_en(fea)

        ppg = self.ppg_linear(ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1200, -1])).permute([0, 2, 1])
        sig = self.sig_linear(sig.permute([0, 2, 1]).reshape([sig.shape[0], 1200, -1])).permute([0, 2, 1])
        fea = self.fea_linear(fea.permute([0, 2, 1]).reshape([fea.shape[0], 1200, -1])).permute([0, 2, 1])
        
        #sequence
        
        #heart_10h = torch.concat([sig, fea], axis = 1)
        #heart_10h = torch.concat([ppg, sig, fea], axis = 1)
        heart_10h = ppg + sig + fea
        
        #heart_10h = self.linear(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])
        
        #fea = torch.concat([ppg, sig, fea], axis = 0).permute([0, 2, 1])
        
        #[batch, fea, len]
        if mask != None:
            heart_10h = heart_10h * ~mask[:, :1, :]

        fea = heart_10h.permute([0, 2, 1])
        
        #heart_10h = self.sequence(heart_10h)#, cnn
        
        #heart_10h = self.sequence(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])#, lstm&attn
        #heart_10h = self.sequence_2(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])#, lstm&attn
        
        
        #classifier
        
        heart_10h = self.classifier_1(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm1(heart_10h)  # 添加标准化层

        
        #fea = heart_10h.permute(0, 2, 1)
        heart_10h = self.classifier_2(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm2(heart_10h)  # 添加标准化层
        
        heart_10h = self.classifier_3(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm3(heart_10h)  # 添加标准化层
        
        lab = heart_10h.permute(0, 2, 1)
        
        heart_10h = self.softmax( heart_10h.permute(0, 2, 1) )
        #heart_10h = heart_10h.permute(0, 2, 1)
        
        return fea, heart_10h[:, :, :], lab

'''
class ALL_10h(nn.Module):
    def __init__(self):
        super(ALL_10h, self).__init__()

        
        self.time_len = 1200
        
        #encoder
        self.fea_en = fea_encoder(self.time_len)
        
        self.sig_en = sig_encoder(self.time_len)
        
        self.ppg_en = ppg_encoder(self.time_len)

        #self.ppg_linear = nn.Linear(256, 64)
        self.ppg_linear = nn.Linear(1024, 256)
        #self.sig_linear = nn.Linear(384, 128)
        #self.fea_linear = nn.Linear(384, 128)
        
        #self.linear = nn.Linear(768, 384)
        
        #self.ppg_linear = nn.Linear(512, 128)
        
        #sequence
        #self.sequence_1 = self.seq_block(768)
        #self.sequence_2 = self.seq_block(768)

        self.sequence = cnn_seq_block(768)#256, 768
        
        #self.sequence = lstm_seq_block()
        #self.sequence = attn_seq_block(768)
        
        #classifier
        #self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=4, kernel_size=1, dilation=1)#256, 768               64
        
        self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=1, dilation=1)#256, 768               64
        self.classifier_2 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1, dilation=1)
        
        #self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_2 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_3 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1, dilation=1)

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(64)#64
        self.batch_norm2 = nn.BatchNorm1d(4)#4
        
        #self.batch_norm1 = nn.BatchNorm1d(256)#64
        #self.batch_norm2 = nn.BatchNorm1d(64)
        #self.batch_norm3 = nn.BatchNorm1d(4)

        #self.batch_norm1 = nn.BatchNorm1d(4)
        
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, ppg, sig, fea, mask = None):
        
        #ppg = F.pad(ppg, (25 * 1024, 25 * 1024), mode='constant', value=0)
        #sig = F.pad(sig, (25 * 60, 25 * 60), mode='constant', value=0)
        #fea = F.pad(fea, (25 * 3, 25 * 3), mode='constant', value=0)

        #encoder
        ppg = self.ppg_en(ppg)
        ppg = self.ppg_linear(ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1200, -1])).permute([0, 2, 1])

        sig = self.sig_en(sig)
        
        fea = F.pad(fea, (1, 1), mode='constant', value=0)
        fea = self.fea_en(fea)

        #ppg = ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1200, -1]).permute([0, 2, 1])
        
        
        
        #sig = self.sig_linear(sig.permute([0, 2, 1]).reshape([sig.shape[0], 1200, -1])).permute([0, 2, 1])
        #fea = self.fea_linear(fea.permute([0, 2, 1]).reshape([fea.shape[0], 1200, -1])).permute([0, 2, 1])
        
        #sequence
        
        heart_10h = torch.concat([ppg, sig, fea], axis = 1)#fea
        
        #heart_10h = torch.concat([ppg, fea], axis = 1)#fea
        #heart_10h = ppg + sig + fea
        
        #heart_10h = self.linear(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])
        
        #fea = torch.concat([ppg, sig, fea], axis = 0).permute([0, 2, 1])
        
        #[batch, fea, len]
        if mask != None:
            class cnn_seq_bloc = heart_10h * ~mask[:, :1, :]

        #fea = heart_10h.permute([0, 2, 1])
        
        heart_10h = self.sequence(heart_10h)#, cnn
        
        #fea = heart_10h.permute([0, 2, 1])
        
        #heart_10h = self.sequence(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])#, lstm&attn
        #heart_10h = self.sequence_2(heart_10h.permute([0, 2, 1])).pclass cnn_seq_blocermute([0, 2, 1])#, lstm&attn
        

        #fea = heart_10h.permute([0, 2, 1])
        
        #classifier
        heart_10h = self.classifier_1(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm1(heart_10h)  # 添加标准化层
        
        #fea = heart_10h.permute(0, 2, 1)
        heart_10h = self.classifier_2(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm2(heart_10h)  # 添加标准化层

        fea = heart_10h.permute([0, 2, 1])
        
        lab = heart_10h.permute(0, 2, 1)
        
        heart_10h = self.softmax( heart_10h.permute(0, 2, 1) )
        #heart_10h = heart_10h.permute(0, 2, 1)
        
        return fea, heart_10h[:, :, :], lab

class ipaskew_class(nn.Module):
    def __init__(self):
        super(ipaskew_class, self).__init__()

        self.encoder1 = nn.Sequential(#16, 128, 256, 768
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),            
            )
        self.encoder2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),            
            )
        

    def forward(self, heart_10h):

        heart_10h_1 = self.encoder1(heart_10h.permute([0, 2, 1]))[:, :, 0]
        
        heart_10h_2 = self.encoder2(heart_10h.permute([0, 2, 1]))[:, :, 0]
        
        return heart_10h_1, heart_10h_2


class ALL_10h_30(nn.Module):
    def __init__(self):
        super(ALL_10h_30, self).__init__()

        
        self.time_len = 1
        
        #encoder
        #self.fea_en = fea_encoder(self.time_len)
        self.fea_en = nn.Sequential(
            nn.Linear(28, 64),
            nn.LayerNorm(normalized_shape=(64) ),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(64, 128),
            nn.LayerNorm(normalized_shape=(128) ),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Linear(128, 256),
            nn.LayerNorm(normalized_shape=(256) ),
            nn.LeakyReLU(negative_slope=0.15),
            #en_block(en_channel[5], ou_channel[5], norm_shape[5], pool_shape[5], 17, 8),
            #en_block(en_channel[6], ou_channel[6], norm_shape[6], pool_shape[6], 17, 8),
            nn.Dropout(p = 0.1),
            #en_block(en_channel[5], ou_channel[5], norm_shape[5], pool_shape[5], 17, 8),
            #en_block(en_channel[6], en_channel[6], norm_shape[6], pool_shape[6], 1),
            #en_block(en_channel[7], en_channel[7], norm_shape[7], pool_shape[7], 1),
        )
        
        self.sig_en = sig_encoder(self.time_len)
        
        self.ppg_en = ppg_encoder(self.time_len)

        #self.ppg_linear = nn.Linear(256, 64)
        self.ppg_linear = nn.Linear(1024, 256)
        #self.sig_linear = nn.Linear(384, 128)
        #self.fea_linear = nn.Linear(384, 128)
        
        #self.linear = nn.Linear(768, 384)
        
        #self.ppg_linear = nn.Linear(512, 128)
        
        #sequence
        #self.sequence_1 = self.seq_block(768)
        #self.sequence_2 = self.seq_block(768)

        #self.sequence = cnn_seq_block(768)#256, 768
        
        #self.sequence = lstm_seq_block()
        #self.sequence = attn_seq_block(768)
        
        #classifier
        #self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=4, kernel_size=1, dilation=1)#256, 768               64
        
        self.classifier_1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dilation=1)#256, 768               64
        self.classifier_2 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1, dilation=1)
        
        #self.classifier_1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_2 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dilation=1)#256, 768               64
        #self.classifier_3 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=1, dilation=1)

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(64)#64
        self.batch_norm2 = nn.BatchNorm1d(4)#4
        
        #self.batch_norm1 = nn.BatchNorm1d(256)#64
        #self.batch_norm2 = nn.BatchNorm1d(64)
        #self.batch_norm3 = nn.BatchNorm1d(4)

        #self.batch_norm1 = nn.BatchNorm1d(4)
        
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, ppg, sig, fea, mask = None):
        
        #ppg = F.pad(ppg, (25 * 1024, 25 * 1024), mode='constant', value=0)
        #sig = F.pad(sig, (25 * 60, 25 * 60), mode='constant', value=0)
        #fea = F.pad(fea, (25 * 3, 25 * 3), mode='constant', value=0)

        #encoder
        ppg = self.ppg_en(ppg)
        ppg = self.ppg_linear(ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1, -1])).permute([0, 2, 1])

        sig = self.sig_en(sig)
        
        #fea = F.pad(fea, (1, 1), mode='constant', value=0)
        fea = self.fea_en(fea.permute([0, 2, 1])).permute([0, 2, 1])

        #ppg = ppg.permute([0, 2, 1]).reshape([ppg.shape[0], 1200, -1]).permute([0, 2, 1])
        
        
        
        #sig = self.sig_linear(sig.permute([0, 2, 1]).reshape([sig.shape[0], 1200, -1])).permute([0, 2, 1])
        #fea = self.fea_linear(fea.permute([0, 2, 1]).reshape([fea.shape[0], 1200, -1])).permute([0, 2, 1])
        
        #sequence
        
        #heart_10h = torch.concat([ppg, sig, fea], axis = 1)#fea
        heart_10h = fea#fea

        #heart_10h = torch.concat([ppg, fea], axis = 1)#fea
        #heart_10h = ppg + sig + fea
        
        #heart_10h = self.linear(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])
        
        #fea = torch.concat([ppg, sig, fea], axis = 0).permute([0, 2, 1])
        
        #[batch, fea, len]
        if mask != None:
            heart_10h = heart_10h * ~mask[:, :1, :]

        #fea = heart_10h.permute([0, 2, 1])
        
        #heart_10h = self.sequence(heart_10h)#, cnn
        
        #heart_10h = self.sequence(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])#, lstm&attn
        #heart_10h = self.sequence_2(heart_10h.permute([0, 2, 1])).permute([0, 2, 1])#, lstm&attn
        

        fea = heart_10h.permute([0, 2, 1])
        
        #classifier
        heart_10h = self.classifier_1(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm1(heart_10h)  # 添加标准化层
        
        #fea = heart_10h.permute(0, 2, 1)
        heart_10h = self.classifier_2(heart_10h)
        heart_10h = self.relu(heart_10h)  # 添加激活层
        heart_10h = self.batch_norm2(heart_10h)  # 添加标准化层

        #fea = heart_10h.permute([0, 2, 1])
        
        lab = heart_10h.permute(0, 2, 1)
        
        heart_10h = self.softmax( heart_10h.permute(0, 2, 1) )
        #heart_10h = heart_10h.permute(0, 2, 1)
        
        return fea, heart_10h[:, :, :], lab