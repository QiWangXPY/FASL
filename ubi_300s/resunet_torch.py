import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, weight_decay=0.0, residual=True, stochastic_depth=True, activation='gelu'):
        super(conv_block, self).__init__()
        self.residual = residual
        self.stochastic_depth = stochastic_depth

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.get_padding(kernel_size), stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self.get_activation(activation)

        if residual:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=self.get_padding((1, 1))) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        if self.residual:
            if self.proj is not None:
                residual = self.proj(residual)
            if self.stochastic_depth:
                return StochasticDepth(survival_probability=0.9)([residual, x])
            else:
                return x + residual
        else:
            return x

    def get_padding(self, kernel_size):
        # 计算填充量以确保输出大小为输入大小
        return tuple((k // 2 for k in kernel_size))

    def get_activation(self, activation):
        if activation == 'gelu':
            return F.gelu
        else:
            raise ValueError("Unsupported activation function")

def determine_depth(temporal_shape, temporal_max_pool_size):

    depth = 0
    while temporal_shape % 2 == 0:
        depth += 1
        temporal_shape /= round(temporal_max_pool_size)
    depth -= 1
    return depth
    
class ResUNet(nn.Module):
    def __init__(self, input_shape, num_classes, num_outputs, depth=None, init_filter_num=8,
                 filter_increment_factor=2 ** (1 / 3), kernel_size=(16, 1), max_pool_size=(2, 1),
                 activation='gelu', output_layer='sigmoid', weight_decay=0.0,
                 residual=False, stochastic_depth=False):

        super(ResUNet, self).__init__()

        if depth is None:
            depth = determine_depth(temporal_shape=input_shape[0], temporal_max_pool_size=max_pool_size[0])

        # 这里我们用 PyTorch 的张量来表示输入
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

        # 预分配特征和其他列表
        self.features_list = []
        self.kernel_size_list = []
        self.max_pool_size_list = []

        # 其他需要的层可以在这里定义
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # Encoder
        for i in range(depth):
            self.features_list.append(features)
            self.kernel_size_list.append(kernel_size)
            self.max_pool_size_list.append(max_pool_size)

            # Feature extractor (conv_block)
            self.encoder_layers.append(conv_block(in_channels=int(features), out_channels=int(features),
                                                       kernel_size=kernel_size, activation=activation,
                                                       weight_decay=weight_decay, residual=residual, 
                                                       stochastic_depth=stochastic_depth))

            features *= filter_increment_factor

            # Convolution and Batch Normalization
            self.encoder_layers.append(nn.Conv2d(int(features), int(features), kernel_size=max_pool_size, 
                                                  stride=max_pool_size, padding=self.get_padding(max_pool_size)))
            self.encoder_layers.append(nn.BatchNorm2d(int(features)))

            # Update kernel_size and max_pool_size
            kernel_size = [min(ks, input_shape[1 if i == 0 else 2]) for ks in kernel_size]
            if input_shape[2] / max_pool_size[1] < 1:
                max_pool_size = (max_pool_size[0], 1)

        # Middle part
        self.middle_conv = conv_block(in_channels=int(features), out_channels=int(features), 
                                            kernel_size=kernel_size, activation=activation,
                                            weight_decay=weight_decay, residual=residual)

        # Decoder
        for i in reversed(range(depth)):
            self.decoder_layers.append(nn.ConvTranspose2d(self.features_list[i], self.features_list[i],
                                                           kernel_size=self.max_pool_size_list[i],
                                                           stride=self.max_pool_size_list[i], padding='same'))
            self.decoder_layers.append(nn.BatchNorm2d(self.features_list[i]))

        # Final layers
        self.final_conv1 = nn.Conv1d(init_filter_num, init_filter_num, kernel_size=1, padding='same')
        self.final_conv2 = nn.Conv1d(init_filter_num, num_classes, kernel_size=1, padding='same')
        self.classifier = nn.Linear(num_classes, num_classes)

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

        # Middle part
        x = self.middle_conv(x)

        # Decoder
        for i in range(self.depth):
            x = self.decoder_layers[2 * i](x)  # Transpose conv + BN
            x = torch.cat((skips[self.depth - 1 - i], x), dim=1)  # Concatenate with skip connection
            x = self.middle_conv(x)  # Conv block

        # Cut-off zero-padded segment
        if (zeros_to_add > 0) and (zeros_to_add / 2 == zeros_to_add // 2):
            x = x[:, zeros_to_add // 2: -zeros_to_add // 2, :, :]

        # Reshape for Conv1D
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for Conv1D

        # Non-linear activation
        x = self.final_conv1(x)
        if self.input_shape[0] // self.num_outputs > 0:
            x = F.avg_pool1d(x, kernel_size=x.size(2) // self.num_outputs)  # Average pooling if needed

        # Non-linear activation for final layer
        x = self.final_conv2(x)

        # Classification
        x = self.classifier(x)

        return x    

    def forward(self, x):
        # Zero-padding and feature extraction
        # Implement zero-padding if needed based on input shape

        # Encoder
        for i in range(self.depth):
            x = self.enc_layers[2 * i](x)  # Conv block
            self.skips[i] = x  # Save skip connection
            x = self.enc_layers[2 * i + 1](x)  # Conv + BN

        # Middle part
        x = self.middle_conv(x)

        # Decoder
        for i in range(self.depth):
            x = self.dec_layers[2 * i](x)  # Transpose conv + BN
            x = torch.cat((self.skips[self.depth - 1 - i], x), dim=1)  # Concatenate with skip connection
            x = self.middle_conv(x)  # Conv block

        # Reshape and apply final layers
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for Conv1D
        x = self.final_conv1(x)
        x = F.avg_pool1d(x, kernel_size=x.size(2) // num_outputs)  # Average pooling if needed
        x = self.final_conv2(x)

        # Classification
        x = self.classifier(x)

        return x