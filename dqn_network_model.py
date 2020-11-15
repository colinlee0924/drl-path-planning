#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import config

def __reset_param_impl__(dqn_cnn):
    """
    """
    # --- do init ---
    raise NotImplementedError

T.manual_seed(999)
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()

class DeepQNetwork_cnn(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, bias=True):
        super().__init__()
        self.lr = lr
        self.n_outputs = n_actions
        self.actions = np.arange(n_actions)

        # convolution layers
        self.conv_1  = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv_2  = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # self.conv_3  = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn_32   = nn.BatchNorm2d(32)
        self.bn_64   = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()
        n1 = self.conv_1.kernel_size[0] * self.conv_1.kernel_size[1] * self.conv_1.out_channels
        n2 = self.conv_2.kernel_size[0] * self.conv_2.kernel_size[1] * self.conv_2.out_channels
        # n3 = self.conv_3.kernel_size[0] * self.conv_3.kernel_size[1] * self.conv_3.out_channels
        self.conv_1.weight.data.normal_(0, math.sqrt(2. / n1))
        self.conv_2.weight.data.normal_(0, math.sqrt(2. / n2))
        # self.conv_3.weight.data.normal_(0, math.sqrt(2. / n3))


        self.cnn = nn.Sequential(
            self.conv_1, self.bn_32, self.relu, #nn.MaxPool2d((4, 4)),#,
            self.conv_2, self.bn_64, self.relu, nn.MaxPool2d((2, 2))
            # self.conv_3, self.bn_64, self.relu
        )

        # check the output of cnn, which is [fc1_dims]
        self.cnn_outputs_length = self.cnn_out_dim(input_dims)
        self.fc_inputs_length = self.cnn_outputs_length + 7

        # fully connected layers
        # self.fc1 =  nn.Linear(self.fc_inputs_length, 128)
        self.fc1 =  nn.Linear(self.fc_inputs_length, 256)
        self.fc3 =  nn.Linear(256, 128)
        self.fc2 =  nn.Linear(128, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

        self.fc = nn.Sequential(
            self.fc1, self.relu,
            self.fc3, self.relu,
            self.fc2
        )

        # check the cuda device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # optimizer
        self.loss      = nn.MSELoss() #  nn.L1Loss() #
        if config.optim == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, state, excution=False):
        cnn_input   = state[0].to(device=self.device, dtype=T.float)
        fc_features = state[1].to(device=self.device, dtype=T.float)

        cnn_out   = self.cnn(cnn_input)
        # if excution:
        #     print(cnn_out)
        cnn_out   = cnn_out.reshape(-1, self.cnn_outputs_length)
        cnn_out   = self.flatten(cnn_out)
        fcn_input = T.cat([cnn_out, fc_features], 1)
        actions = self.fc(fcn_input)
        return actions

    def reset_param(self):
        __reset_param_impl__(self)

    def cnn_out_dim(self, input_dims):
        return self.cnn(T.zeros(1, 1, *input_dims)
                       ).flatten().shape[0]

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, bias=True):
        super().__init__()
        self.lr = lr
        self.n_outputs = n_actions
        self.actions = np.arange(n_actions)

        # fully connected layers
        # self.fc1 =  nn.Linear(self.fc_inputs_length, 128)
        self.fc1 =  nn.Linear(7, 256)
        self.fc3 =  nn.Linear(256, 128)
        self.fc2 =  nn.Linear(128, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

        self.fc = nn.Sequential(
            self.fc1, self.relu,
            self.fc3, self.relu,
            self.fc2
        )

        # check the cuda device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # optimizer
        self.loss      = nn.MSELoss() #  nn.L1Loss() #
        if config.optim == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, state, excution=False):
        fcn_input = state.to(device=self.device, dtype=T.float)
        actions = self.fc(fcn_input)
        return actions

    def reset_param(self):
        __reset_param_impl__(self)
