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
            self.conv_1, self.bn_32, self.relu, #,
            self.conv_2, self.bn_64, self.relu
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
        return self.cnn(T.zeros(1, 2, *input_dims)
                       ).flatten().shape[0]


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, 
                       n_actions, c_step=300,
                       max_mem_size=2000, eps_end=0.01, eps_dec=20000, start_eps_dec=1000):
        self.gamma   = gamma
        self.epsilon = epsilon
        self.eps_max = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.start_eps_dec = start_eps_dec

        self.action_space = [i for i in range(n_actions)]
        self.mem_size   = max_mem_size
        self.batch_size = batch_size
        self.c_step     = c_step
        self.lr         = lr

        self.mem_cntr   = 0
        self.learn_cntr = 0
        self.loss_his   = []

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.q_eval = DeepQNetwork_cnn(self.lr,
                                       n_actions=n_actions, input_dims=input_dims,
                                       fc1_dims=256,
                                       fc2_dims=32).to(self.device)

        self.q_target = DeepQNetwork_cnn(self.lr,
                                         n_actions=n_actions,input_dims=input_dims,
                                         fc1_dims=256,
                                         fc2_dims=32).to(self.device)

        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        # transition: [s, a, r, s_, done]
        self.state_memory      = np.zeros((self.mem_size, 1, *input_dims), dtype=np.float32)
        self.state2_memory     = np.zeros((self.mem_size, 7)             , dtype=np.int32)
        self.action_memory     = np.zeros(self.mem_size                  , dtype=np.int32)
        self.reward_memory     = np.zeros(self.mem_size                  , dtype=np.float32)
        self.new_state_memory  = np.zeros((self.mem_size, 1, *input_dims), dtype=np.float32)
        self.new_state2_memory = np.zeros((self.mem_size, 7)             , dtype=np.int32)
        self.terminal_memory   = np.zeros(self.mem_size                  , dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx]     = state[0]
        self.state2_memory[idx]    = state[1]
        self.action_memory[idx]    = action
        self.reward_memory[idx]    = reward
        self.new_state_memory[idx] = state_[0]
        self.new_state2_memory[idx] = state_[1]
        self.terminal_memory[idx]  = done

        self.mem_cntr += 1

    @T.no_grad()
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            self.q_eval.eval()
            state   = T.tensor([observation[0]]).to(self.q_eval.device)
            state2  = T.tensor([observation[1]]).to(self.q_eval.device)
            actions = self.q_eval([state, state2], excution=True)
            action  = T.argmax(actions).item()
            ####################################
            # print("*********************")
            # print(actions, action)
            # print("*********************")
            ####################################
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        loss = None

        if self.mem_cntr >= self.batch_size:

            max_mem   = min(self.mem_cntr, self.mem_size)
            batch     = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_idx = np.arange(self.batch_size, dtype=np.int32)

            state_batch     = T.tensor(self.state_memory[batch]).to(self.q_eval.device)
            state2_batch    = T.tensor(self.state2_memory[batch]).to(self.q_eval.device)
            reward_batch    = T.tensor(self.reward_memory[batch]).to(self.q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
            new_state2_batch = T.tensor(self.new_state2_memory[batch]).to(self.q_eval.device)
            terminal_batch  = T.tensor(self.terminal_memory[batch]).to(self.q_eval.device)
            action_batch    = self.action_memory[batch]

            self.q_eval.train()
            q_eval = self.q_eval.forward([state_batch, state2_batch])[batch_idx, action_batch]
            # q_eval = q_eval[action_batch]

            with T.no_grad():
                self.q_target.eval()
                q_next = self.q_target.forward([new_state_batch, new_state2_batch])

            q_next[terminal_batch] = 0.
            q_target               = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

            # caculate the loss between q target and q eval
            loss = self.q_eval.loss(q_eval, q_target)
            self.loss_his.append(loss)

            # update the evaluation q network
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            # for param in self.q_eval.parameters():
            #     param.grad.data.clamp_(-1, 1)
            if config.clip:
                nn.utils.clip_grad_norm_(self.q_eval.parameters(), config.clip)

            tot_norm = 0; cntr = 0
            for parm in self.q_eval.parameters():
                grad_norm = parm.grad.data.norm().item()
                tot_norm += grad_norm
                cntr += 1
                print("[{}] ".format(cntr), grad_norm)
            print("avg. : ", tot_norm / cntr)


            self.q_eval.optimizer.step()

            # epsilon decay
            if self.epsilon >= self.eps_min:
                if self.learn_cntr < self.start_eps_dec:
                    self.epsilon = self.eps_max
                else:
                    self.epsilon = self.eps_min + (self.eps_max - self.eps_min) *\
                                    math.exp(-1. * (self.learn_cntr-self.start_eps_dec) / self.eps_dec)
            else:
                self.epsilon = self.eps_min

            self.learn_cntr += 1

            # replace targent network very c steps
            if self.learn_cntr % self.c_step == 0:
                self.q_target.load_state_dict(self.q_eval.state_dict())

        return loss
