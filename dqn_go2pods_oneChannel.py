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

from dqn_network_model import DeepQNetwork_cnn

T.manual_seed(999)
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()

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

        self.q_eval = DeepQNetwork(self.lr,
                                   n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=256,
                                   fc2_dims=32).to(self.device)

        self.q_target = DeepQNetwork(self.lr,
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

            # tot_norm = 0; cntr = 0
            # for parm in self.q_eval.parameters():
            #     grad_norm = parm.grad.data.norm().item()
            #     tot_norm += grad_norm
            #     cntr += 1
            #     print("[{}] ".format(cntr), grad_norm)
            # print("avg. : ", tot_norm / cntr)


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
