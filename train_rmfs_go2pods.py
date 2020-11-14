#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

'''
The version for single agv start pts and goal location are randomly generated
one channel, 2 conv layers
'''

import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from tensorboardX import SummaryWriter

# from dqn_cnn_model_oneChannel_fcn_5Actions import Agent
from dqn_rmfs_go2pods import Agent
# from env_rmfs_go2pods import Maze
from go2pods_2ws import Maze

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#params
import config
PRETRAINED = None# config.PRETRAINED #None # False # True #
EPISODE_MAX = config.EPISODE_MAX #10000

env = Maze(
        n_agv=config.n_agv,
        n_task=config.n_task,
        maze_height=config.MAP_HEIGHT,
        maze_width=config.MAP_WIDTH
        )

b  = tk.Button(env, text="Let's start it!", command=env.nothing)
b1 = tk.Button(env, text="Let's start it!", command=env.nothing)
# b2    = tk.Button(env, text="'s start it!", command=env.nothing)
b.pack()
b1.pack()
# b2.pack()

agent = Agent(
            lr=config.lr,
            gamma=config.gamma,
            batch_size=config.batch_size,
            max_mem_size=config.max_mem_size,
            c_step=config.c_step,
            epsilon=config.epsilon,
            eps_end=config.eps_end,
            eps_dec=config.eps_dec,
            start_eps_dec=config.start_eps_dec,
            n_actions=env.n_actions,
            input_dims=env.map_dim
        )

if PRETRAINED != None:
    agent.q_eval.load_state_dict(torch.load(PRETRAINED))
    agent.q_target.load_state_dict(agent.q_eval.state_dict())
    agent.q_target.eval()

with SummaryWriter() as writer:

    n_gotcha, n_boomCar, n_boomWall = 0, 0, 0
    lstReward, lstPerform = [], []
    lstGotcha, lstBoomCar, lstBoomWall = [], [], []

    for episode in tqdm(range(EPISODE_MAX)):
        b.destroy()
        b  = tk.Button(env, text='Episode: {}'.format(episode), command=env.nothing)
        b.pack()

        b1.destroy()
        b1 = tk.Button(env, text='Epsilon: %.3f'%agent.epsilon, command=env.nothing)
        b1.pack()

        total_reward, num_tookStep = 0, 0
        state = env.reset(episode)
        done  = False

        while not done:
            # env.render((0.05 if agent.epsilon <= 0.1 else None))
            env.render()

            #select an action
            action = agent.choose_action(state)
            state_, reward, done = env.step(action)

            total_reward += reward
            num_tookStep += 1

            threshold = (500 if episode >= 300 else 1500)
            if num_tookStep >= threshold and not done:
                done = True
            else:
                agent.store_transition(state, action, reward, state_, done)
                if done and reward > 0:
                    for _ in range(10):
                        agent.store_transition(state, action, reward, state_, done)

                #perform one step of the training
                loss = agent.learn()

                if loss and (agent.learn_cntr % 100 == 0):
                    writer.add_scalar(
                        "Loss/train", loss, agent.learn_cntr
                    )

                #transit to new state
                state = state_

                if env.done:
                    lstReward.append(total_reward)
                    lstPerform.append(num_tookStep)
                    lstGotcha.append(env.gotcha)
                    lstBoomCar.append(env.boomCar)
                    lstBoomWall.append(env.boomWall)
                    n_gotcha += env.gotcha
                    n_boomCar += env.boomCar
                    n_boomWall += env.boomWall

        if env.done:
            writer.add_scalar(
                "reward/episode"  , total_reward, episode
                )
            writer.add_scalar(
                "tookStep/episode", num_tookStep, episode
                )
            writer.add_scalar(
                "exporation/episode", agent.epsilon, episode
            )

        if episode % 1000 == 0 and episode != 0:
            # torch.save(agent.q_eval.state_dict(), "model/waitPenalty-2car-wall-test-QevalNet-pooling-{}.pkl".format(episode))
            torch.save(
                agent.q_eval.state_dict(), "model/1Channel-fcn-1car-wall-randomGoal-{}.pkl".format(episode)
                )

    env.destroy()
    torch.save(agent.q_eval.state_dict(), "model/1Channel-fcn-1car-wall-randomGoal-final.pkl")

    # Result
    dfReward   = pd.DataFrame(lstReward)
    dfPerform  = pd.DataFrame(lstPerform)
    dfGotcha   = pd.DataFrame(lstGotcha)
    dfBoomCar  = pd.DataFrame(lstBoomCar)
    dfBoomWall = pd.DataFrame(lstBoomWall)

    dfReward.columns   = ["Reward"]
    dfPerform.columns  = ["Steps to arrive"]
    dfGotcha.columns   = ["Gotcha"]
    dfBoomCar.columns  = ["BoomCar"]
    dfBoomWall.columns = ["BoomWall"]

    result = pd.concat([dfReward, dfPerform, dfGotcha, dfBoomCar, dfBoomWall], axis = 1)
    result.to_csv("result/1Channel-fcn-1car-wall-randomGoal-result-{}.csv".format(77))

    # Loss
    dfLossHis = pd.DataFrame([agent.loss_his[idx].data.item() for idx in range(len(agent.loss_his))])
    dfLossHis.to_csv("result/1Channel-fcn-1car-wall-randomGoal-loss-his-{}.csv".format(77))

    # Statistic
    lst_success = [n_gotcha, n_boomCar, n_boomWall]
    df_success = pd.DataFrame(lst_success)
    df_success.to_csv("result/1Channel-fcn-1car-wall-randomGoal-result-success-{}.csv".format(77))

    writer.flush()

    plt.figure()
    plt.plot(range(len(agent.loss_his)), agent.loss_his)
    plt.title("Loss - LearningStep")

    plt.figure()
    plt.plot(range(len(lstReward)), lstReward)
    plt.title("Reward - Episode")

    plt.figure()
    plt.plot(range(len(lstPerform)), lstPerform)
    plt.title("StepsToGoal - Episode")
    plt.show()
