# -*- coding: utf-8 -*-
"""
The environment whose start pts and goal location are randomly generated
"""

from collections import defaultdict
from copy import deepcopy, copy
# from tkinter import *
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

import random
import config

UNIT = 20   # pixels
MAZE_H = config.MAP_HEIGHT #9#32  # grid height
MAZE_W = config.MAP_WIDTH #9#32  # grid width

#Nailed the hells on the fixed position
np.random.seed(999) #you can change the map with the value of the seed
random.seed(999)

class Maze(tk.Tk, object):
    def __init__(self, n_agv, n_task, maze_height=MAZE_H, maze_width=MAZE_W):
        super(Maze, self).__init__()
        # self.action_space = ['up', 'down', 'left', 'right', 'wait']
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        # self.n_features = 2
        self.task_n = n_task #1#4 #10
        self.agv_n = n_agv #1#4 #8
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.done = False
        self.title('maze')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, (MAZE_H + 3) * UNIT))
        self.geometry('{0}x{1}'.format(self.maze_width * UNIT, (self.maze_height + 5) * UNIT))
        self.map_dim = (self.maze_height, self.maze_width)
        self._build_maze()
        #statistic
        self.step_counter = 0
        # self.state = np.zeros((4, MAZE_W, MAZE_H)) # shaped in [3, W, H]
        self.state = np.zeros((2, self.maze_height, self.maze_width)) # shaped in [3, W, H]

        self.gotcha, self.boomCar, self.boomWall = 0, 0, 0
        self.testing = False
        self.seed = 999

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.maze_height * UNIT,
                           width=self.maze_width * UNIT)

        # create grids
        for columns in range(0, self.maze_width * UNIT, UNIT):
            x0, y0, x1, y1 = columns, 0, columns, self.maze_height * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for rows in range(0, self.maze_height * UNIT, UNIT):
            x0, y0, x1, y1 = 0, rows, self.maze_width * UNIT, rows
            self.canvas.create_line(x0, y0, x1, y1)

        import rmfs_layout as my_layout

        self.walls, self.wall_pos = my_layout.walls, my_layout.wall_pos
        self.froms, self.from_pos = my_layout.froms, my_layout.from_pos
        self.tos  , self.to_pos   = my_layout.tos  , my_layout.to_pos
        self.pods , self.pod_pos  = my_layout.pods , my_layout.pod_pos

        self.entrances   , self.exits           = my_layout.entrances   , my_layout.exits
        self.entrance_pos, self.exit_pos        = my_layout.entrance_pos, my_layout.exit_pos
        self.queues      , self.queue_pos       = my_layout.queues      , my_layout.queue_pos
        self.picking_pts , self.picking_pts_pos = my_layout.picking_pts , my_layout.picking_pts_pos
        self.flip_zone   , self.flip_zone_pos   = my_layout.flip_zone   , my_layout.flip_zone_pos

        # create tasks and agvs
        self.oval , self.rect      = [], []
        self.agent, self.__the_walls__ = [], []
        self.__walls__ = defaultdict(list)

        self.__pods__     , self.__picking_pts__ = [], []
        self.__entrances__, self.__exits__       = [], []
        self.__queues__   , self.__flip_zones__  = [], []
        self.finished_agv = set()
        self.loaded_agv   = set()

        ## walls
        for wall in self.walls:
            self.__the_walls__.append(self.canvas.create_rectangle(wall[0],
                                                                   wall[1],
                                                                   wall[2],
                                                                   wall[3],
                                                               fill='black'))
        ## entrance
        for entrance in self.entrances:
            self.__entrances__.append(self.canvas.create_rectangle(entrance[0], entrance[1],
                                                                   entrance[2], entrance[3],
                                                                   fill='DarkOrange2'))

        ## queues
        for queue in self.queues:
            self.__queues__.append(self.canvas.create_rectangle(queue[0], queue[1],
                                                                queue[2], queue[3],
                                                                fill='PaleGreen1'))

        ## flip zone
        for flip_zone in self.flip_zone:
            self.__flip_zones__.append(self.canvas.create_rectangle(flip_zone[0], flip_zone[1],
                                                                    flip_zone[2], flip_zone[3],
                                                                    fill='peach puff'))

        ## picking points
        for picking_pts in self.picking_pts:
            self.__picking_pts__.append(self.canvas.create_rectangle(picking_pts[0], picking_pts[1],
                                                                     picking_pts[2], picking_pts[3],
                                                                     fill='tomato2'))
        ## exit
        for exit in self.exits:
            self.__exits__.append(self.canvas.create_rectangle(exit[0], exit[1],
                                                               exit[2], exit[3],
                                                               fill='gold2'))
        ## pods
        for pod in self.pods:
            self.__pods__.append(self.canvas.create_rectangle(pod[0], pod[1],
                                                              pod[2], pod[3],
                                                              fill='deep sky blue'))

        # self.froms   , self.tos      = [], []
        # self.from_pos, self.to_pos   = [], []
        self.froms   , self.tos      = [], defaultdict(list)
        self.from_pos, self.to_pos   = [], defaultdict(list)
        self.need_pods     = defaultdict(list)
        self.task_priority = defaultdict(list)

        self.request_ws, self.request_ws_pos = [], []
        self.request_ws_idx                  = []
        origin = np.array([10, 10])
        # to create tasks(find pods)
        for num_task in range(config.n_task):
            belong_to_agv = num_task % self.agv_n
            the_pod_idx = np.random.choice(range(len(self.pod_pos)))
            the_pod     = self.pod_pos[the_pod_idx]
            while the_pod in self.to_pos.values():
                the_pod_idx = np.random.choice(range(len(self.pod_pos)))
                the_pod     = self.pod_pos[the_pod_idx]
            x, y   = the_pod[0], the_pod[1]
            center = origin + np.array([UNIT * x, UNIT * y])
            self.tos[belong_to_agv].append([center[0] - 7, center[1] - 7,
                                            center[0] + 7, center[1] + 7])
            self.to_pos[belong_to_agv].append([x, y])

            the_ws_idx = np.random.choice(range(2))
            the_ws     = self.entrance_pos[the_ws_idx]
            x, y       = the_ws[0], the_ws[1]
            center     = origin + np.array([UNIT * x, UNIT * y])
            # self.request_ws.append([center[0] - 7, center[1] - 7,
            #                         center[0] + 7, center[1] + 7])
            # self.request_ws_pos.append([x, y])
            self.request_ws_idx.append(the_ws_idx)
            self.tos[belong_to_agv].append([center[0] - 7, center[1] - 7,
                                            center[0] + 7, center[1] + 7])
            self.to_pos[belong_to_agv].append([x, y])
            self.need_pods[belong_to_agv].append(self.__pods__[the_pod_idx])

            the_priority = np.random.choice(range(self.task_n))
            the_priority = the_priority / self.task_n
            self.task_priority[belong_to_agv].append(the_priority)

        # to create agvs
        # the_agv = [[6, 7]]
        for _ in range(config.n_agv):
            x = np.random.randint(1, 17)#(maze_width - 1))
            y = np.random.randint(1, (self.maze_height - 1))
            while [x, y] in self.wall_pos or [x, y] in self.from_pos:
                x = np.random.randint(1, 17)#(maze_width - 1))
                y = np.random.randint(1, (self.maze_height - 1))
            # x, y = the_agv[_][0], the_agv[_][1]
            center = origin + np.array([UNIT * x, UNIT * y])

            self.froms.append([center[0] - 7, center[1] - 7,
                               center[0] + 7, center[1] + 7])
            self.from_pos.append([x, y])

        ## agvs(rect) n tasks(oval)
        for agv_id in range(self.agv_n):
            color = ('yellow' if agv_id==(self.agv_n-1) else 'orange')
            self.oval.append(self.canvas.create_oval(self.tos[agv_id][0][0],
                                                     self.tos[agv_id][0][1],
                                                     self.tos[agv_id][0][2],
                                                     self.tos[agv_id][0][3],
                                                     fill=color))

            if agv_id == (self.agv_n - 1):
                self.agent = self.canvas.create_rectangle(self.froms[0][0], self.froms[0][1],
                                                          self.froms[0][2], self.froms[0][3],
                                                          fill='red')
                self.rect.append(self.agent)
            else:
                self.rect.append(self.canvas.create_rectangle(self.froms[0][0], self.froms[0][1],
                                                              self.froms[0][2], self.froms[0][3],
                                                              fill='green'))
            self.tos[agv_id] = self.tos[agv_id][1:]
            self.froms       = self.froms[1:]

        # pack all
        self.canvas.pack()
        self.n_wait = 0

    def reset(self, episode=0):
        np.random.seed(self.seed)
        if episode % 3 == 0:
            self.seed += 1
        self.done = False
        self.canvas.destroy()
        self._build_maze()
        self.gotcha, self.boomCar, self.boomWall = 0, 0, 0
        self.n_wait = 0

        self.agent_map   = np.zeros((self.maze_height, self.maze_width))
        self.task_map    = np.zeros((self.maze_height, self.maze_width))
        self.others_map  = np.zeros((self.maze_height, self.maze_width))
        self.loaded_map  = np.zeros((self.maze_height, self.maze_width))
        self.visited_map = np.zeros((self.maze_height, self.maze_width))
        self.obs_map_og  = np.zeros((self.maze_height, self.maze_width))
        self.pods_map_og = np.zeros((self.maze_height, self.maze_width))
        self.pods_map    = np.zeros((self.agv_n, self.maze_height, self.maze_width))
        self.obs_map     = np.zeros((self.agv_n, self.maze_height, self.maze_width))

        return self.get_state()

    def get_state(self, agent_id=(config.n_agv - 1)):
        # Position of learning agent
        next_coords = self.canvas.coords(self.rect[agent_id])
        agent_pos_x = int((next_coords[0] - 3) / UNIT)
        agent_pos_y = int((next_coords[1] - 3) / UNIT)
        self.agent_map[agent_pos_y][agent_pos_x] = 1.
        if agent_id in self.loaded_agv:
            self.loaded_map[agent_pos_y][agent_pos_x] = self.task_priority[agent_id][0]
        # Position of task
        task_coords = self.canvas.coords(self.oval[agent_id])
        task_pos_x = int((task_coords[0] - 3) / UNIT)
        task_pos_y = int((task_coords[1] - 3) / UNIT)
        self.task_map[task_pos_y][task_pos_x] = 0.75

        # Position of other agents
        for others in self.rect:
            if others == self.rect[agent_id]:
                continue
            others_coords = self.canvas.coords(others)
            others_pos_x = int((others_coords[0] - 3) / UNIT)
            others_pos_y = int((others_coords[1] - 3) / UNIT)
            # self.others_map[others_pos_y][others_pos_x] = 1.
            self.others_map[others_pos_y][others_pos_x] = 0.25

        # Position of walls
        for walls in self.__the_walls__:
            walls_coords = self.canvas.coords(walls)
            walls_pos_x = int((walls_coords[0] - 3) / UNIT)
            walls_pos_y = int((walls_coords[1] - 3) / UNIT)
            self.obs_map_og[walls_pos_y][walls_pos_x] = 0.5

        for walls in self.__picking_pts__: #workstation might be the obs, too!
            walls_coords = self.canvas.coords(walls)
            walls_pos_x = int((walls_coords[0] - 3) / UNIT)
            walls_pos_y = int((walls_coords[1] - 3) / UNIT)
            self.obs_map_og[walls_pos_y][walls_pos_x] = 0.5
        for walls in self.__flip_zones__: #workstation might be the obs, too!
            walls_coords = self.canvas.coords(walls)
            walls_pos_x = int((walls_coords[0] - 3) / UNIT)
            walls_pos_y = int((walls_coords[1] - 3) / UNIT)
            self.obs_map_og[walls_pos_y][walls_pos_x] = 0.5
        for walls in self.__queues__: #workstation might be the obs, too!
            walls_coords = self.canvas.coords(walls)
            walls_pos_x = int((walls_coords[0] - 3) / UNIT)
            walls_pos_y = int((walls_coords[1] - 3) / UNIT)
            self.obs_map_og[walls_pos_y][walls_pos_x] = 0.5
        for walls in self.__exits__: #workstation might be the obs, too!
            walls_coords = self.canvas.coords(walls)
            walls_pos_x = int((walls_coords[0] - 3) / UNIT)
            walls_pos_y = int((walls_coords[1] - 3) / UNIT)
            self.obs_map_og[walls_pos_y][walls_pos_x] = 0.5
        self.__the_walls__ = self.__the_walls__ + self.__queues__ +\
                             self.__picking_pts__ + self.__flip_zones__+\
                             self.__exits__
        for agv_id in range(self.agv_n):
            self.__walls__[agv_id] = deepcopy(self.__the_walls__)
            self.obs_map[agv_id]   = np.array(self.obs_map_og)

        for the_pod in self.__pods__:
            pod_coords = self.canvas.coords(the_pod)
            pod_pos_x = int((pod_coords[0] - 3) / UNIT)
            pod_pos_y = int((pod_coords[1] - 3) / UNIT)
            self.pods_map_og[pod_pos_y][pod_pos_x] = 0.5

        self.agent_map_og  = np.array(self.agent_map)
        self.others_map_og = np.array(self.others_map)
        self.task_map_og   = np.array(self.task_map)
        self.loaded_map_og = np.array(self.loaded_map)
        # self.visited_map_og = np.zeros((self.maze_width, self.maze_height))

        self.state[0] = self.agent_map_og + self.others_map_og + self.task_map_og + self.obs_map[-1] + self.pods_map[-1]
        self.state[1] = self.loaded_map_og

        # terminal condition
        if np.array(self.oval).sum() == 0:
            self.done = True

        x_direction = agent_pos_x - task_pos_x
        y_direction = agent_pos_y - task_pos_y
        xy_distance = (x_direction**2 + y_direction**2)**(1 / 2.0)
        self.state_fcn = np.array([agent_pos_x, agent_pos_y, task_pos_x, task_pos_y, x_direction, y_direction, xy_distance])#,
                                   # self.task_priority[self.agv_n-1][0])

        return np.array([self.state, self.state_fcn])

    def move_agent(self, agent, action, learning_mode=True):
        agent_pos   = self.canvas.coords(agent)
        base_action = np.array([0, 0])

        self.penalty = False

        # UP #
        if action == 0:
            if agent_pos[1] > UNIT:# * 2:
                base_action[1] -= UNIT
                if learning_mode != False:
                    self.n_wait = 0
            else:
                base_action = np.array([0, 0])
                if learning_mode != False:
                    self.penalty = True

        # DOWN #
        elif action == 1:
            if agent_pos[1] < (self.maze_height - 1) * UNIT:# -1) * UNIT:
                base_action[1] += UNIT
                if learning_mode != False:
                    self.n_wait = 0
            else:
                base_action = np.array([0, 0])
                if learning_mode != False:
                    self.penalty = True

        # RIGHT #
        elif action == 2:
            if agent_pos[0] < (self.maze_width - 1) * UNIT: # -1) * UNIT:
                base_action[0] += UNIT
                if learning_mode != False:
                    self.n_wait = 0
            else:
                base_action = np.array([0, 0])
                if learning_mode != False:
                    self.penalty = True

        # LEFT #
        elif action == 3:
            if agent_pos[0] > UNIT:# * 2:
                base_action[0] -= UNIT
                if learning_mode != False:
                    self.n_wait = 0
            else:
                base_action = np.array([0, 0])
                if learning_mode != False:
                    # self.n_wait += 1
                    self.penalty = True

        # WAIT #
        elif action == 4:
            base_action = np.array([0, 0])
            if learning_mode != False:
                self.n_wait += 1

        # Start to move
        self.canvas.move(agent, base_action[0], base_action[1])

        # if learning_mode:
        #     self.last_action = base_action
        if True:
            self.last_action[agent] = base_action

        # make sure other agvs won't bump into the learning agent
        agent_pos = self.canvas.coords(agent)
        if learning_mode == False:
            if agent_pos == self.canvas.coords(self.agent):
                self.canvas.move(agent, ((-1) * base_action[0]), ((-1) * base_action[1])) # back 2 last pos

    def find_task(self, agent, agent_id, learning_mode=False):
        if learning_mode == True:
            self.reward = 0

        # Next state infomation about the position of agent
        if learning_mode:
            next_coords = deepcopy(self.canvas.coords(agent))
            task_coords = deepcopy(self.canvas.coords(self.oval[self.agv_n - 1]))
        else:
            next_coords = deepcopy(self.canvas.coords(self.rect[agent_id]))
            task_coords = deepcopy(self.canvas.coords(self.oval[agent_id]))

        #### Reward function ###
        reach_target = (next_coords == self.canvas.coords(self.oval[agent_id]))
        if not reach_target:
            # if learning_mode:
            if True:
                # bump into others
                other_avgs_position = [self.canvas.coords(agvs) for agvs in self.rect if agvs != self.rect[agent_id]]
                walls_location = [self.canvas.coords(walls) for walls in self.__walls__[agent_id]]

                if next_coords in other_avgs_position:
                    if learning_mode:
                        self.reward = config.crash_penalty #-1.

                        if self.testing:
                            self.done = False #True
                            self.boomCar = 1
                        else:
                            self.done = False

                    if self.done == False:
                        self.canvas.move(agent, ((-1) * self.last_action[agent][0]), ((-1) * self.last_action[agent][1])) # back 2 last pos

                # bump into walls(obstacle)
                elif next_coords in walls_location:
                    if learning_mode:
                        self.reward = config.crash_penalty #-10000. #00
                        self.done = False #True
                        self.boomWall = 1
                    if self.done == False:
                        self.canvas.move(agent, ((-1) * self.last_action[agent][0]), ((-1) * self.last_action[agent][1])) # back 2 last pos

                # bump into the wall(layout)
                elif self.penalty:
                    if learning_mode:
                        self.reward = config.crash_penalty #-10000.
                        self.done = False
                    if self.done == False:
                        self.canvas.move(agent, ((-1) * self.last_action[agent][0]), ((-1) * self.last_action[agent][1])) # back 2 last pos

                # wait penalty
                elif self.n_wait:
                    self.reward = config.wait_penalty * self.n_wait #-0.1 * self.n_wait#-10#-0.1
                    self.done = False

                else:
                    self.reward = config.step_penalty
                    self.done = False

        # Got the task
        else:
            if learning_mode == True:
                self.reward = config.goal_reward #100. #1. #00
                self.gotcha = 1
                # print(""); print("********** Good Job ***********"); print()

            # if agent_id < self.agv_n - 1:
            self.canvas.delete(self.oval[agent_id])

            n_task_remain = len(self.tos[agent_id])
            if n_task_remain != 0:
                color = ('yellow' if agent_id==(self.agv_n-1) else 'orange')
                self.oval[agent_id] = self.canvas.create_oval(
                                                   self.tos[agent_id][0][0],
                                                   self.tos[agent_id][0][1],
                                                   self.tos[agent_id][0][2],
                                                   self.tos[agent_id][0][3],
                                                   fill=color)
                self.tos[agent_id] = self.tos[agent_id][1:]
                self.froms         = self.froms[1:]
                self.pods_map[agent_id] = np.array(self.pods_map_og)
                if n_task_remain % 2:
                    finished_pod = self.need_pods[agent_id].pop(0)
                    self.canvas.delete(finished_pod)
                    # print('### check check ###')
                    # print(self.__walls__[agent_id])
                    # print(self.pods_map[agent_id])
                    self.__walls__[agent_id] += self.__pods__
                    self.obs_map[agent_id]   += self.pods_map[agent_id]
                    self.loaded_agv.add(agent_id)
                else:
                    self.finished_agv.add(agent_id)
                    self.loaded_agv.remove(agent_id)
                    self.canvas.move(self.rect[agent_id], 60, 0)
                    self.canvas.move(self.rect[agent_id], 0, 40)
                    self.task_priority[agent_id] = self.task_priority[agent_id][1:]
            else:
                self.finished_agv.add(agent_id)
                self.loaded_agv.remove(agent_id)
                self.canvas.move(self.rect[agent_id], 60, 0)
                self.canvas.move(self.rect[agent_id], 0, 40)
                self.task_priority[agent_id] = self.task_priority[agent_id][1:]
                self.oval[agent_id] = 0
                self.task_priority[agent_id] = [0]

            # terminal condition
            if np.array(self.oval).sum() == 0:
                self.done = True

        #### Next state ###
        next_coords = self.canvas.coords(agent)
        agent_pos_x = int((next_coords[0] - 3) / UNIT)
        agent_pos_y = int((next_coords[1] - 3) / UNIT)
        # print(agent_pos_x, agent_pos_y)

        task_pos_x = int((task_coords[0] - 3) / UNIT)
        task_pos_y = int((task_coords[1] - 3) / UNIT)

        # Position of task
        if learning_mode:
            self.task_map[task_pos_y][task_pos_x] = 0.75
        # Position of learning agent
        if agent_id == (self.agv_n - 1):
            self.agent_map[agent_pos_y][agent_pos_x] = 1.
            self.visited_map[agent_pos_y][agent_pos_x] = 0.05
        # Position of other agents
        else:
            self.others_map[agent_pos_y][agent_pos_x] = 0.25

        if agent_id in self.loaded_agv:
            self.loaded_map[agent_pos_y][agent_pos_x] = self.task_priority[agent_id][0]

        x_direction = agent_pos_x - task_pos_x
        y_direction = agent_pos_y - task_pos_y
        xy_distance = (x_direction**2 + y_direction**2)**(1 / 2.0)
        self.state_fcn = np.array([agent_pos_x, agent_pos_y, task_pos_x, task_pos_y, x_direction, y_direction, xy_distance])#,
                                   # self.task_priority[self.agv_n-1][0]])

        if learning_mode:
            self.reward -= xy_distance

    def step(self, action):
        self.last_action = {}
        self.agent_map  = np.zeros((self.maze_height, self.maze_width))
        self.others_map = np.zeros((self.maze_height, self.maze_width))
        self.task_map   = np.zeros((self.maze_height, self.maze_width))
        self.loaded_map = np.zeros((self.maze_height, self.maze_width))
        self.pods_map   = np.zeros((self.agv_n, self.maze_height, self.maze_width))
        self.step_counter += 1

        learning_agent = self.rect[self.agv_n - 1]
        if learning_agent != 0 and (self.agv_n - 1) not in self.finished_agv:
            self.move_agent(learning_agent, action)
            self.find_task(learning_agent , (self.agv_n - 1), learning_mode=True)

        if self.done != True:
            for agent_id in range(self.agv_n - 1):
                if agent_id in self.finished_agv:
                    continue
                action = random.randint(0, self.n_actions)
                # action = 4
                self.move_agent(self.rect[agent_id], action  , learning_mode=False)
                self.find_task(self.rect[agent_id] , agent_id, learning_mode=False)

        # self.state[0] = self.agent_map_og + self.others_map_og + self.task_map_og + self.obs_map
        self.state[0] = self.agent_map + self.others_map + self.task_map + self.obs_map[-1] + self.pods_map[-1]
        self.state[1] = np.array(self.loaded_map)

        # print("")
        # print("================= New state ===================")
        # print(self.state[0])
        # print("")
        # print(self.reward, self.done)
        # print(self.get_state())
        # print(self.state_fcn.shape)

        return np.array([self.state, self.state_fcn]), self.reward, self.done


    def render(self, motion_speed=None):
        if motion_speed != None:
            time.sleep(motion_speed)#0.01
        self.update()

    def nothing(self):
        print("nothing")


def run(slow_motion):
    np.set_printoptions(linewidth=1000)
    b  = tk.Button(env, text="Let's start it!", command=env.nothing)
    b1 = tk.Button(env, text="Come on!"       , command=env.nothing)
    for episode in range(5):
        # np.random.seed(episode)
        # env.canvas.create_text(MAZE_H * UNIT + 20, MAZE_H * UNIT + 20, text = "WOOOOOOOOW", font = ("arial", 18), fill = "red")
        b.destroy()
        b = tk.Button(env,text='Episode: {}'.format(episode), command=env.nothing)
        b.pack()
        print()
        print("Episode {}: ".format(episode))
        print()
        state     = env.reset(episode)
        done      = False
        step_cntr = 0
        while not done:
            b1.destroy()
            b1 = tk.Button(env,text='Step: {}'.format(step_cntr), command=env.nothing)
            b1.pack()
            env.render(slow_motion)

            action = np.random.randint(4)
            state_, reward, done = env.step(action)
            print(state)
            print(action, reward, done)
            print()
            state      = state_
            step_cntr += 1

    # env.destroy()
    # env.reset()
    # env.after(1000, run)

if __name__ == '__main__':
    gui_mode = input("Would you like to open the GUI? [Y/n]")
    slow_mo = input("Would you like to slow down the animation? [Y/n]")
    if slow_mo.lower() == 'y':
        motion_speed = float(input('Control the speed in __ ? [0.1]'))
    else:
        motion_speed = None
    # Create the env
    env = Maze(n_agv=config.n_agv,
               n_task=config.n_task,
               maze_width=config.MAP_WIDTH,
               maze_height=config.MAP_HEIGHT)


    # Check the animation mode
    if gui_mode.lower() != 'y':
        env.withdraw()

    # Run the env
    env.after(1000, run(motion_speed))
    env.mainloop()
