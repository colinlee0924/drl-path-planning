from collections import defaultdict
from copy import deepcopy, copy
# -*- coding: utf-8 -*-
"""
The layout of rmfs
"""

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

UNIT   = config.UNIT #20   # pixels
MAZE_H = config.MAP_HEIGHT#32  # grid height
MAZE_W = config.MAP_WIDTH#32  # grid width
maze_height = config.MAP_HEIGHT#32  # grid height
maze_width  = config.MAP_WIDTH#32  # grid width

#Nailed the hells on the fixed position
np.random.seed(999) #you can change the map with the value of the seed
random.seed(999)

###############
### Layout ####
###############

n_ws   = 2      # num of workstations
n_pods = 71     # num of pods

# create origin
origin = np.array([10, 10])

froms   , tos      = [], []
from_pos, to_pos   = [], []
pods    , pod_pos  = [], []
request_ws_idx     = []

walls    , wall_pos      = [], []
entrances, entrance_pos  = [], []
exits    , exit_pos      = [], []
queues   , queue_pos     = [], []
flip_zone, flip_zone_pos = [], []
picking_pts              = []

# to create the layout
the_wall     = [[1, 1], [1, 9]]
the_entrance = [[17, 1], [17, 6]]
the_exit     = [[17, 4], [17, 9]]
queue_1      = [[18, 1], [19, 1], [20, 1], [20, 2],
                [18, 4], [19, 4], [20, 4]]
queue_2      = [[18, 6], [19, 6], [20, 6], [20, 7],
                [18, 9], [19, 9], [20, 9]]

queue_pos       = queue_1 + queue_2
picking_pts_pos = [[20, 3], [20, 8]]

flip_zone_pos_1 = [[18, 2], [19, 2],
                   [18, 3], [19, 3]]
flip_zone_pos_2 = [[18, 7], [19, 7],
                   [18, 8], [19, 8]]
flip_zone_pos   = flip_zone_pos_1 + flip_zone_pos_2

the_pods_top  = [[x + 2, 1] for x in range(12)]
the_pods_bot  = [[x + 2, 9] for x in range(12)]
the_pods_left = [[1, y + 2] for y in range(7)]
the_pods_mid_top1 = [[x + 3, 3] for x in range(5)] +\
                    [[x + 9, 3] for x in range(5)]
the_pods_mid_top2 = [[x + 3, 4] for x in range(5)] +\
                    [[x + 9, 4] for x in range(5)]
the_pods_mid_bot1 = [[x + 3, 6] for x in range(5)] +\
                    [[x + 9, 6] for x in range(5)]
the_pods_mid_bot2 = [[x + 3, 7] for x in range(5)] +\
                    [[x + 9, 7] for x in range(5)]
the_pods = the_pods_bot + the_pods_top + the_pods_left +\
           the_pods_mid_top1 + the_pods_mid_top2 +\
           the_pods_mid_bot1 + the_pods_mid_bot2

for x_y in the_entrance:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])

    entrances.append([center1[0] - 7, center1[1] - 7,
                      center1[0] + 7, center1[1] + 7])
    entrance_pos.append([x, y])

for x_y in the_exit:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])

    exits.append([center1[0] - 7, center1[1] - 7,
                 center1[0] + 7, center1[1] + 7])
    exit_pos.append([x, y])

for x_y in picking_pts_pos:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])
    picking_pts.append([center1[0] - 7, center1[1] - 7,
                        center1[0] + 7, center1[1] + 7])

for x_y in flip_zone_pos:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])
    flip_zone.append([center1[0] - 7, center1[1] - 7,
                      center1[0] + 7, center1[1] + 7])

for x_y in queue_pos:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])
    queues.append([center1[0] - 7, center1[1] - 7,
                   center1[0] + 7, center1[1] + 7])

for x_y in the_wall:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])

    walls.append([center1[0] - 7, center1[1] - 7,
                  center1[0] + 7, center1[1] + 7])
    wall_pos.append([x, y])

# to create the walls(layout)
for x in range(maze_width):
    y1, y2 = 0, (maze_height - 1)
    center1 = origin + np.array([UNIT * x, UNIT * y1])
    center2 = origin + np.array([UNIT * x, UNIT * y2])

    walls.append([center1[0] - 7, center1[1] - 7,
                  center1[0] + 7, center1[1] + 7])
    walls.append([center2[0] - 7, center2[1] - 7,
                  center2[0] + 7, center2[1] + 7])
    wall_pos.append([x, y1])
    wall_pos.append([x, y2])

for y in range(maze_height):
    x1, x2 = 0, (maze_width - 1)
    center1 = origin + np.array([UNIT * x1, UNIT * y])
    center2 = origin + np.array([UNIT * x2, UNIT * y])

    walls.append([center1[0] - 7, center1[1] - 7,
                  center1[0] + 7, center1[1] + 7])
    walls.append([center2[0] - 7, center2[1] - 7,
                  center2[0] + 7, center2[1] + 7])
    wall_pos.append([x1, y])
    wall_pos.append([x2, y])

for x_y in the_pods:
    x, y = x_y[0], x_y[1]
    center1 = origin + np.array([UNIT * x, UNIT * y])

    pods.append([center1[0] - 7, center1[1] - 7,
                       center1[0] + 7, center1[1] + 7])
    pod_pos.append([x, y])

# to create tasks(find pods)
for _ in range(config.n_task):
    the_ws_idx  = np.random.choice(range(2))
    the_pod_idx = np.random.choice(range(len(pod_pos)))
    the_pod     = pod_pos[the_pod_idx]
    while the_pod in to_pos:
        the_pod_idx = np.random.choice(range(len(pod_pos)))
        the_pod     = pod_pos[the_pod_idx]
    x, y = the_pod[0], the_pod[1]
    center = origin + np.array([UNIT * x, UNIT * y])
    tos.append([center[0] - 7, center[1] - 7,
                center[0] + 7, center[1] + 7])
    to_pos.append([x, y])
    request_ws_idx.append(the_ws_idx)

# to create agvs
# the_agv = [[6, 7]]
for _ in range(config.n_agv):
    x = np.random.randint(1, 17)#(maze_width - 1))
    y = np.random.randint(1, (maze_height - 1))
    while [x, y] in wall_pos or [x, y] in from_pos:
        x = np.random.randint(1, 17)#(maze_width - 1))
        y = np.random.randint(1, (maze_height - 1))
    # x, y = the_agv[_][0], the_agv[_][1]
    center = origin + np.array([UNIT * x, UNIT * y])

    froms.append([center[0] - 7, center[1] - 7,
                  center[0] + 7, center[1] + 7])
    from_pos.append([x, y])
