
UNIT = 20   # pixels
MAP_WIDTH  = 23 #22 # 5
MAP_HEIGHT = 23 #11 # 5
n_agv  = 1
n_task = 1

EPISODE_MAX = 50000
PRETRAINED  = None #"3Channel-fcn-1car-wall-randomGoal-17000.pkl" # None

lr           = 1e-3
gamma        = 0.9
batch_size   = 64
max_mem_size = 10000
c_step       = 1000 # 800
epsilon      = 0.6 # 0.5
eps_end      = 0.1
eps_dec      = 20000 + 10000
start_eps_dec = 1000

clip = 500 # 1000 # None
optim = 'adam'

wait_penalty  = -80 #-0.1
step_penalty  = -0.05
crash_penalty = -100
goal_reward   =  300
terminal_wall = False
one_goal      = False #True
color = 'red' # 'blue'
