import tensorflow as tf
import numpy as np

exp = 'Exp4_16_25e-8/'
result_path = 'Result/' + exp
result_txt = result_path + 'Result_' +  exp[:-1] + '.txt'
ckpt_dir = result_path + 'ckpt/'
data_dir = 'data/'
train_path = data_dir + 'tra_user15_loc88'
test_path_00_20 = data_dir + 'tes_loc00_20'
test_path_10_30 = data_dir + 'tes_loc10_30'
test_path_20_40 = data_dir + 'tes_loc20_40'
test_path_30_50 = data_dir + 'tes_loc30_50'
test_path_40_60 = data_dir + 'tes_loc40_60'
test_path_50_70 = data_dir + 'tes_loc50_70'
test_path_60_80 = data_dir + 'tes_loc60_80'
test_path_70_90 = data_dir + 'tes_loc70_90'
test_path_80_00 = data_dir + 'tes_loc80_00'
test_path_list = [test_path_00_20, test_path_10_30, test_path_20_40,
                  test_path_30_50, test_path_40_60, test_path_50_70,
                  test_path_60_80, test_path_70_90, test_path_80_00]

RENDER = False
TS_STATE = True
USE_RNN = True
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

height = 30
power = 10
reference = 1e4
num_user = 15
flight_rate = 2.0
MAX_EP_STEPS = 75
state_gain = 30

time_step = 3
action_dim = 2
s_dim = 30
a_dim = 10
concat_dim = s_dim + a_dim
hidden_dim = 64
action_flag = ['ab', 're'][0]
action_bound = [2 * np.pi, flight_rate]
var = 2.0
var_decay = 0.999

SAMPLE_EPISODES = 15000
POINT_SAMPLE_EPISODES = 50
NUM_LOC_UAV = int(SAMPLE_EPISODES/POINT_SAMPLE_EPISODES)
MAX_EPISODES = SAMPLE_EPISODES + 15000

LR_A = 2.5e-7   # learning rate for actor
LR_C = 2.5e-7   # learning rate for critic
LR_decay = 1.0
LR_decay_episode = 1000
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=50, rep_iter_c=50)
][1]            # you can try different target replacement strategies
MEMORY_CAPACITY = MAX_EP_STEPS * SAMPLE_EPISODES
# MEMORY_CAPACITY = 1500
BATCH_SIZE = 64

episode_show = 20
sample_show = 100
draw_path_ep = 30
draw_reward_ep = 100
early_stop_ep = 1500
print_per_line = 5
y_show_max = 75
y_show_min = 30

p_state_s  = tf.placeholder(tf.float32, [None, time_step, num_user])
p_state_s_ = tf.placeholder(tf.float32, [None, time_step, num_user])
p_state_a  = tf.placeholder(tf.float32, [None, time_step, 1])
p_state_a_ = tf.placeholder(tf.float32, [None, time_step, 1])
p_reward   = tf.placeholder(tf.float32, [None, 1], name='r')

init_w = tf.random_normal_initializer(0., 0.1)
init_b = tf.constant_initializer(0.1)
