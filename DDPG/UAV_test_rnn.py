import os
import ddpg_rnn_model4
import time
import numpy as np
from config import *
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(precision = 5, suppress = True, linewidth = 400, threshold = np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# exp = 'Exp4_16_4e-7/'
flight_rate = 2.0
MAX_EP_STEPS = 95
STAGE1_STEPS = 95
STAGE2_STEPS = MAX_EP_STEPS - STAGE1_STEPS
TEST_EPISODE = 30
model = ['sum_rate_model', 'final_rate_model']
model_restore = model[1]

model_restore_path = ckpt_dir + model_restore
test_dir = result_path + 'test_%s_loc_48_step%s_rate%s/' % \
           (model_restore[0], str(STAGE1_STEPS), str(flight_rate))
if not os.path.exists(test_dir): os.mkdir(test_dir)
test_txt = test_dir + 'test_result.txt'

start_str = '\n### Test(%s) Result\n' % model_restore
f_test = open(test_txt, 'w')

def angle_clip(angle):
    clip_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return  clip_angle

def RNN_state(input, hidden_dim):
    input = tf.cast(input, tf.float32)
    with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units = hidden_dim)
        output, h_end = tf.nn.dynamic_rnn(cell, input, dtype = tf.float32)
    return h_end

def get_rate(x_user, y_user, location_UAV):
    distance_sq = (x_user - location_UAV[0]) ** 2 + \
               (y_user - location_UAV[1]) ** 2 + height ** 2
    signal_strength = reference * power / distance_sq
    rate = np.log(1 + signal_strength)
    rate_sum = np.sum(rate)
    return signal_strength, rate, rate_sum

def get_location(location, angle, action, flag):
    new_angle = 0
    if flag == 're':
        new_angle = (angle + action[0]) % action_bound[0]
    if flag == 'ab':
        new_angle = action[0]

    distance = action[1]
    new_action = np.zeros([2])
    new_location = location + distance * np.array([np.cos(new_angle), np.sin(new_angle)])
    while not(np.max(new_location) < 100 and np.min(new_location)) > 0:
        new_angle = (new_angle + np.clip(np.random.normal(0.75 * np.pi, 0.5 * var)
                                         , 0.5 * np.pi, 1.0 * np.pi)) % action_bound[0]
        new_location = location + distance * np.array([np.cos(new_angle), np.sin(new_angle)])

    if flag == 're':
        new_action[0] = (new_angle + action_bound[0] - angle) % action_bound[0]
    if flag == 'ab':
        new_action[0] = new_angle
    new_action[1] = distance

    return new_location, new_angle, new_action

def get_optimal_location(x_user, y_user):
    rate_list = np.zeros([50000])
    x_user, y_user = x_user.tolist(), y_user.tolist()
    location_UAV_op = tf.Variable(np.random.uniform(50, 70, size = [2, 1]), dtype=tf.float32)
    distance_sq = tf.reduce_sum(tf.pow(tf.subtract(location_UAV_op, [x_user, y_user]), 2), 0) \
                  + height ** 2
    rate_sum = -tf.reduce_sum(tf.log(1 + reference * power / distance_sq))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(rate_sum)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            sess.run(train_step)
            rate_list[i + 1] = sess.run(rate_sum)
            if (abs(rate_list[i + 1] - rate_list[i]) <= 1e-7):
                break
        opt_rate = sess.run(-1 * rate_sum)
        opt_location = np.squeeze(sess.run(location_UAV_op))
    return opt_rate, opt_location

def draw_location_test(location, x_user_1, y_user_1, x_user_2, y_user_2, title, savepath,
                  op_xpoint, op_ypoint, op = False):
    location = np.transpose(location)
    plt.figure(facecolor = 'w', figsize = (20, 20))
    plt.scatter(x_user_1, y_user_1, c = 'red', marker = 'x', s = 150, linewidths = 4)
    # plt.scatter(x_user_2, y_user_2, c = 'red', marker = 'x', s = 150, linewidths = 4)
    plt.plot(location[0][:STAGE1_STEPS + 1], location[1][:STAGE1_STEPS + 1],
             c = 'magenta', marker = 'o', linewidth = 3.5, markersize = 5)
    # plt.plot(location[0][STAGE1_STEPS + 1:], location[1][STAGE1_STEPS + 1:],
    #          c = 'blue', marker = 'o', linewidth = 3.5, markersize = 5)
    plt.plot(location[0][STAGE1_STEPS], location[1][STAGE1_STEPS],
             c = 'magenta', marker = 'o', markersize = 15)
    # plt.plot(location[0][MAX_EP_STEPS + 1], location[1][MAX_EP_STEPS + 1],
    #          c = 'blue', marker = 'o', markersize = 15)
    if op:
        plt.plot(op_xpoint[0], op_ypoint[0], c = 'orange', marker = 'o', markersize = 15)
        # plt.plot(op_xpoint[1], op_ypoint[1], c = 'orange', marker = 'o', markersize = 15)
    plt.title(title, fontsize = 30)
    plt.xlim((0, 105))
    plt.ylim((0, 105))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(savepath)

def print_action(action_s, decimal_place, ele_perline, write_result=False):
    trans_meaning = '(%-0' + str(decimal_place[0] + 4 + decimal_place[0]/10) + 'f  %-0' + \
                              str(decimal_place[1] + 2 + decimal_place[1]/10) + 'f)'
    angle_l = np.array(np.transpose(action_s)[0] * 180 / np.pi)
    distance_l = np.transpose(action_s)[1]
    for k in range(len(angle_l)):
        angle = round(angle_l[k], decimal_place[0])
        distance = round(distance_l[k], decimal_place[1])
        print(trans_meaning % (angle, distance), end="\t\t")
        if write_result: f_test.write(trans_meaning % (angle, distance) + '\t\t')

        if (k + 1) % ele_perline == 0:
            print('')
            if write_result: f_test.write('\n')
        if (k + 1) == MAX_EP_STEPS:
            if (k + 1) % ele_perline == 0:
                print('')
                if write_result: f_test.write('\n')
            else:
                print('\n')
                if write_result: f_test.write('\n\n')

sess = tf.Session()
actor = ddpg_rnn_model4.Actor_rnn(sess, action_dim, action_bound, LR_A,
        REPLACEMENT, dense_dim = hidden_dim)
critic = ddpg_rnn_model4.Critic_rnn(sess, concat_dim, action_dim, LR_C,
        GAMMA, REPLACEMENT, actor.a, actor.a_, dense_dim = hidden_dim)
actor.add_grad_to_graph(critic.a_grads)

saver = tf.train.Saver()
saver.restore(sess, model_restore_path)

location_store = np.zeros([TEST_EPISODE, MAX_EP_STEPS + 1, 2])
action_store = np.zeros([TEST_EPISODE, MAX_EP_STEPS, action_dim])
strength_store = np.zeros([MAX_EP_STEPS + time_step + 1, num_user])

final_rate_store = np.zeros([TEST_EPISODE, MAX_EP_STEPS])
sum_rate_store = np.zeros([TEST_EPISODE, 2])
angle_store = np.zeros([MAX_EP_STEPS + time_step + 1])
angle0 = 0

time_st = time.time()
f_test.write(start_str)
print(start_str)

for i in range(TEST_EPISODE):
    x0_user_stage1 = np.loadtxt(test_path_list[3])[i,:num_user]
    y0_user_stage1 = np.loadtxt(test_path_list[7])[i, num_user:2 * num_user]
    x0_user_stage2 = np.loadtxt(test_path_50_70)[i,:num_user]
    y0_user_stage2 = np.loadtxt(test_path_80_00)[i,:num_user]

    location_user_stage1 = np.array([x0_user_stage1, y0_user_stage1])
    location_user_stage2 = np.array([x0_user_stage2, y0_user_stage2])
    opt_rate_stage1, opt_location_stage1 = get_optimal_location(x0_user_stage1, y0_user_stage1)
    opt_rate_stage2, opt_location_stage2 = get_optimal_location(x0_user_stage2, y0_user_stage2)

    location_UAV0 = np.array([10.0, 10.0])
    location = location_UAV0
    action_init = np.dot(np.random.uniform(0, 1, [time_step + 1, 1]),
                         np.array([[action_bound[0], action_bound[1]]]))
    angle = angle0

    for i_init in range(time_step + 1):
        location, angle, _  = get_location(location, angle, action_init[i_init], action_flag)
        strength_store[i_init] = get_rate(x0_user_stage1, y0_user_stage1, location)[0]
        angle_store[i_init] = angle
    state = ((strength_store[1:time_step + 1] -
              strength_store[:time_step]) * state_gain)[np.newaxis, :]
    state_angle = angle_store[1: time_step + 1][np.newaxis, :][:, :, np.newaxis]

    location_store[i][0] = location
    angle_store[0] = angle
    action = action_init[-1]

    for j in range(MAX_EP_STEPS):
        if j < STAGE1_STEPS:
            stage_flag = 1
        else:
            stage_flag = 2

        x0_user, y0_user = np.zeros([concat_dim]), np.zeros([concat_dim])
        if stage_flag == 1:
            x0_user, y0_user = x0_user_stage1, y0_user_stage1
        if stage_flag == 2:
            x0_user, y0_user = x0_user_stage2, y0_user_stage2

        action = actor.choose_action(state, state_angle)
        location_, angle, action = get_location(location, angle_store[j], action, action_flag)

        strength, rate, _ = get_rate(x0_user, y0_user, location)
        strength_, rate_, final_rate = get_rate(x0_user, y0_user, location_)
        strength_store[j + time_step + 1] = strength_
        state_ = ((strength_store[j + 2: j + time_step + 2] -
                  strength_store[j + 1: j + time_step + 1]) * state_gain)[np.newaxis, :]
        state_angle_ = angle_store[j + 2: j + time_step + 2][np.newaxis, :][:, :, np.newaxis]
        reward = np.sum(state_[0][-1])

        state = state_
        state_angle = state_angle_
        location = location_
        location_store[i][j + 1] = location
        action_store[i][j] = action
        final_rate_store[i][j] = final_rate
        angle_store[j + 1] = angle

    sum_rate_store[i][0] = np.sum(final_rate_store[i][:STAGE1_STEPS+2]) / STAGE1_STEPS
    sum_rate_store[i][1] = np.sum(final_rate_store[i][STAGE1_STEPS+2:]) / STAGE2_STEPS

    f_test.write('### Episode %s/%s\n' % (str(i+1).zfill(3), str(TEST_EPISODE).zfill(3)))
    print('### Episode %s/%s' % (str(i+1).zfill(3), str(TEST_EPISODE).zfill(3)))

    f_test.write('Stage1\n' + str(location_user_stage1) + '\n')
    result_str = 'Final rate: %.3f/%.3f\t  Sum rate: %.3f' % \
                 (final_rate_store[i][STAGE1_STEPS-1], opt_rate_stage1, sum_rate_store[i][0])+\
                 '\tExplore: %.2f' % var+\
                 '\tLocation: %s → %s (%s)' % \
                 (location_store[i][1], location_store[i][STAGE1_STEPS], opt_location_stage1)
    print('Stage1\n' + result_str)
    f_test.write(result_str + '\n')
    print_action(action_store[i,:STAGE1_STEPS], [2, 3], print_per_line, True)

    draw_location_test(location_store[i], x0_user_stage1, y0_user_stage1, x0_user_stage2, y0_user_stage2,
                  'Stage1    Final rate: %.3f/%.3f    Sum rate: %.3f' %
                  (final_rate_store[i][STAGE1_STEPS - 1], opt_rate_stage1, sum_rate_store[i][0])
                  , test_dir + 'Test_epsiode%s.jpg' % str(i + 1).zfill(3),
                  op_xpoint = [opt_location_stage1[0], opt_location_stage2[0]],
                  op_ypoint = [opt_location_stage1[1], opt_location_stage2[1]],
                  op = True
                  )

    x_store = np.transpose(location_store[i])[0]
    y_store = np.transpose(location_store[i])[1]

    f_test.write('x_uav:\t')
    for i_x in range(np.shape(x_store)[0]):
        f_test.write(str(x_store[i_x]) + ' ')
    f_test.write('\n')

    f_test.write('y_uav:\t')
    for i_y in range(np.shape(y_store)[0]):
        f_test.write(str(x_store[i_y]) + ' ')
    f_test.write('\n\n')

time_ed = time.time()
print('End! Time:\t%.3f' % (time_ed - time_st))
f_test.write('End! Time:\t%.3f\n' % (time_ed - time_st))
f_test.close()