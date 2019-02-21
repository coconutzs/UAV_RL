import os
import time
import ddpg_rnn_model4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *

np.set_printoptions(precision = 3, suppress = True, linewidth = 120, threshold = np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not os.path.exists(result_path): os.mkdir(result_path)
if not os.path.exists(ckpt_dir): os.mkdir(ckpt_dir)

f_result = open(result_txt, 'w')
f_result.write('### ' + exp[:-1] + ' ###\n')
print('### ' + exp[:-1] + ' ###\n')

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
        new_angle = angle + action[0]
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

def draw_location(location, x_user, y_user, title, savepath, gif=False, gif_dir='',
                  op=False, op_xpoint=0.0, op_ypoint=0.0):
    location = np.transpose(location)
    plt.figure(facecolor='w', figsize=(20, 20))
    plt.scatter(x_user, y_user, c='red', marker='x', s=150, linewidths=4)
    plt.plot(location[0], location[1], c='blue', marker='o', linewidth=3.5, markersize=7.5)
    plt.plot(location[0][-1], location[1][-1], c='blue', marker='o', markersize=12.5)
    if op:
        plt.plot(op_xpoint, op_ypoint, c='magenta', marker='o', markersize=12.5)
    plt.title(title, fontsize=30)
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(savepath)
    if gif:
        if not os.path.exists(gif_dir): os.mkdir(gif_dir)
        for i in range(MAX_EP_STEPS + 1):
            gif_path = gif_dir + 'step_%s' % str(i).zfill(3)
            plt.figure(facecolor='w', figsize=(20, 20))
            plt.scatter(x_user, y_user, c='red', marker='x', s=150, linewidths=4)
            plt.plot(op_xpoint, op_ypoint, c='magenta', marker='o', markersize=12.5)

            plt.plot(location[0, :i + 1], location[1, :i + 1], c='blue', marker='o', markersize=4.0)
            plt.plot(location[0][i], location[1][i], c='blue', marker='o', markersize=15.0)
            plt.xlim((0, 100))
            plt.ylim((0, 100))
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)

            location_r = np.array([location[0][i], location[1][i]])
            _, _, rate = get_rate(x0_user, y0_user, location_r)

            title = 'Final rate: %.3f/%.3f' % (rate, opt_rate)
            plt.title(title, fontsize=30)

            plt.savefig(gif_path)

def print_action(action_s, decimal_place, ele_perline, write_result=False):
    trans_meaning = '(%-0' + str(decimal_place[0] + 5 + decimal_place[0]/10) + 'f  %-0' + \
                              str(decimal_place[1] + 2 + decimal_place[1]/10) + 'f)'
    angle_l = np.array(np.transpose(action_s)[0] * 180 / np.pi)
    distance_l = np.transpose(action_s)[1]
    for k in range(MAX_EP_STEPS):
        angle = round(angle_l[k], decimal_place[0])
        distance = round(distance_l[k], decimal_place[1])
        print(trans_meaning % (angle, distance), end="\t")
        if write_result: f_result.write(trans_meaning % (angle, distance) + '\t')

        if (k + 1) % ele_perline == 0:
            print('')
            if write_result: f_result.write('\n')
        if (k + 1) == MAX_EP_STEPS:
            if (k + 1) % ele_perline == 0:
                print('')
                if write_result: f_result.write('\n')
            else:
                print('\n')
                if write_result: f_result.write('\n\n')

sess = tf.Session()
actor = ddpg_rnn_model4.Actor_rnn(sess, action_dim, action_bound, LR_A,
        REPLACEMENT, dense_dim = hidden_dim)
critic = ddpg_rnn_model4.Critic_rnn(sess, concat_dim, action_dim, LR_C,
        GAMMA, REPLACEMENT, actor.a, actor.a_, dense_dim = hidden_dim)
actor.add_grad_to_graph(critic.a_grads)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
M = ddpg_rnn_model4.Memory_rnn(MEMORY_CAPACITY,
                              dims=2 * time_step * (num_user + 1) + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

location_store = np.zeros([MAX_EPISODES, MAX_EP_STEPS + 1, 2])
action_store = np.zeros([MAX_EPISODES, MAX_EP_STEPS, action_dim])
strength_store = np.zeros([MAX_EP_STEPS + time_step + 1, num_user])

final_rate_store = np.zeros([MAX_EPISODES, MAX_EP_STEPS])
angle_store = np.zeros([MAX_EP_STEPS + time_step + 1, 1])
sum_rate_store = np.zeros([MAX_EPISODES])
time_store = np.zeros([MAX_EPISODES])
var_store  = np.zeros([MAX_EPISODES])

show_list = []
angle_list = np.zeros([MAX_EP_STEPS])
distance_list = np.zeros([MAX_EP_STEPS])
noAdvance_ep, stop_ep, best_episode_s, best_episode_f = 0, 0, 0, 0

x0_user = np.loadtxt(train_path)[1]
y0_user = np.loadtxt(train_path)[0]

f_result.write('### Training Users Location ###\n' )
f_result.write(str(np.array([x0_user, y0_user])) + '\n\n')

print('\n### Compute optimal location...')
time_opt_st = time.time()
opt_rate, opt_location = get_optimal_location(x0_user, y0_user)
time_opt_ed = time.time()
print('### Finish! Time: %-.2f\n' % (time_opt_ed - time_opt_st))

angle0 = 0

for i in range(MAX_EPISODES):
    location_UAV0 = np.array([10.0, 10.0])
    location = location_UAV0
    action_init = np.dot(np.random.uniform(0, 1, [time_step + 1, 1]),
                         np.array([[action_bound[0], action_bound[1]]]))
    angle = angle0

    # load train data
    # choose_loc = np.random.randint(9)
    # data_tra = np.loadtxt(test_path_list[choose_loc])[0]
    # np.random.shuffle(data_tra)
    # x0_user = data_tra[:num_user]
    # y0_user = data_tra[num_user: 2*num_user]
    # if i > SAMPLE_EPISODES:
    #     x0_user = np.loadtxt(train_path)[1]
    #     y0_user = np.loadtxt(train_path)[0]

    for i_init in range(time_step + 1):
        location, angle, _  = get_location(location, angle, action_init[i_init], action_flag)
        strength_store[i_init] = get_rate(x0_user, y0_user, location)[0]
        angle_store[i_init] = angle
    state = ((strength_store[1:time_step + 1] -
              strength_store[:time_step]) * state_gain)[np.newaxis, :]
    state_angle = angle_store[1: time_step + 1][np.newaxis, :]

    time_st = time.time()
    location_store[i][0] = location
    angle_store[0] = angle
    action = action_init[-1]

    for j in range(MAX_EP_STEPS):
        action_raw = actor.choose_action(state, state_angle)
        action[0] = np.clip(np.random.normal(action_raw[0], var), 0, 2 * np.pi)
        action[1] = np.clip(np.random.normal(action_raw[1], var), 0, action_bound[1])
        location_, angle, action = get_location(location, angle_store[j], action, action_flag)

        strength, rate, _ = get_rate(x0_user, y0_user, location)
        strength_, rate_, final_rate = get_rate(x0_user, y0_user, location_)
        strength_store[j + time_step + 1] = strength_
        state_ = ((strength_store[j + 2: j + time_step + 2] -
                  strength_store[j + 1: j + time_step + 1]) * state_gain)[np.newaxis, :]
        state_angle_ = angle_store[j + 2: j + time_step + 2][np.newaxis, :]
        reward = np.sum(state_[0][-1])

        M.store_transition(np.reshape(state, [time_step * num_user]),
                           np.reshape(state_angle, [time_step]),
                           action, reward,
                           np.reshape(state_, [time_step * num_user]),
                           np.reshape(state_angle_, [time_step]))

        if i > SAMPLE_EPISODES:
            if i % LR_decay_episode == 0:
                LR_A *= LR_decay
                LR_C *= LR_decay
            var *= var_decay
            b_Memory = M.sample(BATCH_SIZE)
            b_state = np.reshape(b_Memory[:, :time_step * num_user],
                                 [BATCH_SIZE, time_step, num_user])
            b_state_angle = np.reshape(b_Memory[:, time_step * num_user : time_step*(num_user+1)],
                                 [BATCH_SIZE, time_step, 1])
            b_action = b_Memory[:, time_step*(num_user+1) : time_step*(num_user+1)+action_dim]
            b_reward = b_Memory[:, -time_step*(num_user+1)-1 : -time_step*(num_user+1)]
            b_state_ = np.reshape(b_Memory[:, -time_step*(num_user+1) : -time_step],
                                  [BATCH_SIZE, time_step, num_user])
            b_state_angle_ = np.reshape(b_Memory[:, -time_step:],
                                 [BATCH_SIZE, time_step, 1])
            # print(b_state)
            critic.learn(b_state, b_state_angle, b_action, b_reward,
                         b_state_, b_state_angle_)
            actor.learn(b_state_, b_state_angle_)

        state = state_
        state_angle = state_angle_
        location = location_
        location_store[i][j + 1] = location
        action_store[i][j] = action
        final_rate_store[i][j] = final_rate
        angle_store[j + 1] = angle
        var_store[i] = var

    sum_rate_store[i] = np.sum(final_rate_store[i]) / MAX_EP_STEPS
    if i >= SAMPLE_EPISODES:
        if sum_rate_store[i] >= sum_rate_store[best_episode_s]:
            noAdvance_ep = 0
            best_episode_s = i
            saver.save(sess, ckpt_dir + 'sum_rate_model')
        if final_rate_store[i][-1] >= final_rate_store[best_episode_f][-1]:
            noAdvance_ep = 0
            best_episode_f = i
            saver.save(sess, ckpt_dir + 'final_rate_model')
        else:
            noAdvance_ep += 1

        if abs(final_rate_store[i][-1] - opt_rate) <= 0.5:
            draw_path_ep = 1
        else:
            draw_path_ep = 50

    time_ed = time.time()
    time_store[i] = time_ed - time_st

    if i < SAMPLE_EPISODES:
        if (i + 1) % sample_show == 0:
            # print('Train data:\t%d' % choose_loc)
            print('Episode: %s/%s (Sampling)' % (str(i + 1).zfill(6), str(MAX_EPISODES).zfill(6)),
                  '\tFinal rate: %.3f/%.3f\t  Sum rate: %.3f' % (final_rate_store[i][-1],
                                                                 opt_rate, sum_rate_store[i]),
                  '\tExplore: %.2f' % var, '\nLocation: %s → %s'
                  % (location_store[i][1], location_store[i][-1]),
                  '\tTime: %.2f' % np.sum(time_store[(i + 1) - sample_show:(i + 1)]))
            print_action(action_store[i], [2, 3], print_per_line)

    else:
        if i == SAMPLE_EPISODES:
            print('### End sampling! Time: %.2f' % np.sum(time_store))
            print('### Start training...')
        if (i + 1) % episode_show == 0:
            print('Episode: %s/%s' % (str(i + 1).zfill(6), str(MAX_EPISODES).zfill(6)),
                  '\tFinal rate: %.3f/%.3f\t  Sum rate: %.3f' % (final_rate_store[i][-1],
                                                                 opt_rate, sum_rate_store[i]),
                  '\tExplore: %.2f' % var, '\nLocation: %s → %s'
                  % (location_store[i][1], location_store[i][-1]),
                  '\tTime: %-.2f' % np.sum(time_store[(i + 1) - episode_show:(i + 1)]),
                  '\tNo advance episodes:\t%-d' % noAdvance_ep)
            print_action(action_store[i], [2, 3], print_per_line)

        if (i + 1) % draw_path_ep == 0:
            draw_location(location_store[i], x0_user, y0_user,
                          'Final rate: %.3f/%.3f    Sum rate: %.3f' % (final_rate_store[i][-1],
                                                                       opt_rate, sum_rate_store[i]),
                          result_path + 'UAVPath_epsiode%s.jpg' % str(i + 1).zfill(6),
                          op=True, op_xpoint=opt_location[0], op_ypoint=opt_location[1])

        if (i + 1) % draw_reward_ep == 0:
            plt.figure(facecolor='w', figsize=(20, 20))
            plt.plot(np.arange((i + 1) - SAMPLE_EPISODES),
                     sum_rate_store[SAMPLE_EPISODES:i + 1], linewidth=3.5)
            plt.ylim((y_show_min, y_show_max))
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.savefig(result_path + 'sum_rate_epsiode%s.jpg' % str(i + 1).zfill(6))

            plt.figure(facecolor='w', figsize=(20, 20))
            plt.plot(np.arange((i + 1) - SAMPLE_EPISODES),
                     final_rate_store[SAMPLE_EPISODES:i + 1, -1], linewidth=3.5)
            plt.ylim((y_show_min, y_show_max))
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.savefig(result_path + 'final_rate_epsiode%s.jpg' % str(i + 1).zfill(6))

    stop_ep = i
    if noAdvance_ep >= early_stop_ep:
        print('### ' + exp[:-1])
        print('### Early stop at episode %s/%s!' % (str(i - early_stop_ep + 1).zfill(6),
                                                    str(MAX_EPISODES).zfill(6)))
        f_result.write('### Early stop at episode %s/%s!\n' % (str(i - early_stop_ep + 1).zfill(6),
                                                               str(MAX_EPISODES).zfill(6)))
        break

plt.figure(facecolor='w', figsize=(20, 20))
plt.plot((best_episode_f - SAMPLE_EPISODES) * np.ones([int(final_rate_store[best_episode_f][-1] + 5)]),
         np.arange(int(final_rate_store[best_episode_f][-1] + 5)), color='red', linewidth=3.5)
plt.plot(np.arange(stop_ep - SAMPLE_EPISODES), final_rate_store[SAMPLE_EPISODES + 1:stop_ep + 1, -1],
         linewidth=3.5)
plt.title('Final rate: %.3f/%.3f  Sum rate: %.3f' % (final_rate_store[best_episode_f][-1],
                                                     opt_rate, sum_rate_store[best_episode_f]), fontsize=30)
plt.xlim((0, stop_ep - SAMPLE_EPISODES - early_stop_ep + 500))
plt.ylim((y_show_min, int(final_rate_store[best_episode_f][-1] + 5)))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(result_path + 'End_final_rate_epsiode%s.jpg'
            % str(stop_ep - early_stop_ep + 1).zfill(6))

plt.figure(facecolor='w', figsize=(20, 20))
plt.plot((best_episode_s - SAMPLE_EPISODES) * np.ones([int(sum_rate_store[best_episode_s] + 100)]),
         np.arange(int(sum_rate_store[best_episode_s] + 100)), color='red', linewidth=3.5)
plt.plot(np.arange(stop_ep - SAMPLE_EPISODES), sum_rate_store[SAMPLE_EPISODES + 1:stop_ep + 1],
         linewidth=3.5)
plt.title('Final rate: %.3f/%.3f  Sum rate: %.3f'
          % (final_rate_store[best_episode_s][-1],
             opt_rate, sum_rate_store[best_episode_s]), fontsize=30)
plt.xlim((0, stop_ep - SAMPLE_EPISODES - early_stop_ep + 500))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim((y_show_min, int(sum_rate_store[best_episode_s] + 10)))
plt.savefig(result_path + 'End_sum_rate_epsiode%s.jpg' % str(stop_ep - early_stop_ep + 1).zfill(6))

angle_list_f = np.array(np.transpose(action_store[best_episode_f])[0] * 180 / np.pi)
angle_list_s = np.array(np.transpose(action_store[best_episode_s])[0] * 180 / np.pi)
distance_list_f = np.transpose(action_store[best_episode_f])[1]
distance_list_s = np.transpose(action_store[best_episode_s])[1]

print('### End training! Total Time:\t%-.2f' % np.sum(time_store))
f_result.write('### End training! Total Time:\t%-.2f\n' % np.sum(time_store))
draw_location(location_store[best_episode_f], x0_user, y0_user,
              'Final rate: %.3f/%.3f    Sum rate: %.3f' % (final_rate_store[best_episode_f][-1],
                                                           opt_rate, sum_rate_store[best_episode_f]),
              result_path + 'Best_f_epsiode%s.jpg' % (str(best_episode_f + 1).zfill(6)),
              gif=True, gif_dir=result_path + 'Best_f_gif/', op=True,
              op_xpoint=opt_location[0], op_ypoint=opt_location[1])
draw_location(location_store[best_episode_s], x0_user, y0_user,
              'Final rate: %.3f/%.3f    Sum rate: %.3f' % (final_rate_store[best_episode_s][-1],
                                                           opt_rate, sum_rate_store[best_episode_s]),
              result_path + 'Best_s_epsiode_s%s.jpg' % (str(best_episode_s + 1).zfill(6)),
              gif=True, gif_dir=result_path + 'Best_s_gif/', op=True,
              op_xpoint=opt_location[0], op_ypoint=opt_location[1])

result_str = '### Best Episode(Final rate): %s/%s' \
             % (str(best_episode_f + 1).zfill(6), str(MAX_EPISODES).zfill(6)) + \
             '\tFinal rate: %.3f/%.3f\t  Sum rate: %.3f' % (final_rate_store[best_episode_f][-1],
                                                            opt_rate, sum_rate_store[best_episode_f]) + \
             '\tLocation: %s → %s' \
             % (location_store[best_episode_f][1], location_store[best_episode_f][-1]) + \
             '\tVar: %.2f' % var_store[best_episode_f]
print(result_str)
f_result.write(result_str + '\n')
print_action(action_store[best_episode_f], [2, 3], print_per_line, write_result=True)

result_str = '### Best Episode(Sum rate): %s/%s' \
             % (str(best_episode_s + 1).zfill(6), str(MAX_EPISODES).zfill(6)) + \
             '\tFinal rate: %.3f/%.3f\t  Sum rate: %.3f' % (final_rate_store[best_episode_s][-1],
                                                            opt_rate, sum_rate_store[best_episode_s]) + \
             '\tLocation: %s → %s' \
             % (location_store[best_episode_s][1], location_store[best_episode_s][-1]) + \
             '\tVar: %.2f' % var_store[best_episode_s]
print(result_str)
f_result.write(result_str + '\n')
print_action(action_store[best_episode_s], [2, 3], print_per_line, write_result=True)
np.savetxt(result_path + 'final_rate', final_rate_store[SAMPLE_EPISODES + 1:stop_ep + 1, -1])
np.savetxt(result_path + 'sum_rate', sum_rate_store[SAMPLE_EPISODES + 1:stop_ep + 1])
f_result.close()
