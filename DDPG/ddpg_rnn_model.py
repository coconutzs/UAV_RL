import tensorflow as tf
import numpy as np
from config import *

class RNN(object):
    def __init__(self, hidden_dim, s, s_, scope_name):
        self.S  = s
        self.S_ = s_
        self.scope_name = scope_name
        self.hidden_dim = hidden_dim
        with tf.variable_scope(self.scope_name):
            self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units = self.hidden_dim)
            _,  self.output  = tf.nn.dynamic_rnn(self.cell, self.S,  dtype = tf.float32)
            _,  self.output_ = tf.nn.dynamic_rnn(self.cell, self.S_, dtype = tf.float32)

class FC(object):
    def __init__(self, hidden_dim, s, s_, scope_name):
        self.S  = s
        self.S_ = s_
        self.scope_name = scope_name
        self.hidden_dim = hidden_dim
        with tf.variable_scope(self.scope_name):
            self.output  = tf.layers.dense(self.S, self.hidden_dim, activation=tf.nn.relu,
                                          kernel_initializer=init_w, bias_initializer=init_b,
                                          name='state', trainable=True)
            self.output_ = tf.layers.dense(self.S_, self.hidden_dim, activation=tf.nn.relu,
                                          kernel_initializer=init_w, bias_initializer=init_b,
                                          name='state_', trainable=True)

class Actor_rnn(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement, dense_dim):
        self.S  = state
        self.S_ = state_
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a  = self._build_net(self.S, dense_dim, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(self.S_, dense_dim, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t +
                                           self.replacement['tau'] * e) for t, e in zip(self.t_params,
                                                                                        self.e_params)]

    def _build_net(self, s, dense_dim, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, dense_dim, activation=tf.nn.relu,
                            kernel_initializer=init_w, bias_initializer=init_b, name='l1_dense',
                            trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid,
                                          kernel_initializer=init_w, bias_initializer=init_b,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
                # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s1, s2):   # batch update
        self.sess.run(self.train_op, feed_dict = {p_state_s: s1, p_state_a: s2})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s1, s2):
        # s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={p_state_s: s1, p_state_a: s2})[0]  # single action
        # return a_batch

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys,
            # so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=10 * self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

class Critic_rnn(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma,
                 replacement, a, a_, dense_dim):
        self.S  = state
        self.S_ = state_
        self.R  = tf.placeholder(tf.float32)
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q  = self._build_net(self.S, self.a, dense_dim, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(self.S_, a_, dense_dim, 'target_net', trainable=False)
            # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]
            # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t +
                                    self.replacement['tau'] * e) for t, e in zip(self.t_params,
                                                                                 self.e_params)]

    def _build_net(self, s, a, hidden_dim, scope, trainable):
        with tf.variable_scope(scope):
            with tf.variable_scope('l1'):
                n_l1 = hidden_dim
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w,
                                       trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w,
                                       trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s1, s2, a, r, s1_, s2_):
        self.sess.run(self.train_op, feed_dict={p_state_s: s1, p_state_a: s2, self.a: a,
                                                self.R: r, p_state_s_: s1_, p_state_a_: s2_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

class Memory_rnn(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s1, s2, a, r, s1_, s2_):
        transition = np.hstack((s1, s2, a, [r], s1_, s2_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

if USE_RNN:
    state_s = RNN(30, p_state_s, p_state_s_, 'rnn_s')
    state_a = RNN(10, p_state_a, p_state_a_, 'rnn_a')
else:
    reshape_s  = tf.reshape(p_state_s,  [-1, time_step * num_user])
    reshape_s_ = tf.reshape(p_state_s_, [-1, time_step * num_user])
    reshape_a  = tf.reshape(p_state_a,  [-1, time_step])
    reshape_a_ = tf.reshape(p_state_a,  [-1, time_step])

    state_s = FC(30, reshape_s, reshape_s_, 'fc_s')
    state_a = FC(10, reshape_a, reshape_a_, 'fc_a')

state  = tf.concat([state_s.output,  state_a.output],  1)
state_ = tf.concat([state_s.output_, state_a.output_], 1)
